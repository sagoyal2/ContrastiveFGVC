"""Extraction CLI (§8, §10). Writes the pooled ``features/`` tree.

    python -m src.extract --model llava15_7b --dataset cub --split train
    python -m src.extract --model llava15_7b --dataset cub --split train --sweep
    python -m src.extract --model clip_l14_336 --dataset cub --split test

**The one rule** (§8): pooling happens inside this loop, before anything touches
disk. ``[576, 4096]`` per image becomes ``[4096]`` — a 576x reduction — which is
what turns 113 GB of per-token cache into 36 GB of pooled features.

Pooling is per example and length-aware (``seq_len``), so right-padding in a
ragged batch can never leak a pad token into an average or make ``final`` point
at padding (§13).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .data.registry import get_dataset, load_dataset_config, subset_classes
from .data.templates import PROBE_PROMPT
from .models.base import pool_tokens
from .models.registry import build_model, load_model_config
from .paths import ensure_dirs, feature_dir, use_storage_hf_cache


def _batches(n: int, size: int):
    for i in range(0, n, size):
        yield i, min(i + size, n)


class FeatureWriter:
    """Accumulates pooled vectors in RAM, writes one fp16 ``.npy`` per key.

    CUB is 11.8k images x ~34 keys x 4 KB = well under a GB, so buffering is
    simpler and faster than incremental writes. A dataset that does not fit
    should stream to ``np.lib.format.open_memmap`` instead.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.buf: Dict[str, List[np.ndarray]] = {}

    def add(self, key: str, x: torch.Tensor) -> None:
        self.buf.setdefault(key, []).append(x.detach().to(torch.float16).cpu().numpy())

    def flush(self, labels: np.ndarray, manifest: dict) -> Dict[str, tuple]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        shapes = {}
        for key, chunks in self.buf.items():
            arr = np.concatenate(chunks, axis=0)
            assert arr.shape[0] == labels.shape[0], (
                f"{key}: {arr.shape[0]} rows vs {labels.shape[0]} labels"
            )
            np.save(self.out_dir / f"{key}.npy", arr)
            shapes[key] = tuple(arr.shape)
        np.save(self.out_dir / "labels.npy", labels)
        manifest["keys"] = {k: list(v) for k, v in shapes.items()}
        manifest["n"] = int(labels.shape[0])
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return shapes


def extract_encoder(model, ds, args, out_dir: Path) -> None:
    """T0-c / T0.avg / T0.cls for a bare encoder (§5). No LLM pass."""
    writer = FeatureWriter(out_dir)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    t_start = time.time()

    for lo, hi in _batches(n, args.batch_size):
        images = [ds[i][0] for i in range(lo, hi)]
        out = model.encode_images(images)
        writer.add("T0c", out["t0c"])
        writer.add("T0.avg", out["t0"].mean(1))
        if out["t0_cls"] is not None:
            writer.add("T0.cls", out["t0_cls"])
        if lo % (args.batch_size * 20) == 0:
            done = hi
            rate = done / max(time.time() - t_start, 1e-6)
            print(f"  {done}/{n}  {rate:.1f} img/s", flush=True)

    labels = ds.labels[:n]
    manifest = _manifest(model, ds, args, extra={"prompt": None, "kind": "encoder"})
    shapes = writer.flush(labels, manifest)
    print(f"wrote {out_dir}")
    for k, s in sorted(shapes.items()):
        print(f"   {k:<16} {s}")


def extract_vlm(model, ds, args, out_dir: Path) -> None:
    """All taps for a VLM, one forward pass per batch (§5)."""
    cfg = load_model_config(args.model)
    sweep_layers = tuple(cfg.get("sweep_layers", ())) if args.sweep else ()

    writer = FeatureWriter(out_dir)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    t_start = time.time()
    span_widths, first_batch_note = set(), {}

    for lo, hi in _batches(n, args.batch_size):
        images = [ds[i][0] for i in range(lo, hi)]
        taps = model.forward_taps(images, PROBE_PROMPT, layers=sweep_layers)
        b = taps.batch_size()
        m = taps.t2.shape[1]
        span_widths.add(m)

        # T0 / T1 are the full patch grid: every position is real, so no length
        # masking is needed. T2 likewise (it is already sliced to the span).
        writer.add("T0.avg", taps.t0.mean(1))
        if taps.t0_cls is not None:
            writer.add("T0.cls", taps.t0_cls)
        writer.add("T1.avg", taps.t1.mean(1))
        writer.add("T1.final", taps.t1[:, -1])
        writer.add("T2.avg", taps.t2.mean(1))
        writer.add("T2.final", taps.t2[:, -1])
        # T3 is the padded full sequence, so it *does* need seq_len.
        #
        # In LLaDA-V's masked mode the sequence carries k appended [MASK] tokens.
        # Pooling T3 over the raw seq_len would average them in and would make
        # "T3.final" the last MASK rather than the last prompt token — the same key
        # name meaning a different tensor than in clean mode, which is precisely how
        # an incomparable number gets into a table. So T3.* is always the
        # prompt+image sequence, and the answer span is reported only as Tmask.*.
        k_mask = taps.extra["mask_states"].shape[1] if "mask_states" in taps.extra else 0
        t3_len = [n - k_mask for n in taps.seq_len]
        assert all(n > 0 for n in t3_len), f"mask span {k_mask} >= sequence length"
        writer.add("T3.avg", pool_tokens(taps.t3, "avg", t3_len))
        writer.add("T3.final", pool_tokens(taps.t3, "final", t3_len))
        # §11 check 2: position 0 must probe near chance. Cheap to store, and it
        # is the check that catches slicing the wrong tensor entirely.
        writer.add("T3.pos0", taps.t3[:, 0])

        for k, h in taps.mid.items():
            writer.add(f"mid.L{k}.avg", h.mean(1))
            writer.add(f"mid.L{k}.final", h[:, -1])

        # §9 masked mode: the answer-span states. `Tmask.avg` is the mean over the
        # k mask positions, `Tmask.first` the first one — the position the decoder
        # commits to earliest. These are the on-policy tap for a diffusion decoder
        # and have no LLaVA analogue, so they are reported separately.
        if "mask_states" in taps.extra:
            ms = taps.extra["mask_states"]
            writer.add("Tmask.avg", ms.mean(1))
            writer.add("Tmask.first", taps.extra["mask_first"])
            writer.add("Tmask.last", ms[:, -1])

        if not first_batch_note:
            first_batch_note = {
                "m": int(m),
                "img_span_first": list(taps.img_span[0]),
                "seq_len_first": int(taps.seq_len[0]),
                "d_vis": int(taps.t0.shape[-1]),
                "d_llm": int(taps.t3.shape[-1]),
            }
            print(f"  span={taps.img_span[0]} m={m} seq_len={taps.seq_len[0]} "
                  f"d_llm={taps.t3.shape[-1]}", flush=True)

        if lo % (args.batch_size * 10) == 0:
            rate = hi / max(time.time() - t_start, 1e-6)
            eta = (n - hi) / max(rate, 1e-6) / 60
            print(f"  {hi}/{n}  {rate:.1f} img/s  eta {eta:.1f} min", flush=True)

    assert len(span_widths) == 1, f"image-token count varied across batches: {span_widths}"
    labels = ds.labels[:n]
    manifest = _manifest(model, ds, args, extra={
        "prompt": PROBE_PROMPT,
        "prompt_sha1": hashlib.sha1(PROBE_PROMPT.encode()).hexdigest(),
        "kind": "vlm",
        "sweep_layers": list(sweep_layers),
        **first_batch_note,
    })
    shapes = writer.flush(labels, manifest)
    print(f"wrote {out_dir}  ({time.time() - t_start:.0f}s)")
    for k, s in sorted(shapes.items()):
        print(f"   {k:<16} {s}")


def _manifest(model, ds, args, extra: dict) -> dict:
    d = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "num_classes": ds.num_classes,
        "batch_size": args.batch_size,
        "torch": torch.__version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **model.manifest(),
    }
    d.update(extra)
    return d


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="first N images — SMOKE TESTS ONLY. The loaders emit images in "
                        "class order, so this is a class prefix, not a sample.")
    p.add_argument("--classes", type=int, default=None,
                   help="keep all images of N randomly chosen classes (§7). The correct "
                        "way to shrink a 1000-way set; --limit is not.")
    p.add_argument("--sweep", action="store_true", help="also write the §8.3B layer sweep")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llada-mode", default=None, choices=["clean", "masked"],
                   help="§9. 'clean' is one forward pass with no [MASK] in the answer "
                        "span — structurally comparable to LLaVA. 'masked' appends k "
                        "[MASK] tokens and taps their positions, which is the state "
                        "LLaDA-V actually decodes from but has no LLaVA analogue.")
    p.add_argument("--llada-k", type=int, default=None,
                   help="number of [MASK] tokens in the answer span (§9)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    use_storage_hf_cache()
    ensure_dirs()

    mcfg = load_model_config(args.model)
    if args.batch_size is None:
        args.batch_size = int(mcfg.get("batch_size", 32))

    ds_tag = args.dataset if args.classes is None else f"{args.dataset}_c{args.classes}"
    # A masked-mode run taps different tensors under a different input; writing it
    # into the clean-mode tree would silently replace features the clean numbers
    # were computed from.
    model_tag = args.model
    if args.llada_mode == "masked":
        model_tag = f"{args.model}_masked{args.llada_k or ''}"
    out_dir = feature_dir(model_tag, ds_tag, args.split)
    if (out_dir / "manifest.json").exists() and not args.overwrite:
        print(f"{out_dir} already populated; pass --overwrite to redo")
        return

    overrides = {}
    if args.llada_mode is not None:
        overrides["llada_mode"] = args.llada_mode
    if args.llada_k is not None:
        overrides["llada_k"] = args.llada_k

    print(f"building {args.model} ...", flush=True)
    model = build_model(args.model, device=args.device, **overrides)

    ds = get_dataset(args.dataset, args.split)
    if args.classes is not None:
        ds = subset_classes(ds, args.classes)
    print(f"{args.dataset}/{args.split}: {len(ds)} images, {ds.num_classes} classes", flush=True)
    if mcfg["kind"] == "encoder":
        extract_encoder(model, ds, args, out_dir)
    else:
        extract_vlm(model, ds, args, out_dir)


if __name__ == "__main__":
    main()
