"""Experiment 1 — baseline generative accuracy (§6.1).

    python -m src.evaluate_generative --model llava15_7b --dataset cub
    python -m src.evaluate_generative --model clip_l14_336 --dataset cub   # zero-shot

Two different things share this CLI because they occupy the same row of Table 1:

* **VLMs** generate open-endedly (greedy, 32 new tokens) and the text is pushed
  through the §6.1 matching cascade -> strict / lenient / %unmatched.
* **Encoders** do standard contrastive zero-shot over the T0-c space with the
  dataset's template ensemble. There is no generation and no matcher, so only one
  number exists; it is reported in the strict column with the others blank.

Raw generations are always written next to the summary. The matcher is the part
most likely to be revised, and re-running it on saved text costs seconds where
re-generating costs GPU-hours.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from .data.registry import get_dataset, load_dataset_config, subset_classes
from .data.templates import (DEFAULT_QUESTION, PROBE_PROMPT, generative_prompt,
                             zeroshot_prompts)
from .match import Matcher, score
from .models.registry import build_model, load_model_config
from .paths import RESULTS_DIR, ensure_dirs, use_storage_hf_cache


def resolve_prompt(args) -> tuple:
    """Return ``(prompt, question_key, question_text)`` for Experiment 1.

    Only the *question* varies. The chat frame, the decoding settings and the
    matcher are held fixed, so the two rows differ in exactly one thing.
    """
    questions = load_dataset_config(args.dataset).get("generative_questions", {})
    q = questions.get(args.question)
    if q is None:
        if args.question == "default":
            q = DEFAULT_QUESTION
        else:
            raise ValueError(
                f"dataset {args.dataset!r} has no generative question {args.question!r}; "
                f"available: {sorted(questions) or ['default']}"
            )
    prompt = PROBE_PROMPT if args.question == "default" else generative_prompt(q)
    return prompt, args.question, q


def run_generative(model, ds, args) -> dict:
    prompt, qkey, qtext = resolve_prompt(args)
    gens: List[str] = []
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    t0 = time.time()
    for lo in range(0, n, args.batch_size):
        hi = min(lo + args.batch_size, n)
        images = [ds[i][0] for i in range(lo, hi)]
        batch_gens = model.generate(images, prompt, max_new_tokens=args.max_new_tokens)
        # LLaDA-V's generate_with_embeds takes inputs_embeds of shape (1, l, d) and
        # silently returns one row for a whole batch. Without this assert the later
        # zip(gens, labels) truncates to the short list and reports a score over a
        # fraction of the split as if it were the whole thing.
        assert len(batch_gens) == len(images), (
            f"{args.model}: generate() returned {len(batch_gens)} rows for "
            f"{len(images)} images — use --batch-size 1 for this model"
        )
        gens += batch_gens
        if lo % (args.batch_size * 10) == 0:
            rate = hi / max(time.time() - t0, 1e-6)
            print(f"  {hi}/{n}  {rate:.1f} img/s  eta {(n - hi) / max(rate, 1e-6) / 60:.1f} min",
                  flush=True)

    labels = ds.labels[:n].tolist()
    matcher = Matcher(ds.classnames, ds.name, use_embeddings=not args.no_embeddings)
    results = matcher.match(gens)
    summary = score(results, labels)
    summary.update(
        model=args.model, dataset=args.dataset, split=args.split,
        mode="generative", max_new_tokens=args.max_new_tokens,
        decoding=getattr(model, "last_decode_params", None)
        or "greedy (do_sample=False, num_beams=1)",
        question_key=qkey, question=qtext, prompt=prompt,
        secs=round(time.time() - t0, 1),
    )

    suffix = "" if qkey == "default" else f"_{qkey}"
    raw = (RESULTS_DIR / "generations"
           / f"{args.model}_{args.dataset}_{args.split}{suffix}.jsonl")
    raw.parent.mkdir(parents=True, exist_ok=True)
    with open(raw, "w") as fh:
        for g, r, y in zip(gens, results, labels):
            fh.write(json.dumps({
                "generation": g, "pred": r.pred, "rung": r.rung,
                "score": round(r.score, 4), "label": y,
                "label_name": ds.classnames[y],
            }) + "\n")
    print(f"raw generations -> {raw}")
    return summary


def run_zeroshot(model, ds, args) -> dict:
    """CLIP/SigLIP2 zero-shot over T0-c with the dataset's template ensemble."""
    dcfg = load_dataset_config(args.dataset)
    templates = dcfg.get("templates", ["a photo of a {}."])
    prompts = zeroshot_prompts(ds.classnames, templates)

    t0 = time.time()
    # Ensemble = mean of the L2-normed per-template embeddings, renormalised.
    weights = []
    for per_class in prompts:
        emb = model.encode_texts(per_class)
        w = emb.mean(0)
        weights.append(w / w.norm())
    W = torch.stack(weights)  # [C, d]

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    preds = []
    for lo in range(0, n, args.batch_size):
        hi = min(lo + args.batch_size, n)
        images = [ds[i][0] for i in range(lo, hi)]
        f = model.encode_images(images)["t0c"]
        preds.append((f.float() @ W.float().T).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    labels = ds.labels[:n]

    top1 = 100 * float((preds == labels).mean())
    return {
        "model": args.model, "dataset": args.dataset, "split": args.split,
        "mode": "zeroshot", "strict": top1, "lenient": top1, "pct_unmatched": 0.0,
        "n": int(n), "templates": templates, "secs": round(time.time() - t0, 1),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--classes", type=int, default=None,
                   help="keep all images of N randomly chosen classes; must match "
                        "the value used at extraction so the label sets agree")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--question", default="default",
                   help="key in the dataset config's generative_questions (§6.1)")
    p.add_argument("--no-embeddings", action="store_true", help="stop the cascade at rung 4")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    use_storage_hf_cache()
    ensure_dirs()
    mcfg = load_model_config(args.model)
    if args.batch_size is None:
        args.batch_size = int(mcfg.get("batch_size", 32))

    model = build_model(args.model, device=args.device)
    ds = get_dataset(args.dataset, args.split)
    if args.classes is not None:
        ds = subset_classes(ds, args.classes)
        args.dataset = f"{args.dataset}_c{args.classes}"
    print(f"{args.dataset}/{args.split}: {len(ds)} images, {ds.num_classes} classes", flush=True)

    summary = (run_zeroshot if mcfg["kind"] == "encoder" else run_generative)(model, ds, args)

    tag = "" if summary.get("question_key", "default") == "default" else f"_{summary['question_key']}"
    out = RESULTS_DIR / "generative" / f"{args.model}_{args.dataset}_{args.split}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rungs"}, indent=2))
    if "rungs" in summary:
        print("rungs:", summary["rungs"])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
