"""Phase 1 gates and the §11 sanity checks, evaluated against what is on disk.

    python -m src.checks --dataset cub --model llava15_7b

Prints a PASS/FAIL line per gate. §0.5 is explicit that Phase 2 must not start
until these hold: the remaining phases cost 17.7x the extraction of Phase 1, and
every one of these failures would silently corrupt all of it.

The mechanical checks that need a live model (span width, T1 identity, blanking)
live in ``tests/test_span.py`` — run pytest for those. This module covers the
checks that are statements about *numbers*, which only exist after probing.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .paths import RESULTS_DIR


def _load(csv_path=None) -> pd.DataFrame:
    path = csv_path or RESULTS_DIR / "table2_probes.csv"
    df = pd.read_csv(path)
    df["key"] = df["tap"] + np.where(df["pooling"] == "-", "", "." + df["pooling"].astype(str))
    return df


def _acc(df, model, dataset, key, head) -> Optional[float]:
    sub = df[(df["model"] == model) & (df["dataset"] == dataset)
             & (df["key"] == key) & (df["head"] == head)]
    return None if sub.empty else float(sub["top1"].mean())


def _line(ok: Optional[bool], label: str, detail: str) -> bool:
    tag = "PASS" if ok else ("FAIL" if ok is False else "SKIP")
    print(f"[{tag}] {label}: {detail}")
    return bool(ok)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cub")
    p.add_argument("--model", default="llava15_7b")
    p.add_argument("--head", default="logreg")
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    df = _load(args.csv)
    m, d, h = args.model, args.dataset, args.head
    results = []

    # --- §11 check 2 / Phase-1 structural: position 0 must collapse to near chance.
    pos0 = _acc(df, m, d, "T3.pos0", h)
    t3f = _acc(df, m, d, "T3.final", h)
    # The control is **causal-specific**. Position 0 probes at chance only because
    # causal attention prevents it from seeing the image at all. LLaDA-V's
    # attention is bidirectional, so position 0 attends over all 729 image tokens
    # and legitimately carries image information (measured: 22.71%, versus 0.33%
    # for both LLaVA sizes). Applying the check there would report a bug that is
    # not there — so it is skipped, and the bidirectional case is instead asserted
    # the other way round: pos0 should look like any other non-image position.
    is_causal_model = "llada" not in m.lower()
    if pos0 is None:
        results.append(_line(None, "position-0 collapse", "no T3.pos0 probe on disk"))
    elif is_causal_model:
        # "Near chance" is generous here: Zhang et al. report 0.7% on Flowers102,
        # but the bar that matters is that it is nowhere near the real tap.
        ok = pos0 < 5.0 and (t3f is None or pos0 < 0.25 * t3f)
        results.append(_line(ok, "position-0 collapse",
                             f"T3.pos0 = {pos0:.2f}%  vs T3.final = {t3f:.2f}%"))
    else:
        # Bidirectional: every non-image position sees the same thing, so pos0 and
        # final should agree. A *collapsed* pos0 here would mean the attention mask
        # is accidentally causal.
        ok = t3f is not None and abs(pos0 - t3f) < 5.0
        results.append(_line(ok, "position-0 parity (bidirectional model)",
                             f"T3.pos0 = {pos0:.2f}% vs T3.final = {t3f:.2f}% — expect ~equal; "
                             f"a collapsed pos0 would mean the mask is causal"))

    # --- Gate 3 / §11 check 3: FINER's direction, T0.avg > T1.avg.
    t0 = _acc(df, m, d, "T0.avg", h)
    t1 = _acc(df, m, d, "T1.avg", h)
    if t0 is None or t1 is None:
        results.append(_line(None, "gate 3 — FINER direction", "T0.avg / T1.avg missing"))
    else:
        results.append(_line(t0 > t1, "gate 3 — FINER direction",
                             f"T0.avg = {t0:.2f}% {'>' if t0 > t1 else '<='} T1.avg = {t1:.2f}% "
                             f"(drop {t0 - t1:+.2f})"))

    # --- Gate 4: the probe-vs-generation gap must exist.
    gen_path = RESULTS_DIR / "generative" / f"{m}_{d}_test.json"
    if not gen_path.exists() or t3f is None:
        results.append(_line(None, "gate 4 — probe >> generative",
                             "need both the generative run and the T3.final probe"))
    else:
        gen = json.loads(gen_path.read_text())
        g = gen["lenient"]
        results.append(_line(t3f > g + 10, "gate 4 — probe >> generative",
                             f"T3.final probe = {t3f:.2f}% vs generative lenient = {g:.2f}% "
                             f"(gap {t3f - g:+.2f})"))

    # --- Gate 2: the encoder reference. CLIP-L on ImageNet val.
    clip_in = _acc(df, "clip_l14_336", "imagenet_val", "T0c", h)
    if clip_in is None:
        results.append(_line(None, "gate 2 — CLIP-L ImageNet reference", "not run"))
    else:
        # See configs/datasets/imagenet_val.yaml: probing 40/class of val is a
        # lower-data proxy for Zhang et al.'s 85.2%, so the band is deliberately
        # wide and its floor is what matters.
        ok = 70.0 <= clip_in <= 90.0
        results.append(_line(ok, "gate 2 — CLIP-L ImageNet reference",
                             f"T0c probe = {clip_in:.2f}% (expect 70-90 on the val-only carve; "
                             f"published full-train reference is 85.2)"))

    # --- §9 / §11 check 6: LLaDA bidirectionality, if that model has been run.
    #
    # §9 predicts "T2.avg ~= T2.final for LLaDA-V, but not for LLaVA", because
    # LLaDA's attention is not causal so the last image token is not privileged.
    # Testing that as an absolute epsilon was a mistake: two effects push the
    # diffusion model's `final` down without any causal attention being involved.
    # LLaDA-V decodes from appended mask positions, not from the last image token,
    # so nothing trains that position to summarise the image; and under
    # `aspect_ratio: pad` the bottom-right patch is letterbox padding
    # (notes/decisions.md). What the prediction actually implies, and what is
    # diagnostic of a span bug, is *comparative*: the bidirectional model's
    # avg-vs-final gap must be clearly narrower than the causal models'.
    causal_gaps = []
    for m_causal in [x for x in df["model"].unique() if "llava" in str(x)]:
        a, f = _acc(df, m_causal, d, "T2.avg", h), _acc(df, m_causal, d, "T2.final", h)
        if a is not None and f is not None:
            causal_gaps.append(a - f)

    for llada in [x for x in df["model"].unique() if "llada" in str(x)]:
        a = _acc(df, llada, d, "T2.avg", h)
        f = _acc(df, llada, d, "T2.final", h)
        if a is None or f is None:
            continue
        gap = a - f
        if not causal_gaps:
            results.append(_line(None, "§11 check 6 — LLaDA bidirectionality",
                                 f"T2.avg-T2.final = {gap:.2f} but no causal model to compare against"))
            continue
        ref = sum(causal_gaps) / len(causal_gaps)
        results.append(_line(gap < 0.8 * ref, "§11 check 6 — LLaDA bidirectionality",
                             f"T2.avg-T2.final = {gap:.2f} (avg {a:.2f}, final {f:.2f}) vs "
                             f"causal reference {ref:.2f} — expect clearly narrower"))

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} evaluated checks passed")


if __name__ == "__main__":
    main()
