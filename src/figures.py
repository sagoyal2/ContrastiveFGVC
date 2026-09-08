"""Figures (§12).

    python -m src.figures --figure fig5 --datasets cub
    python -m src.figures --figure sweep --datasets cub
    python -m src.figures --figure masked --datasets cub

Reads whatever is in ``results/table2_probes.csv`` and plots that — never assumes
all five datasets exist, so the same command works at every phase (§0.5).

``fig5`` is the recreation and extension of FINER Fig. 5: the original compares
only T0 vs T1 (before/after projection); the extra series carry it through the
LLM stack, which is the contribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FIGURES_DIR, RESULTS_DIR

FIG5_SERIES = ["T0.avg", "T1.avg", "T2.avg", "T2.final", "T3.final"]
# Encoders first (they are the ceilings), then the VLMs in size order, then the
# diffusion model. Alphabetical order puts LLaDA-V second and reads as noise.
MODEL_ORDER = ["clip_l14_336", "siglip2_so400m", "llava15_7b", "llava15_13b", "lladav_8b"]
SERIES_LABEL = {
    "T0.avg": "T0 pre-projection (ViT −2)",
    "T1.avg": "T1 post-projection",
    "T2.avg": "T2 LLM, image span (avg)",
    "T2.final": "T2 LLM, image span (final)",
    "T3.final": "T3 LLM, ':' token",
}


def load_probes(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or RESULTS_DIR / "table2_probes.csv"
    df = pd.read_csv(path)
    df["key"] = df["tap"] + np.where(df["pooling"] == "-", "", "." + df["pooling"].astype(str))
    return df


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds. The spread matters on small FGVC sets (§6.2)."""
    return (df.groupby(["model", "dataset", "key", "head"])["top1"]
              .agg(["mean", "std", "count"]).reset_index())


def fig5(df: pd.DataFrame, datasets: Sequence[str], head: str, out: Path) -> None:
    g = _agg(df[df["head"] == head])
    # A masked-mode run is the same model under a different tap protocol, not a
    # different model: its T0/T1/T2 bars are identical to clean mode by
    # construction, so a sixth panel would duplicate three of five bars and read
    # as a separate system. Clean-vs-masked belongs in its own table (WRITEUP §5).
    present = {m for m in g["model"].unique() if "_masked" not in str(m)}
    models = ([m for m in MODEL_ORDER if m in present]
              + sorted(present - set(MODEL_ORDER)))
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 4.8), squeeze=False)

    gen = _generative_reference()
    for ax, model in zip(axes[0], models):
        sub = g[g["model"] == model]
        ds_present = [d for d in datasets if d in set(sub["dataset"])]
        x = np.arange(len(ds_present))
        series = [s for s in FIG5_SERIES if s in set(sub["key"])]
        w = 0.8 / max(len(series), 1)

        for i, key in enumerate(series):
            vals = [sub[(sub["dataset"] == d) & (sub["key"] == key)]["mean"].mean()
                    for d in ds_present]
            errs = [sub[(sub["dataset"] == d) & (sub["key"] == key)]["std"].mean()
                    for d in ds_present]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs, capsize=2,
                   label=SERIES_LABEL.get(key, key))

        # Reference marks (§12): the probe-vs-generation gap is the point of putting
        # them on the same axes. Labelled by mode, because a zero-shot number and a
        # generative-plus-matcher number are different measurements and a shared
        # line style would imply otherwise.
        for j, d in enumerate(ds_present):
            ref = gen.get((model, d))
            if ref is None:
                continue
            lab = ("zero-shot top-1" if ref["mode"] == "zeroshot"
                   else "generative top-1 (strict)")
            ax.hlines(ref["value"], j - 0.45, j + 0.45, color="k", ls="--", lw=1.4,
                      label=lab if j == 0 else None)
            # Left-aligned at the bar-group edge: a strict generative score of 0.0
            # sits on the axis, where a centred label collides with the tick text.
            ax.annotate(f"{ref['value']:.1f}", (j - 0.44, ref["value"]), xytext=(2, 5),
                        textcoords="offset points", ha="left", fontsize=8,
                        color="k", fontweight="bold")

        ax.set_xticks(x, ds_present)
        ax.set_title(model)
        ax.set_ylabel("linear-probe top-1 (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
    # One figure-level legend, built from every panel's handles: the per-axes
    # version landed on whichever model was drawn last, and the encoders carry
    # only the T0 series, so most of the key was invisible.
    handles, labels = {}, []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
                labels.append(l)
    fig.legend([handles[l] for l in labels], labels, fontsize=9,
               loc="lower center", ncol=min(len(labels), 4), frameon=False,
               bbox_to_anchor=(0.5, -0.20))
    fig.suptitle(f"Probe accuracy at each tap ({head} head) — FINER Fig. 5 recreation + extension")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"-> {out}")


def sweep(df: pd.DataFrame, datasets: Sequence[str], head: str, out: Path) -> None:
    """Layer sweep (§8.3B): flat-and-low from layer 0 means the projector lost it;
    high-early-decaying-late means the stack discards it progressively."""
    d = df[df["key"].str.startswith("mid.")].copy()
    if d.empty:
        print("no mid.* rows — extract with --sweep first")
        return
    d["layer"] = d["key"].str.extract(r"mid\.L(\d+)\.").astype(int)
    d["pool"] = d["key"].str.rsplit(".", n=1).str[-1]
    g = d.groupby(["model", "dataset", "layer", "pool"])["top1"].agg(["mean", "std"]).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, pool in zip(axes, ["avg", "final"]):
        for (model, ds), sub in g[g["pool"] == pool].groupby(["model", "dataset"]):
            sub = sub.sort_values("layer")
            ax.errorbar(sub["layer"], sub["mean"], yerr=sub["std"], marker="o",
                        capsize=2, label=f"{model} / {ds}")
        ax.set_xlabel("LLM layer (0 = projected image embedding)")
        ax.set_title(f"pooling = {pool}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("linear-probe top-1 (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Where in the LLM stack the class signal goes")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"-> {out}")


MASKED_TAPS = ["T0.avg", "T1.avg", "T2.avg", "T2.final", "T3.avg", "T3.final",
               "T3.pos0", "Tmask.avg", "Tmask.first", "Tmask.last"]


def masked(df: pd.DataFrame, datasets: Sequence[str], head: str, out: Path) -> None:
    """LLaDA-V clean vs masked (§9), with the answer-span taps.

    Kept out of fig5 deliberately: this is one model under two tap protocols, not
    two models. The controlled part is that T0/T1 are bit-identical between them —
    the image path is untouched — so every post-LLM difference is caused by the
    presence of the [MASK] span alone.
    """
    g = _agg(df[df["head"] == head])
    g = g[g["dataset"].isin(datasets)]
    clean = g[g["model"] == "lladav_8b"].set_index("key")["mean"]
    msk = g[g["model"].str.contains("_masked", na=False)].set_index("key")["mean"]
    if clean.empty or msk.empty:
        print("need both lladav_8b and a masked variant probed")
        return

    taps = [t for t in MASKED_TAPS if t in msk.index or t in clean.index]
    x = np.arange(len(taps))
    fig, ax = plt.subplots(figsize=(11, 5.0))
    w = 0.38
    cv = [clean.get(t, np.nan) for t in taps]
    mv = [msk.get(t, np.nan) for t in taps]
    ax.bar(x - w / 2, cv, w, label="clean — no [MASK] span", color="#4C72B0")
    ax.bar(x + w / 2, mv, w, label="masked, k=32 — the decoding configuration",
           color="#DD8452")

    for i, (a, b) in enumerate(zip(cv, mv)):
        if np.isfinite(a) and np.isfinite(b) and abs(b - a) >= 0.05:
            ax.annotate(f"{b - a:+.1f}", (i, max(a, b) + 1.5), ha="center", fontsize=8,
                        fontweight="bold", color="#C44E52" if b > a else "#555555")
        elif np.isfinite(b) and not np.isfinite(a):
            ax.annotate("masked\nonly", (i + w / 2, b + 1.5), ha="center", fontsize=7.5,
                        color="#555555")

    # The pre-projection ceiling: how much of the encoder's signal any tap retains.
    if np.isfinite(clean.get("T0.avg", np.nan)):
        ax.axhline(clean["T0.avg"], color="k", ls=":", lw=1.2)
        # Left-anchored: the legend occupies the top right.
        ax.annotate(f"T0 pre-projection ceiling  {clean['T0.avg']:.2f}",
                    (-0.45, clean["T0.avg"]), xytext=(0, 5),
                    textcoords="offset points", ha="left", fontsize=8)

    ax.set_xticks(x, taps, rotation=20, ha="right")
    ax.set_ylabel("linear-probe top-1 (%)")
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("LLaDA-V: what the [MASK] answer span changes (CUB, logreg head)\n"
                 "T0/T1 are identical by construction — the image path is untouched",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"-> {out}")


def _generative_reference(question: str = "fine") -> Dict[tuple, dict]:
    """Reference marks for fig5, one per (model, dataset).

    Two things this must not do silently:

    * **Pick a question by filename order.** A model can have several generative
      results — CUB has both Zhang et al.'s "What type of object…" and the
      species-level question — and keying on (model, dataset) alone lets whichever
      file globs last win. The question is chosen explicitly here, falling back to
      whatever exists if the preferred one was not run.
    * **Plot `lenient`.** On CUB the lenient number is dominated by the
      sentence-embedding rung resolving "a sparrow" to an arbitrary one of ~20
      sparrow species, so it measures the matcher's tie-breaking. `strict` is the
      defensible figure and is what the mark shows. Encoders have no matcher, so
      their strict and lenient are equal by construction.
    """
    by_key: Dict[tuple, list] = {}
    for p in sorted((RESULTS_DIR / "generative").glob("*.json")):
        d = json.loads(p.read_text())
        by_key.setdefault((d["model"], d["dataset"]), []).append(d)

    out = {}
    for key, cands in by_key.items():
        chosen = next((c for c in cands if c.get("question_key") == question), None)
        if chosen is None:
            chosen = next((c for c in cands if c["mode"] == "zeroshot"), cands[0])
        out[key] = {
            "value": chosen["strict"],
            "mode": chosen["mode"],
            "question": chosen.get("question_key", "-"),
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure", required=True, choices=["fig5", "sweep", "masked"])
    p.add_argument("--datasets", default="cub")
    p.add_argument("--head", default="logreg")
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_probes(Path(args.csv) if args.csv else None)
    datasets = args.datasets.split(",")
    name = {"fig5": "fig5_recreation", "sweep": "layer_sweep",
            "masked": "llada_masked"}[args.figure]
    out = FIGURES_DIR / f"{name}.pdf"
    {"fig5": fig5, "sweep": sweep, "masked": masked}[args.figure](df, datasets, args.head, out)
    plt.close("all")
    # A PNG alongside the PDF: the PDF is the deliverable, the PNG is what gets
    # pasted into a message.
    df2 = df
    fig_png = out.with_suffix(".png")
    {"fig5": fig5, "sweep": sweep, "masked": masked}[args.figure](df2, datasets, args.head, fig_png)


if __name__ == "__main__":
    main()
