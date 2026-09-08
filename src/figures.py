"""Figures (§12).

    python -m src.figures --figure fig5 --datasets cub
    python -m src.figures --figure sweep --datasets cub

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
    present = set(g["model"].unique())
    models = ([m for m in MODEL_ORDER if m in present]
              + sorted(present - set(MODEL_ORDER)))
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.6), squeeze=False)

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

        # Reference marks: generative top-1 (§12) — the probe-vs-generation gap is
        # the point of drawing them on the same axes.
        for j, d in enumerate(ds_present):
            v = gen.get((model, d))
            if v is not None:
                ax.hlines(v, j - 0.45, j + 0.45, color="k", ls="--", lw=1.4,
                          label="generative top-1" if j == 0 else None)

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
               loc="lower center", ncol=min(len(labels), 6), frameon=False,
               bbox_to_anchor=(0.5, -0.06))
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


def _generative_reference() -> Dict[tuple, float]:
    out = {}
    for p in (RESULTS_DIR / "generative").glob("*.json"):
        d = json.loads(p.read_text())
        out[(d["model"], d["dataset"])] = d["lenient"]
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure", required=True, choices=["fig5", "sweep"])
    p.add_argument("--datasets", default="cub")
    p.add_argument("--head", default="logreg")
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_probes(Path(args.csv) if args.csv else None)
    datasets = args.datasets.split(",")
    name = {"fig5": "fig5_recreation", "sweep": "layer_sweep"}[args.figure]
    out = FIGURES_DIR / f"{name}.pdf"
    {"fig5": fig5, "sweep": sweep}[args.figure](df, datasets, args.head, out)
    plt.close("all")
    # A PNG alongside the PDF: the PDF is the deliverable, the PNG is what gets
    # pasted into a message.
    df2 = df
    fig_png = out.with_suffix(".png")
    {"fig5": fig5, "sweep": sweep}[args.figure](df2, datasets, args.head, fig_png)


if __name__ == "__main__":
    main()
