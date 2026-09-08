"""Render the deliverable tables (§12) from whatever results exist on disk.

    python -m src.report --table1
    python -m src.report --table2 --dataset cub
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .paths import RESULTS_DIR


def table1() -> str:
    rows = []
    for p in sorted((RESULTS_DIR / "generative").glob("*.json")):
        d = json.loads(p.read_text())
        rows.append({
            "model": d["model"], "dataset": d["dataset"], "mode": d["mode"],
            # Without this the two generative rows per model are indistinguishable,
            # and the whole point of the pair is which question was asked.
            "question": d.get("question_key", "-" if d["mode"] == "zeroshot" else "default"),
            "strict": round(d["strict"], 2), "lenient": round(d["lenient"], 2),
            "%unmatched": round(d["pct_unmatched"], 2), "n": d["n"],
        })
    if not rows:
        return "_no generative results yet_\n"
    df = pd.DataFrame(rows).sort_values(["dataset", "mode", "model", "question"])
    md = ["# Table 1 — generative / zero-shot top-1 (§6.1)", "",
          "Three numbers per cell, never one: `strict` is matcher rungs 1-3 "
          "(normalise / exact / alias), `lenient` adds substring and embedding "
          "matching, and `%unmatched` is what the cascade could not resolve at all. "
          "Encoder rows are contrastive zero-shot, which has no matcher, so their "
          "strict and lenient columns are equal by construction.", ""]
    md.append(df.to_markdown(index=False))
    return "\n".join(md) + "\n"


def table2(dataset: Optional[str] = None) -> str:
    path = RESULTS_DIR / "table2_probes.csv"
    if not path.exists():
        return "_no probe results yet_\n"
    df = pd.read_csv(path)
    if dataset:
        df = df[df["dataset"] == dataset]
    df["key"] = df["tap"] + np.where(df["pooling"] == "-", "", "." + df["pooling"].astype(str))
    g = (df.groupby(["dataset", "model", "head", "key"])["top1"]
           .agg(["mean", "std", "count"]).reset_index())
    g["top1"] = g.apply(
        lambda r: f"{r['mean']:.2f}" + (f" ± {r['std']:.2f}" if r["count"] > 1 else ""), axis=1
    )
    wide = g.pivot_table(index=["dataset", "model", "head"], columns="key",
                         values="top1", aggfunc="first")
    order = [c for c in ["T0c", "T0.avg", "T0.cls", "T1.avg", "T1.final",
                         "T2.avg", "T2.final", "T3.avg", "T3.final",
                         "T3.pos0"] if c in wide.columns]
    rest = sorted(c for c in wide.columns if c not in order)
    wide = wide[order + rest]
    md = ["# Table 2 — linear-probe top-1 by tap (§6.2)", "",
          "Mean ± std over seeds where more than one seed was run. Widths differ "
          "between taps (1024 / 1152 / 4096 / 5120) and a wider linear probe is "
          "strictly more expressive, so a *higher* number at a wider tap is not by "
          "itself evidence of more information — see the PCA-matched control.", ""]
    md.append(wide.reset_index().to_markdown(index=False))
    return "\n".join(md) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table1", action="store_true")
    p.add_argument("--table2", action="store_true")
    p.add_argument("--dataset", default=None)
    args = p.parse_args(argv)

    if args.table1 or not (args.table1 or args.table2):
        out = RESULTS_DIR / "table1_generative.md"
        out.write_text(table1())
        print(table1())
        print(f"-> {out}")
    if args.table2:
        out = RESULTS_DIR / "table2_probes.md"
        out.write_text(table2(args.dataset))
        print(table2(args.dataset))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
