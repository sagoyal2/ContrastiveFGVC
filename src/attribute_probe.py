"""Experiment 3 — the attribute probe (§6.3).

    python -m src.attribute_probe --model llava15_7b --dataset cub --all-taps

Class probes measure *class* separability. The hypothesis is about
**concept-attribute knowledge retrieval**, so this trains a multi-label probe for
CUB's 312 binary attributes at each tap and reports mean average precision. It is
the instrument that distinguishes this study from a re-run of FINER Fig. 5.

Three decisions worth stating in the writeup:

* **Per-image, not per-class attributes.** CUB ships both. Per-class labels would
  make the attribute probe a relabelled class probe; per-image labels are noisier
  but actually ask whether the *visible* attribute survived the stack.
* **Attributes with almost no positives are dropped.** AP on a 30-positive column
  out of 5,994 is dominated by variance, and averaging those in mostly measures
  the label distribution.
* **One batched multi-label head, not 312 sequential binary fits.** The obvious
  implementation — a ``sklearn.LogisticRegression`` per attribute — measured at
  5.4 s per fit on 5,994x1024 features, i.e. ~25 min for one 1024-d tap and
  ~98 min for a 4096-d one: about 12 hours for a single model's nine taps. The
  attributes share the same design matrix, so they are fitted simultaneously as
  one ``nn.Linear(d, n_attr)`` with ``BCEWithLogitsLoss`` on the GPU, which is
  the same model class (independent per-attribute logistic regressions) at a
  fraction of the cost. Note this changes the optimiser from L-BFGS to Adam;
  like the class probe's two heads, do not mix the two within one table.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .data.registry import get_dataset, load_cub_attributes, stratified_val_split
from .probe import available_keys, load_features, load_probe_config, standardize
from .paths import RESULTS_DIR, feature_dir

MIN_POSITIVES = 50


def attribute_targets(dataset: str, split: str) -> tuple:
    if dataset != "cub":
        raise NotImplementedError(f"no per-image attributes wired up for {dataset!r}")
    ds = get_dataset(dataset, split)
    return load_cub_attributes(ds)


def run_one(model: str, dataset: str, key: str, pcfg: dict, seed: int = 0,
            device: str = "cuda", epochs: int = 200) -> dict:
    """Multi-label attribute probe at one tap. Returns mean AP over kept columns."""
    import torch
    from sklearn.metrics import average_precision_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    xtr_all = load_features(model, dataset, "train", key).astype(np.float32)
    xte = load_features(model, dataset, "test", key).astype(np.float32)
    ytr_all, names = attribute_targets(dataset, "train")
    yte, _ = attribute_targets(dataset, "test")
    assert xtr_all.shape[0] == ytr_all.shape[0] and xte.shape[0] == yte.shape[0]

    # A column needs positives on both sides to have a defined AP at all.
    keep = np.flatnonzero((ytr_all.sum(0) >= MIN_POSITIVES) & (yte.sum(0) >= MIN_POSITIVES))
    ytr_all, yte = ytr_all[:, keep], yte[:, keep]

    # Same stratified carve as the class probe, stratified on class rather than on
    # attribute: the split has to be one split, not 312 conflicting ones.
    labels = np.load(feature_dir(model, dataset, "train") / "labels.npy")
    tr_idx, va_idx = stratified_val_split(labels, pcfg["val_fraction"], pcfg["val_seed"])
    xtr, ytr = xtr_all[tr_idx], ytr_all[tr_idx]
    xva, yva = xtr_all[va_idx], ytr_all[va_idx]

    if pcfg["probe"]["standardize"]:
        xtr, xva, xte = standardize(xtr, xva, xte)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    Xtr = torch.from_numpy(xtr).to(dev)
    Ytr = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    Xva = torch.from_numpy(xva).to(dev)
    Yva = torch.from_numpy(yva.astype(np.float32)).to(dev)
    Xte = torch.from_numpy(xte).to(dev)

    head = torch.nn.Linear(Xtr.shape[1], len(keep)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=pcfg["probe"]["lr"],
                           weight_decay=pcfg["probe"]["weight_decay"])
    lossf = torch.nn.BCEWithLogitsLoss()
    bs = int(pcfg["probe"]["batch_size"])
    n = Xtr.shape[0]

    t0 = time.time()
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad(set_to_none=True)
            lossf(head(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            # Selecting on val BCE rather than val mAP: mAP over ~270 columns is
            # noisy epoch to epoch and the argmax lands on a lucky epoch.
            v = lossf(head(Xva), Yva).item()
        if v < best_val:
            best_val, best_state = v, {k: t.detach().clone() for k, t in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        scores = head(Xte).float().cpu().numpy()

    aps = [average_precision_score(yte[:, j], scores[:, j]) for j in range(len(keep))]
    # Baseline AP for a random ranker is the positive rate; report the lift too,
    # otherwise a tap that merely predicts the marginal looks respectable.
    prior = float(yte.mean())
    return {
        "model": model, "dataset": dataset, "key": key, "seed": seed,
        "n_attributes": len(keep), "d": int(xtr.shape[1]),
        "mAP": 100 * float(np.mean(aps)),
        "mAP_std": 100 * float(np.std(aps)),
        "prior_AP": 100 * prior,
        "lift": 100 * (float(np.mean(aps)) - prior),
        "head": "torch_multilabel", "epochs": epochs,
        "secs": round(time.time() - t0, 1),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="cub")
    p.add_argument("--key", help="e.g. T2.avg")
    p.add_argument("--all-taps", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    pcfg = load_probe_config()
    if args.all_taps:
        keys = available_keys(args.model, args.dataset)  # core taps only
    else:
        assert args.key, "pass --key or --all-taps"
        keys = [args.key]

    rows = []
    for key in keys:
        r = run_one(args.model, args.dataset, key, pcfg,
                    device=args.device, epochs=args.epochs)
        rows.append(r)
        print(f"  {args.model:<14} {key:<12} mAP={r['mAP']:.2f} "
              f"(prior {r['prior_AP']:.2f}, lift {r['lift']:+.2f}) "
              f"over {r['n_attributes']} attrs  ({r['secs']}s)", flush=True)

    out = Path(args.out) if args.out else RESULTS_DIR / "table3_attributes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
