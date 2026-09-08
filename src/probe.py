"""Linear probes over cached features (§6.2).

    python -m src.probe --model llava15_7b --dataset cub --tap T2 --pool avg
    python -m src.probe --model llava15_7b --dataset cub --all-taps --head logreg
    python -m src.probe --model llava15_7b --dataset cub --tap T2 --pool avg --head knn

Two heads, and only two (notes/decisions.md):

* ``--head linear`` is Zhang et al. B.4 verbatim — Adam, lr 1e-3, batch 512, 500
  epochs, select on best val top-1. Use it for headline cells.
* ``--head logreg`` is the L-BFGS fast path for sweeps. They differ by ~0.5%, so
  never mix them inside one table or figure (§13).

Standardisation is on by default and should stay on: post-LLM states are RMSNorm
outputs with large outlier dimensions, and an unstandardised 500-epoch Adam run
underfits badly and manufactures a drop that is not there (§13).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from .data.registry import stratified_val_split
from .paths import CONFIGS_DIR, RESULTS_DIR, feature_dir

CORE_TAPS = [
    ("T0", "avg"), ("T0", "cls"),
    ("T1", "avg"), ("T1", "final"),
    ("T2", "avg"), ("T2", "final"),
    ("T3", "avg"), ("T3", "final"), ("T3", "pos0"),
    ("T0c", None),
]


def load_probe_config() -> dict:
    return yaml.safe_load((CONFIGS_DIR / "probe.yaml").read_text())


def key_of(tap: str, pool: Optional[str]) -> str:
    return tap if pool is None else f"{tap}.{pool}"


def load_features(model: str, dataset: str, split: str, key: str) -> np.ndarray:
    path = feature_dir(model, dataset, split) / f"{key}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run src.extract for this (model, dataset, split)")
    return np.load(path)


def load_split(model: str, dataset: str, key: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    x = load_features(model, dataset, split, key)
    y = np.load(feature_dir(model, dataset, split) / "labels.npy")
    return x.astype(np.float32), y


def standardize(train: np.ndarray, *others: np.ndarray):
    """Fit on train only (§6.2). eps guards constant dimensions."""
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True) + 1e-6
    return ((train - mu) / sd, *[(o - mu) / sd for o in others])


# ------------------------------------------------------------------ torch head


def train_linear_torch(
    xtr, ytr, xva, yva, xte, yte, num_classes: int, cfg: dict, seed: int, device: str
) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    Xtr = torch.from_numpy(xtr).to(dev)
    Ytr = torch.from_numpy(ytr).long().to(dev)
    Xva = torch.from_numpy(xva).to(dev)
    Yva = torch.from_numpy(yva).long().to(dev)
    Xte = torch.from_numpy(xte).to(dev)
    Yte = torch.from_numpy(yte).long().to(dev)

    head = torch.nn.Linear(Xtr.shape[1], num_classes).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lossf = torch.nn.CrossEntropyLoss()
    bs = int(cfg["batch_size"])
    n = Xtr.shape[0]

    best_val, best_state = -1.0, None
    for epoch in range(int(cfg["epochs"])):
        head.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad(set_to_none=True)
            lossf(head(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        # Model selection every epoch is what `select: best_val_top1` means; it is
        # one matmul on <1k rows, so the cost is negligible.
        head.eval()
        with torch.no_grad():
            va = (head(Xva).argmax(1) == Yva).float().mean().item()
        if va > best_val:
            best_val = va
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        logits = head(Xte)
        top1 = (logits.argmax(1) == Yte).float().mean().item()
        k5 = min(5, num_classes)
        top5 = (logits.topk(k5, dim=1).indices == Yte[:, None]).any(1).float().mean().item()
    return {"top1": 100 * top1, "top5": 100 * top5, "val_top1": 100 * best_val}


# ---------------------------------------------------------------- sklearn head


def train_logreg(xtr, ytr, xva, yva, xte, yte, num_classes: int, cfg: dict, seed: int) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=int(cfg["max_iter"]), C=float(cfg["C"]), n_jobs=-1, random_state=seed
    )
    clf.fit(np.concatenate([xtr, xva]), np.concatenate([ytr, yva]))
    proba = clf.decision_function(xte)
    top1 = (proba.argmax(1) == yte).mean()
    k5 = min(5, num_classes)
    top5 = np.mean([yte[i] in np.argsort(-proba[i])[:k5] for i in range(len(yte))])
    return {"top1": 100 * top1, "top5": 100 * top5, "val_top1": float("nan")}


# ------------------------------------------------------------------------ main


def run_one(
    model: str, dataset: str, tap: str, pool: Optional[str], head: str,
    seeds: Sequence[int], device: str, pcfg: dict, pca_dim: Optional[int] = None,
) -> List[dict]:
    key = key_of(tap, pool)
    xtr_all, ytr_all = load_split(model, dataset, key, "train")
    xte, yte = load_split(model, dataset, key, "test")

    num_classes = int(max(ytr_all.max(), yte.max())) + 1
    tr_idx, va_idx = stratified_val_split(
        ytr_all, pcfg["val_fraction"], pcfg["val_seed"]
    )
    xtr, ytr = xtr_all[tr_idx], ytr_all[tr_idx]
    xva, yva = xtr_all[va_idx], ytr_all[va_idx]

    if pcfg["probe"]["standardize"]:
        xtr, xva, xte = standardize(xtr, xva, xte)

    if pca_dim is not None and xtr.shape[1] > pca_dim:
        # §6.2 dimensionality-fairness control: a wider linear probe is strictly
        # more expressive, so cross-tap comparisons need a common width.
        from sklearn.decomposition import PCA

        pca = PCA(n_components=pca_dim, random_state=0).fit(xtr)
        xtr, xva, xte = pca.transform(xtr), pca.transform(xva), pca.transform(xte)

    rows = []
    for seed in seeds:
        t0 = time.time()
        if head == "linear":
            r = train_linear_torch(xtr, ytr, xva, yva, xte, yte, num_classes,
                                   pcfg["probe"], seed, device)
        elif head == "logreg":
            r = train_logreg(xtr, ytr, xva, yva, xte, yte, num_classes,
                             pcfg["sklearn"], seed)
        else:
            raise ValueError(f"unknown head {head!r}")
        rows.append({
            "model": model, "dataset": dataset, "tap": tap, "pooling": pool or "-",
            "head": head, "seed": seed, "split": "test", "d": int(xtr.shape[1]),
            "n_train": int(xtr.shape[0]), "num_classes": num_classes,
            "pca_dim": pca_dim or "", "secs": round(time.time() - t0, 1), **r,
        })
        print(f"  {model:<14} {key:<12} {head:<7} seed{seed}  "
              f"top1={r['top1']:.2f}  top5={r['top5']:.2f}  ({rows[-1]['secs']}s)", flush=True)
        if head == "logreg":
            break  # deterministic given the data; seeds would just repeat the fit
    return rows


def available_keys(
    model: str, dataset: str, split: str = "train", include_sweep: bool = False
) -> List[str]:
    """Keys on disk. Sweep (``mid.*``) keys are excluded unless asked for.

    The CUB layer sweep is extracted and analysed; it is not being repeated or
    re-probed (notes/decisions.md). ``--all-taps`` therefore means "all core
    taps", so that a model whose ``mid.*`` files happen to be on disk does not
    silently drag ten extra fits back into every run.
    """
    d = feature_dir(model, dataset, split)
    keys = sorted(p.stem for p in d.glob("*.npy") if p.stem != "labels")
    if not include_sweep:
        keys = [k for k in keys if not k.startswith("mid.")]
    return keys


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--tap")
    p.add_argument("--pool")
    p.add_argument("--all-taps", action="store_true", help="every core key present on disk")
    p.add_argument("--include-sweep", action="store_true",
                   help="also probe the mid.* layer-sweep keys (off by default)")
    p.add_argument("--head", default="linear", choices=["linear", "logreg"])
    p.add_argument("--seeds", default=None, help="comma list; default from probe.yaml")
    p.add_argument("--pca-dim", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="csv to append to (default results/table2_probes.csv)")
    args = p.parse_args(argv)

    pcfg = load_probe_config()
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else pcfg["probe"]["seeds"])

    if args.all_taps:
        targets = []
        for key in available_keys(args.model, args.dataset,
                                  include_sweep=args.include_sweep):
            tap, _, pool = key.partition(".")
            targets.append((key if pool == "" else tap, pool or None))
    else:
        assert args.tap, "pass --tap or --all-taps"
        targets = [(args.tap, args.pool)]

    rows: List[dict] = []
    for tap, pool in targets:
        rows += run_one(args.model, args.dataset, tap, pool, args.head, seeds,
                        args.device, pcfg, args.pca_dim)

    out = Path(args.out) if args.out else RESULTS_DIR / "table2_probes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
