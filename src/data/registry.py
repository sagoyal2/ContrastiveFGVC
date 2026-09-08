"""Dataset access (§7). ``get_dataset(name, split)`` -> indexable (PIL, label).

Design notes:

* **Splits are the dataset's own.** CUB ships ``train_test_split.txt``; we use it
  rather than re-carving, so the 5,994 / 5,794 counts in §7 hold exactly and the
  numbers stay comparable to everyone else's.
* **No preprocessing here.** Each model's own image processor is applied later,
  unmodified (§7). This layer hands back PIL RGB and nothing else.
* **Class names are the human-readable ones**, needed both for the zero-shot
  templates and for the generative matcher (§6.1).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image

from ..paths import DATASETS_DIR, DATASET_CONFIGS_DIR


def load_dataset_config(name: str) -> dict:
    path = DATASET_CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in DATASET_CONFIGS_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no dataset config {name!r}; available: {available}")
    return yaml.safe_load(path.read_text())


@dataclass
class ImageDataset:
    """A split. ``__getitem__`` returns ``(PIL.Image RGB, label_idx)``."""

    name: str
    split: str
    paths: List[Path]
    labels: np.ndarray            # [N] int64
    classnames: List[str]
    image_ids: List[str]          # stable per-image key, for joining attributes

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[Image.Image, int]:
        return Image.open(self.paths[i]).convert("RGB"), int(self.labels[i])

    @property
    def num_classes(self) -> int:
        return len(self.classnames)


# --------------------------------------------------------------------------- CUB


def _cub_classname(raw: str) -> str:
    """``001.Black_footed_Albatross`` -> ``black footed albatross``.

    Lowercased and underscore-free because both the zero-shot template and the
    matcher's rung 1 normalisation (§6.1) work in that space.
    """
    return raw.split(".", 1)[1].replace("_", " ").lower()


def _load_cub(cfg: dict, split: str) -> ImageDataset:
    root = DATASETS_DIR / cfg["root"]
    if not (root / "images.txt").exists():
        raise FileNotFoundError(
            f"CUB not found at {root}. Run scripts/download_cub.sh first."
        )

    def read_pairs(fname: str) -> Dict[str, str]:
        out = {}
        for line in (root / fname).read_text().splitlines():
            if line.strip():
                k, v = line.split(" ", 1)
                out[k] = v
        return out

    images = read_pairs("images.txt")             # id -> relative path
    labels = read_pairs("image_class_labels.txt")  # id -> 1-based class id
    is_train = read_pairs("train_test_split.txt")  # id -> "1" train / "0" test

    classnames = [
        _cub_classname(line.split(" ", 1)[1])
        for line in (root / "classes.txt").read_text().splitlines()
        if line.strip()
    ]

    want = "1" if split == "train" else "0"
    ids = sorted((i for i in images if is_train[i] == want), key=int)
    return ImageDataset(
        name="cub",
        split=split,
        paths=[root / "images" / images[i] for i in ids],
        labels=np.array([int(labels[i]) - 1 for i in ids], dtype=np.int64),
        classnames=classnames,
        image_ids=ids,
    )


def load_cub_attributes(split_ds: ImageDataset) -> Tuple[np.ndarray, List[str]]:
    """Per-image binary attribute matrix ``[N, 312]`` for Experiment 3 (§6.3).

    ``image_attribute_labels.txt`` carries a 1-4 certainty column. Certainty 1 is
    "not visible", which is *absence of evidence*, not evidence of absence — those
    entries are left at 0 but flagged in the returned names only implicitly. We
    binarise on ``is_present`` alone, which is what the CUB paper's own baselines do.
    """
    cfg = load_dataset_config("cub")
    root = DATASETS_DIR / cfg["root"]
    n_attr = int(cfg["attributes"]["count"])

    # The tarball drops attributes.txt *beside* CUB_200_2011/, not inside it,
    # while image_attribute_labels.txt lives in CUB_200_2011/attributes/. Accept
    # either location rather than depending on how it was unpacked.
    rel = cfg["attributes"]["names"]
    names_path = next(
        (p for p in (root / rel, root.parent / rel, root / "attributes" / rel) if p.exists()),
        None,
    )
    if names_path is None:
        raise FileNotFoundError(f"{rel} not found under {root} or {root.parent}")
    names = [
        line.split(" ", 1)[1]
        for line in names_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(names) == n_attr, f"expected {n_attr} attribute names, got {len(names)}"

    pos = {img_id: k for k, img_id in enumerate(split_ds.image_ids)}
    y = np.zeros((len(split_ds), n_attr), dtype=np.uint8)
    with open(root / cfg["attributes"]["file"]) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            img_id, attr_id, present = parts[0], parts[1], parts[2]
            k = pos.get(img_id)
            if k is not None and present == "1":
                y[k, int(attr_id) - 1] = 1
    return y, names



# ------------------------------------------------------------------- ImageNet val


def _load_imagenet_val(cfg: dict, split: str) -> ImageDataset:
    """Val-only, carved per class into probe-train / probe-test (see the config).

    The carve is deterministic (sorted filenames, first k to train) rather than
    randomised, so the split is reproducible without carrying a seed file around.
    """
    root = DATASETS_DIR / cfg["root"]
    index = root / "index.txt"
    if not index.exists():
        raise FileNotFoundError(
            f"{index} missing. Run scripts/prepare_imagenet_val.py first."
        )
    classnames = [
        line.strip() for line in (root / "classnames.txt").read_text().splitlines()
        if line.strip()
    ]

    by_class: Dict[int, List[str]] = {}
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        rel, cls = line.rsplit(" ", 1)
        by_class.setdefault(int(cls), []).append(rel)

    k = int(cfg["probe_split_per_class"])
    paths, labels, ids = [], [], []
    for cls in sorted(by_class):
        rels = sorted(by_class[cls])
        chosen = rels[:k] if split == "train" else rels[k:]
        for rel in chosen:
            paths.append(root / "images" / rel)
            labels.append(cls)
            ids.append(rel)
    return ImageDataset(
        name="imagenet_val",
        split=split,
        paths=paths,
        labels=np.array(labels, dtype=np.int64),
        classnames=classnames,
        image_ids=ids,
    )


_LOADERS = {"cub": _load_cub, "imagenet_val": _load_imagenet_val}


@functools.lru_cache(maxsize=None)
def get_dataset(name: str, split: str) -> ImageDataset:
    assert split in ("train", "test"), f"split must be train/test, got {split!r}"
    cfg = load_dataset_config(name)
    if name not in _LOADERS:
        raise NotImplementedError(f"dataset {name!r} has a config but no loader yet")
    ds = _LOADERS[name](cfg, split)

    expected = cfg.get(f"{split}_size")
    if expected is not None:
        assert len(ds) == expected, f"{name}/{split}: got {len(ds)} images, expected {expected}"
    assert ds.num_classes == cfg["num_classes"]
    return ds


def subset_classes(ds: ImageDataset, n_classes: int, seed: int = 0) -> ImageDataset:
    """Keep every image of ``n_classes`` classes, chosen by a fixed random draw.

    This is the right way to shrink a 1000-way benchmark. Subsampling *images*
    instead would leave ~4 train and ~1 test example per class, which is far too
    few to fit or to evaluate a 1000-way linear probe. Subsampling classes keeps
    the per-class statistics intact and only narrows the label set.

    The draw is random rather than the first ``n``: ImageNet's class order is
    WordNet-derived and strongly semantically clustered (the first ~120 classes
    are almost all fish, birds and reptiles), so taking a prefix would produce a
    much harder, and unrepresentative, fine-grained problem.

    Labels are remapped to ``0..n_classes-1`` so the probe's output layer matches.
    """
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(ds.num_classes, size=n_classes, replace=False))
    remap = {int(c): i for i, c in enumerate(keep)}
    idx = [i for i, y in enumerate(ds.labels) if int(y) in remap]
    return ImageDataset(
        name=f"{ds.name}_c{n_classes}",
        split=ds.split,
        paths=[ds.paths[i] for i in idx],
        labels=np.array([remap[int(ds.labels[i])] for i in idx], dtype=np.int64),
        classnames=[ds.classnames[c] for c in keep],
        image_ids=[ds.image_ids[i] for i in idx],
    )


def classnames(name: str) -> List[str]:
    return get_dataset(name, "test").classnames


def stratified_val_split(
    labels: np.ndarray, val_fraction: float = 0.1, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Indices ``(train, val)``, stratified, fixed seed (§7).

    Every class keeps at least one validation example: on CUB a class has ~30
    train images, and rounding down would silently give some classes no val data
    and bias ``best_val_top1`` selection.
    """
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_val = max(1, int(round(val_fraction * len(idx))))
        va.append(idx[:n_val])
        tr.append(idx[n_val:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))
