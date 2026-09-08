"""Unpack the clip-benchmark webdataset shards into a flat image tree.

Phase 1 needs ImageNet **val only**, for the one external reference number in the
spec (§0.5: CLIP-L linear probe ~85%). ILSVRC/imagenet-1k is gated;
clip-benchmark/wds_imagenet1k is the same 50k val images, ungated, and is the
set the CLIP-benchmark numbers everyone quotes are computed on.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from src.paths import DATASETS_DIR

SRC = DATASETS_DIR / "imagenet_val_wds"
DST = DATASETS_DIR / "imagenet_val"


def main() -> None:
    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "classnames.txt").write_text((SRC / "classnames.txt").read_text())

    rows, n = [], 0
    for tar_path in sorted((SRC / "test").glob("*.tar")):
        with tarfile.open(tar_path) as t:
            pending = {}
            for m in t:
                stem, _, ext = m.name.rpartition(".")
                pending.setdefault(stem, {})[ext] = t.extractfile(m).read()
                d = pending[stem]
                if "cls" in d and "jpg" in d:
                    cls = int(d["cls"].decode().strip())
                    out = DST / "images" / f"{cls:04d}"
                    out.mkdir(exist_ok=True)
                    (out / f"{stem}.jpg").write_bytes(d["jpg"])
                    rows.append(f"{cls:04d}/{stem}.jpg {cls}")
                    pending.pop(stem)
                    n += 1
                    if n % 10000 == 0:
                        print(f"  {n}", flush=True)
    (DST / "index.txt").write_text("\n".join(rows) + "\n")
    print(f"{n} images -> {DST}")


if __name__ == "__main__":
    main()
