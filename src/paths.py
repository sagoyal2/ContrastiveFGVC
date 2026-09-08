"""Single source of truth for where things live on disk.

Everything heavy — datasets, HF weight cache, pooled features, results, figures —
lives under ``DataStorageDLLM``. The repo itself holds only code and configs, and
carries symlinks (``data/``, ``features/``, ``results/``, ``figures/``) pointing
into the storage root so relative paths in the spec still resolve.

Override the storage root with ``VLM_PROBE_STORAGE`` if you move the mount.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STORAGE_ROOT = Path(
    os.environ.get("VLM_PROBE_STORAGE", Path.home() / "DataStorageDLLM")
).resolve()

DATASETS_DIR = STORAGE_ROOT / "datasets"
FEATURES_DIR = STORAGE_ROOT / "features"
RESULTS_DIR = STORAGE_ROOT / "results"
FIGURES_DIR = STORAGE_ROOT / "figures"
HF_CACHE_DIR = STORAGE_ROOT / "hf_cache"
LOGS_DIR = STORAGE_ROOT / "logs"

CONFIGS_DIR = REPO_ROOT / "configs"
MODEL_CONFIGS_DIR = CONFIGS_DIR / "models"
DATASET_CONFIGS_DIR = CONFIGS_DIR / "datasets"

_ALL = (DATASETS_DIR, FEATURES_DIR, RESULTS_DIR, FIGURES_DIR, HF_CACHE_DIR, LOGS_DIR)

# repo-relative name -> storage subdirectory. These are git-ignored (they are
# absolute symlinks and would dangle on any other machine), so a fresh clone has
# to recreate them; doing it here means any CLI entry point fixes it up, and
# scripts/link_storage.sh does the same thing standalone.
_LINKS = {"data": DATASETS_DIR, "features": FEATURES_DIR,
          "results": RESULTS_DIR, "figures": FIGURES_DIR}


def ensure_dirs() -> None:
    for d in _ALL:
        d.mkdir(parents=True, exist_ok=True)
    link_repo_dirs()


def link_repo_dirs() -> None:
    """Point the repo-root convenience symlinks at the storage mount.

    Only ever touches a missing entry or an existing symlink — a real directory
    of the same name is left alone rather than replaced, since that would be a
    destructive surprise for anyone who had put something there.
    """
    for name, target in _LINKS.items():
        link = REPO_ROOT / name
        if link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        elif link.exists():
            continue  # a real directory; not ours to replace
        link.symlink_to(target, target_is_directory=True)


def use_storage_hf_cache() -> None:
    """Point every HF download at the storage mount.

    Model weights are tens of GB; the root filesystem should never see them.
    Call this before importing/loading anything from ``transformers``.
    """
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))


def feature_dir(model: str, dataset: str, split: str) -> Path:
    """``features/<model>/<dataset>/<split>/`` — the §8.4 layout."""
    return FEATURES_DIR / model / dataset / split


def dataset_dir(dataset: str) -> Path:
    return DATASETS_DIR / dataset
