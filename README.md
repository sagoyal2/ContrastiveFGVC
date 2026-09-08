# vlm-probe

Feature-tap + linear-probe harness for the question in
[`vlm-modality-gap-probing-spec.md`](vlm-modality-gap-probing-spec.md): **where
between the vision encoder and the LM does concept–attribute signal get lost?**

The spec is the source of truth. This README is only the operating manual.

## Setup on a new machine

The assumption is that you have exactly two things: **this repo, cloned**, and the
**DataStorageDLLM mount**. Nothing else survives a machine change, and nothing else
needs to.

```bash
git clone <this repo> && cd ContrastiveFGVC
./scripts/bootstrap.sh /path/to/DataStorageDLLM
```

That creates `.venv`, installs from `pyproject.toml`, recreates the storage
symlinks, and prints what the mount already provides. It downloads almost
nothing: the mount carries ~62 GB of model weights (`hf_cache/`), the datasets,
every extracted feature, and ~7 GB of cached Python wheels (`uv_cache/`), which
`bootstrap.sh` points `UV_CACHE_DIR` at.

Persist the two environment variables it prints, so every later shell agrees on
the mount:

```bash
export VLM_PROBE_STORAGE=/path/to/DataStorageDLLM
export UV_CACHE_DIR=$VLM_PROBE_STORAGE/uv_cache
```

Only if you need LLaDA-V — it has no HF port and its modelling code is pinned to
`transformers==4.40`, which the LLaVA-1.5 HF port cannot share:

```bash
./scripts/setup_llada_env.sh   # creates .venv-llada; reuses the fork clone on the mount
```

Datasets are already on the mount. To fetch CUB from scratch:
`./scripts/download_cub.sh`.

### What lives where

| in git | on the mount |
|---|---|
| all code, configs, scripts, tests | `datasets/`, `features/`, `results/`, `figures/` |
| `pyproject.toml` (pinned) | `hf_cache/` — model weights |
| `notes/decisions.md`, `WRITEUP.md` | `uv_cache/` — Python wheels |
| the spec | `src/LLaDA-V` — the pinned fork clone |

No tracked file contains an absolute path; every script resolves the mount from
`$VLM_PROBE_STORAGE`.

## Phase 1 (CUB)

```bash
./scripts/run_phase1.sh          # extraction: encoders, then LLaVA-7B/13B
./scripts/run_phase1_probes.sh   # probes, PCA control, kNN control, attributes
python -m src.checks --dataset cub --model llava15_7b
```

`src.checks` prints the §0.5 Phase-1 gates. **Do not start Phase 2 until they
pass** — the remaining phases cost ~17.7x Phase 1's extraction, and every gate
failure would corrupt all of it silently.

## Individual commands

```bash
python -m src.extract --model llava15_7b --dataset cub --split train
python -m src.probe   --model llava15_7b --dataset cub --all-taps --head logreg
python -m src.evaluate_generative --model llava15_7b --dataset cub
python -m src.attribute_probe --model llava15_7b --dataset cub --all-taps
python -m src.figures --figure fig5 --datasets cub
python -m src.report  --table1 --table2 --dataset cub
```

## Layout

**The repo holds code and configs only — 272 KB, no data.** Everything heavy
lives under a separate storage root and is reached through four git-ignored
symlinks at the repo root:

| symlink | storage subdirectory | holds |
|---|---|---|
| `data/` | `$VLM_PROBE_STORAGE/datasets` | CUB, ImageNet-val |
| `features/` | `$VLM_PROBE_STORAGE/features` | pooled tap tensors (`.npy`) |
| `results/` | `$VLM_PROBE_STORAGE/results` | probe CSVs, generations, tables |
| `figures/` | `$VLM_PROBE_STORAGE/figures` | PDFs and PNGs |

Model weights go to `$VLM_PROBE_STORAGE/hf_cache` — `src.paths.use_storage_hf_cache()`
sets `HF_HOME` before any `transformers` import, so tens of GB never touch the
repo or the root filesystem.

The symlinks are **absolute**, which is exactly why they are git-ignored: committed,
they would dangle on every other machine. `scripts/link_storage.sh` recreates them,
and `src.paths.ensure_dirs()` does the same on any CLI entry point, so a fresh
clone self-heals on first run. Override the root with `VLM_PROBE_STORAGE`.

Also ignored: `.venv/` and `.venv-llada/` (~14 GB together), build artifacts, and
`*.npy` / `*.safetensors` / `*.jsonl` as a backstop in case a path is ever mis-set.

| | |
|---|---|
| `src/models/` | one `TapExtractor` per model; the taps of §5 come out of a single forward pass |
| `src/data/` | dataset loaders (own splits, no re-carving) and every prompt string |
| `src/extract.py` | pools inside the loop, before anything reaches disk — the rule §8 hangs on |
| `src/probe.py` | torch head (Zhang et al. B.4) and the L-BFGS fast path; never mixed in one table |
| `src/checks.py` | the §11 checks that are statements about numbers |
| `tests/test_span.py` | the §11 checks that need a live model |

## Two things to keep straight

* **T0-c is not T0.** T0-c is the contrastive embedding (last-layer CLS through
  `visual_projection`); T0 is the penultimate-layer patch grid the projector
  actually consumes. Conflating them inflates the apparent projector drop.
* **Wider probes are more expressive.** Taps have widths 1024–5120, so a higher
  score at a wider tap is not by itself evidence of more information. The
  PCA-1024 and kNN controls exist for this reason.
