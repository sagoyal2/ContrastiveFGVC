#!/usr/bin/env bash
# Build the second interpreter LLaDA-V needs. Sets up only — runs no model code.
#
# Why a second venv (src/models/llada_v.py, header note 1): the HF repo
# GSAI-ML/LLaDA-V ships config.json + configuration_llada.py and the weights, but
# **not** modeling_llada.py, so `transformers` cannot load it on its own. The
# modelling code (modeling_llada.py, llava_llada.py) lives in
# github.com/ML-GSAI/LLaDA-V at train/llava/model/language_model/, inside a
# LLaVA-NeXT fork whose requirements pin transformers==4.40.0.dev0 and
# tokenizers==0.15.2. That cannot coexist with the transformers==4.56 the
# LLaVA-1.5 HF port needs, hence .venv-llada.
#
# We install the fork's `llava` package with --no-deps and then add only what
# inference needs. Its [train] extra pulls deepspeed==0.14.4, bitsandbytes==0.41
# and flash-attn==2.5.7 — all training-only, all requiring compilation against
# an old torch, none of them needed for a forward pass.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
STORAGE="${VLM_PROBE_STORAGE:-$HOME/DataStorageDLLM}"
SRC="$STORAGE/src"
VENV="$ROOT/.venv-llada"

command -v uv >/dev/null || { echo "install uv first, or run ./scripts/bootstrap.sh"; exit 1; }

# Same persisted wheel cache the main env uses, so this resolves from disk.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STORAGE/uv_cache}"

mkdir -p "$SRC"
if [ ! -d "$SRC/LLaDA-V" ]; then
  git clone --depth 1 https://github.com/ML-GSAI/LLaDA-V.git "$SRC/LLaDA-V"
fi
test -f "$SRC/LLaDA-V/train/llava/model/language_model/modeling_llada.py" \
  || { echo "modeling_llada.py missing from the clone — repo layout changed"; exit 1; }

# Idempotent: reuse an existing venv rather than failing or clearing it, so the
# script can be re-run after a dependency-resolution failure without starting over.
[ -x "$VENV/bin/python" ] || uv venv --python 3.10 "$VENV"
export VIRTUAL_ENV="$VENV"

# cu128 wheels, matching the main venv so both see the same driver.
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0

# The fork's pins, minus the training-only stack. numpy<2 is required by it.
# NOTE: requirements.txt lists transformers==4.40.0.dev0 with tokenizers==0.15.2,
# which is a git-commit build; the *released* 4.40.0 requires tokenizers>=0.19,
# and the two pins together are unsatisfiable. We pin transformers and let
# tokenizers resolve, rather than chasing the exact dev commit.
uv pip install \
  "transformers==4.40.0" "accelerate==0.29.3" \
  "timm==0.9.16" "einops==0.6.1" "einops-exts==0.0.4" \
  "sentencepiece==0.1.99" "numpy==1.26.4" "protobuf" "shortuuid" \
  "pillow" "pyyaml" "scikit-learn" "scipy" "pandas" "matplotlib" "tqdm" "tabulate"

# Rung 5 of the generative matching cascade (§6.1). Without it the cascade stops
# at rung 4 in THIS environment only, so LLaDA-V's Table 1 row would be scored by
# a shorter cascade than the LLaVA rows and the two would not be comparable.
# Pinned to 2.7.0: later releases require transformers>=4.41, which conflicts with
# the 4.40 this fork needs.
uv pip install "sentence-transformers==2.7.0"

# The modelling code itself, and this repo so `python -m src.extract` resolves.
uv pip install --no-deps -e "$SRC/LLaDA-V/train"
uv pip install --no-deps -e "$ROOT"

cat <<'MSG'

Environment built. NOT yet validated — no model code has been run.

Before the first extraction, two API mismatches in src/models/llada_v.py need
checking against this fork's builder.py:
  1. load_pretrained_model() takes torch_dtype as a STRING ("bfloat16"), not a
     torch.dtype. Passing the object falls into a pdb.set_trace() branch.
  2. It defaults to attn_implementation="flash_attention_2"; flash-attn is not
     installed here, so pass "sdpa" or "eager".

Then:
  .venv-llada/bin/python -m src.extract --model lladav_8b --dataset cub --split train
MSG
