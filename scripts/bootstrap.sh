#!/usr/bin/env bash
# One command to go from `git clone` to a working setup on a fresh machine.
#
# The contract this assumes: you have (a) this repo, cloned, and (b) the
# DataStorageDLLM mount. Nothing else survives. Everything heavy — datasets,
# extracted features, results, figures, ~62 GB of model weights, ~7 GB of cached
# Python wheels, and the pinned LLaDA-V fork clone — already lives on that mount,
# so this script downloads almost nothing.
#
#   ./scripts/bootstrap.sh [/path/to/DataStorageDLLM]
#
# The storage root is taken from, in order: the first argument, $VLM_PROBE_STORAGE,
# then ~/DataStorageDLLM.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

STORAGE="${1:-${VLM_PROBE_STORAGE:-$HOME/DataStorageDLLM}}"
STORAGE="$(cd "$STORAGE" 2>/dev/null && pwd)" || {
  echo "storage root not found: ${1:-${VLM_PROBE_STORAGE:-$HOME/DataStorageDLLM}}"
  echo "pass it explicitly:  ./scripts/bootstrap.sh /path/to/DataStorageDLLM"
  exit 1
}
export VLM_PROBE_STORAGE="$STORAGE"
echo "storage root: $STORAGE"

# The wheel cache lives on the mount, so a rebuild resolves from disk instead of
# re-downloading ~7 GB. Exporting it here rather than relying on the shell is the
# difference between a 30-second rebuild and a long one.
export UV_CACHE_DIR="$STORAGE/uv_cache"

command -v uv >/dev/null || {
  echo "installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

echo "=== main environment"
# Idempotent: `uv venv` errors out if one already exists, so reuse it. Re-running
# bootstrap after a failure should continue, not start over.
[ -x "$ROOT/.venv/bin/python" ] || uv venv
VIRTUAL_ENV="$ROOT/.venv" uv pip install -e ".[dev,match]"

echo "=== storage symlinks"
./scripts/link_storage.sh

echo "=== what the mount already provides"
for d in datasets features results figures hf_cache uv_cache src; do
  if [ -d "$STORAGE/$d" ]; then
    printf '  %-10s %s\n' "$d" "$(du -sh "$STORAGE/$d" 2>/dev/null | cut -f1)"
  else
    printf '  %-10s MISSING\n' "$d"
  fi
done

cat <<EOF

Done. Add this to your shell profile so every session agrees on the mount:

  export VLM_PROBE_STORAGE=$STORAGE
  export UV_CACHE_DIR=$STORAGE/uv_cache

Verify (no GPU needed):
  .venv/bin/python -m pytest tests/test_shapes.py -q
  .venv/bin/python -m src.checks --dataset cub --model llava15_7b

LLaDA-V needs a second interpreter (transformers 4.40 vs 4.56). Only if you need it:
  ./scripts/setup_llada_env.sh
EOF
