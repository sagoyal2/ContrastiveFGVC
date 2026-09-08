#!/usr/bin/env bash
# Recreate the four storage symlinks after a fresh clone.
#
# The repo holds code and configs only; datasets, features, results and figures
# live under $VLM_PROBE_STORAGE (default ~/DataStorageDLLM) so the repo stays
# pushable. These symlinks exist purely so the relative paths used in the spec
# and in the shell scripts still resolve from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
STORAGE="${VLM_PROBE_STORAGE:-$HOME/DataStorageDLLM}"

for pair in "data:datasets" "features:features" "results:results" "figures:figures"; do
  link="${pair%%:*}"; target="$STORAGE/${pair##*:}"
  mkdir -p "$target"
  if [ -L "$link" ] || [ ! -e "$link" ]; then
    ln -sfn "$target" "$link"
  elif [ -d "$link" ]; then
    echo "warning: $link is a real directory, not a symlink — leaving it alone"
  fi
done
echo "storage symlinks -> $STORAGE"
ls -ld data features results figures
