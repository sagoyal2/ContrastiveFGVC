#!/usr/bin/env bash
# Phase 1 (§0.5): CUB only, all available models, all core taps.
#
# Order matters only for failure economics: the encoders are minutes and validate
# the probe path, so they run first and a bug surfaces before an hour of LLM
# passes has been spent.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DS=cub

for M in clip_l14_336 siglip2_so400m; do
  for SPLIT in train test; do
    echo "=== extract $M $DS $SPLIT"
    $PY -m src.extract --model "$M" --dataset "$DS" --split "$SPLIT" --overwrite
  done
done

# No --sweep: the CUB layer sweep is already extracted and is not being
# repeated (notes/decisions.md).
for M in llava15_7b llava15_13b; do
  for SPLIT in train test; do
    echo "=== extract $M $DS $SPLIT"
    $PY -m src.extract --model "$M" --dataset "$DS" --split "$SPLIT" --overwrite
  done
done
echo "PHASE1 EXTRACTION DONE"
