#!/usr/bin/env bash
# Experiment 1 (§6.1) with the species-level question, for both LLaVA sizes,
# then LLaDA-V.
#
# Why re-run: the B.4 question "What type of object is in this photo?" was
# written for ImageNet. On CUB it draws "a bird" — 5,747 of 5,794 LLaVA-7B
# generations mentioned a bird and only 46 contained any species string, giving
# 0.97% lenient and 95.7% unmatched. That measures granularity mismatch, not
# recognition. The `fine` question asks at the granularity the label set uses.
# The default-question results stay on disk as the comparison row.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DS=cub

for M in llava15_7b llava15_13b; do
  echo "=== experiment 1 [fine]: $M"
  $PY -m src.evaluate_generative --model "$M" --dataset "$DS" --split test --question fine
done
echo "EXPERIMENT1 FINE DONE"
