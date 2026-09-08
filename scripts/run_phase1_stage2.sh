#!/usr/bin/env bash
# Everything that needs the GPU after run_phase1.sh releases it:
#   * the ImageNet-val encoder reference (Phase-1 gate 2)
#   * Experiment 1, generative / zero-shot (Table 1)
# Probing is CPU-heavy and runs separately.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== gate 2: CLIP-L on ImageNet val"
for SPLIT in train test; do
  $PY -m src.extract --model clip_l14_336 --dataset imagenet_val --split "$SPLIT" --overwrite
done
$PY -m src.evaluate_generative --model clip_l14_336 --dataset imagenet_val --split test

echo "=== experiment 1: zero-shot encoders on CUB"
for M in clip_l14_336 siglip2_so400m; do
  $PY -m src.evaluate_generative --model "$M" --dataset cub --split test
done

echo "=== experiment 1: generative VLMs on CUB"
for M in llava15_7b llava15_13b; do
  $PY -m src.evaluate_generative --model "$M" --dataset cub --split test || \
    echo "!! generative eval failed for $M"
done
echo "PHASE1 STAGE2 DONE"
