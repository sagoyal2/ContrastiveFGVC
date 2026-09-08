#!/usr/bin/env bash
# LLaDA-V (V-DLM) on CUB. Runs in .venv-llada, not .venv (see setup_llada_env.sh).
#
# Smoke test first, always. This is an untested code path against a fork pinned
# to transformers 4.40, and the two things most likely to be wrong — the variable
# image-token count under anyres tiling (§13) and the span arithmetic in
# _spans_from_length — both fail silently into plausible-looking numbers rather
# than raising. Eight images cost seconds and catch them.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-llada/bin/python
DS=cub

echo "=== smoke test: 8 images"
$PY -m src.extract --model lladav_8b --dataset "$DS" --split test --limit 8 --batch-size 4 --overwrite

echo "=== full extraction (no --sweep, per notes/decisions.md)"
for SPLIT in train test; do
  $PY -m src.extract --model lladav_8b --dataset "$DS" --split "$SPLIT" --overwrite
done

echo "=== probes (core taps) + attribute probe"
# Runs in .venv-llada so it reads the features just written; sklearn/torch are
# installed there. Without this the run would end with LLaDA-V extracted but
# unanalysed, and §11 check 6 unverified.
$PY -m src.probe --model lladav_8b --dataset "$DS" --all-taps --head logreg
$PY -m src.attribute_probe --model lladav_8b --dataset "$DS" --all-taps

echo "=== §11 check 6: bidirectionality (expect T2.avg ~= T2.final)"
$PY -m src.checks --dataset "$DS" --model lladav_8b || true

echo "=== experiment 1: generative"
$PY -m src.evaluate_generative --model lladav_8b --dataset "$DS" --split test --question fine || \
  echo "!! generative failed for lladav_8b (diffusion decoding path)"

echo "LLADA RUN DONE"
