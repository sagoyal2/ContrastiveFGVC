#!/usr/bin/env bash
# Phase 1 probing + experiments, after run_phase1.sh.
#
# logreg first, everywhere: it is the L-BFGS fast path (§6.2) and covers every
# tap including the sweep in minutes. The torch head (Zhang et al. B.4 verbatim)
# is then run only on the headline cells, because 500 Adam epochs x 3 seeds x
# every tap is hours for a number that moves ~0.5%. The two never share a table (§13).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DS=cub

for M in clip_l14_336 siglip2_so400m llava15_7b llava15_13b; do
  [ -f "features/$M/$DS/train/manifest.json" ] || { echo "skip $M (not extracted)"; continue; }
  echo "=== probe $M (logreg, all taps)"
  $PY -m src.probe --model "$M" --dataset "$DS" --all-taps --head logreg
done

# Headline cells with the paper's own torch head. Optional: the logreg pass above
# already covers every tap, and the two heads differ by ~0.5%.
for M in llava15_7b llava15_13b; do
  [ -f "features/$M/$DS/train/manifest.json" ] || continue
  for K in T0.avg T1.avg T2.avg T2.final T3.final T3.pos0; do
    echo "=== probe $M $K (torch linear, 3 seeds)"
    $PY -m src.probe --model "$M" --dataset "$DS" --tap "${K%%.*}" --pool "${K##*.}" --head linear
  done
done

# Experiment 3 — the attribute probe.
for M in clip_l14_336 llava15_7b llava15_13b; do
  [ -f "features/$M/$DS/train/manifest.json" ] || continue
  echo "=== attribute probe $M"
  $PY -m src.attribute_probe --model "$M" --dataset "$DS" --all-taps
done
echo "PHASE1 PROBES DONE"
