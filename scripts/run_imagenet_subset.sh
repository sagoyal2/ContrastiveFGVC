#!/usr/bin/env bash
# ImageNet-val, 100-class subsample (§7 coarse-grained contrast).
#
# 100 randomly drawn classes with ALL their images: 4,000 train / 1,000 test.
# Subsampling classes, not images — 10% of images across 1000 classes would leave
# 4 train and 1 test example per class, far too few to fit or evaluate a 1000-way
# probe. See src/data/registry.subset_classes.
#
# NOTE ON COMPARABILITY: this is a 100-way problem, so its top-1 is NOT comparable
# to Zhang et al.'s 77.1% (1000-way at T3.final). Chance is 1% here, not 0.1%, and
# the numbers will be substantially higher. This run tests the pipeline on coarse
# categories and gives the coarse-vs-fine contrast against CUB; it does not
# provide the external anchor. That still needs the full 1000-class run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DS=imagenet_val
NC=100

for M in llava15_7b; do
  for SPLIT in train test; do
    echo "=== extract $M $DS ($NC classes) $SPLIT"
    $PY -m src.extract --model "$M" --dataset "$DS" --split "$SPLIT" --classes "$NC" --overwrite
  done
  echo "=== probe $M (core taps)"
  $PY -m src.probe --model "$M" --dataset "${DS}_c${NC}" --all-taps --head logreg
  echo "=== experiment 1: generative $M"
  $PY -m src.evaluate_generative --model "$M" --dataset "$DS" --split test --classes "$NC"
done
echo "IMAGENET SUBSET DONE"
