#!/usr/bin/env bash
# LLaDA-V in masked mode (§9), k = 32 [MASK] tokens in the answer span.
#
# Clean mode taps the prompt sequence with no masks — structurally comparable to
# LLaVA, but off-policy: LLaDA-V never runs that configuration at inference.
# Masked mode appends k [MASK] tokens and taps their positions, which IS the state
# the diffusion decoder reads. The trade-off (§9) is that k is a free parameter and
# LLaVA has no analogue, so Tmask must be reported as its own row, not merged into
# the T3 column.
#
# Features land in features/lladav_8b_masked32/ so the clean-mode numbers are
# untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-llada/bin/python
DS=cub
K=32
TAG="lladav_8b_masked${K}"

for SPLIT in train test; do
  echo "=== extract $TAG $DS $SPLIT"
  $PY -m src.extract --model lladav_8b --dataset "$DS" --split "$SPLIT" \
      --llada-mode masked --llada-k "$K" --overwrite
done

echo "=== probes"
$PY -m src.probe --model "$TAG" --dataset "$DS" --all-taps --head logreg
echo "=== attribute probe"
$PY -m src.attribute_probe --model "$TAG" --dataset "$DS" --all-taps
echo "LLADA MASKED DONE"
