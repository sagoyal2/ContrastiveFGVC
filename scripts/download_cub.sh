#!/usr/bin/env bash
# CUB-200-2011 (§7). The canonical Caltech host 403s on a bare HEAD but serves
# GETs with a browser UA, so the UA below is load-bearing, not cargo cult.
#
# Unpacks to $STORAGE/datasets/cub/, which leaves images + splits in
# CUB_200_2011/ and attributes.txt one level above it — the loader accepts either
# location.
set -euo pipefail
STORAGE="${VLM_PROBE_STORAGE:-$HOME/DataStorageDLLM}"
DEST="$STORAGE/datasets/cub"
URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"

mkdir -p "$DEST"
cd "$DEST"
if [ ! -f CUB_200_2011.tgz ]; then
  curl -sSL -A "Mozilla/5.0" --retry 3 -C - -o CUB_200_2011.tgz "$URL"
fi
[ -d CUB_200_2011/images ] || tar xzf CUB_200_2011.tgz
echo "CUB at $DEST/CUB_200_2011  ($(ls CUB_200_2011/images | wc -l) class dirs)"
