#!/usr/bin/env bash
# Download a few Waymo Open Motion Dataset shards into data/raw/waymo/.
#
# One-time, per person:
#   1. Sign in at https://waymo.com/open/ with a Google account and accept the license terms.
#   2. Run `gcloud auth login` with that same Google account.
#   3. On https://waymo.com/open/download/, open the Motion dataset, go to the "scenario" folder
#      (uncompressed), and copy its gs:// path (the folder that contains training/ and validation/).
#
# Usage:
#   export WAYMO_MOTION_BUCKET="gs://<path you copied>"
#   bash scripts/download_waymo.sh            # 3 training shards + 1 validation shard (default)
#   N_TRAIN=10 bash scripts/download_waymo.sh # more training shards
set -euo pipefail

BUCKET="${WAYMO_MOTION_BUCKET:?Set WAYMO_MOTION_BUCKET first (see the comments at the top of this script)}"
N_TRAIN="${N_TRAIN:-3}"
OUT="data/raw/waymo"
mkdir -p "$OUT/training" "$OUT/validation"

echo "Listing shards in $BUCKET ..."
TRAIN=$(gsutil ls "$BUCKET/training/" | head -n "$N_TRAIN")
VAL=$(gsutil ls "$BUCKET/validation/" | head -n 1)

gsutil -m cp $TRAIN "$OUT/training/"
gsutil -m cp $VAL "$OUT/validation/"

echo
du -sh "$OUT"/*
echo "Done. Next: python scripts/build_waymo_protos.py && python scripts/waymo_to_parquet.py"
