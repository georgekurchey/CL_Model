#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
cd "$ROOT"

PY="$(command -v python3)"

# Build features from the CL strip (macro/EIA/vol are optional; included if files exist)
"$PY" "$ROOT/features/build_features_live.py" \
  --config "$ROOT/config/pipeline.json" \
  --raw_strip "$ROOT/data/raw/cl_strip.parquet" \
  --macro "$ROOT/data/raw/macro.parquet" \
  --vol "$ROOT/data/raw/vol.parquet" \
  --eia "$ROOT/data/raw/eia.parquet" \
  --out "$ROOT/data/proc/features.csv"

echo "Step 2 complete. Features at: $ROOT/data/proc/features.csv"
