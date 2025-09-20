#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
cd "$ROOT"

# Load FRED key (safe if absent for Yahoo-only paths)
if [ -f "config/secrets.env" ]; then
  # shellcheck disable=SC1091
  . "config/secrets.env"
fi

PY="$(command -v python3)"

# 1) Ingest macro (FRED)
"$PY" "$ROOT/ingest/fred_macro.py" \
  --series DTWEXBGS DGS10 T10YIE DFII10 \
  --out "$ROOT/data/raw/macro.parquet" \
  --backfill_years 15 \
  --overlap_days 5

# 2) Rebuild features to include macro (your Step 2 builder)
"$PY" "$ROOT/features/build_features_live.py" \
  --config "$ROOT/config/pipeline.json" \
  --raw_strip "$ROOT/data/raw/cl_strip.parquet" \
  --macro "$ROOT/data/raw/macro.parquet" \
  --vol "$ROOT/data/raw/vol.parquet" \
  --eia "$ROOT/data/raw/eia.parquet" \
  --out "$ROOT/data/proc/features.csv"

echo "Macro ingested and features refreshed -> $ROOT/data/proc/features.csv"
