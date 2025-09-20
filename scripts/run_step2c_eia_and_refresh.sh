#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
cd "$ROOT"

# Load key if present
[ -f "config/secrets.env" ] && . "config/secrets.env" || true
PY="$(command -v python3)"

# 1) Ingest EIA weekly inventories (short codes)
"$PY" "$ROOT/ingest/eia_weekly.py" \
  --series WCESTUS1 WGTSTUS1 WDISTUS1 \
  --out "$ROOT/data/raw/eia.parquet" \
  --backfill_years 15 \
  --overlap_weeks 6

# 2) Rebuild features (now includes EIA)
"$PY" "$ROOT/features/build_features_live.py" \
  --config "$ROOT/config/pipeline.json" \
  --raw_strip "$ROOT/data/raw/cl_strip.parquet" \
  --macro "$ROOT/data/raw/macro.parquet" \
  --vol "$ROOT/data/raw/vol.parquet" \
  --eia "$ROOT/data/raw/eia.parquet" \
  --out "$ROOT/data/proc/features.csv"

echo "EIA ingested and features refreshed -> $ROOT/data/proc/features.csv"
