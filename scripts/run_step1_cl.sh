#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/georgekurchey/CL_Model"
cd "$ROOT"

# pick python: prefer project venv, else fall back to PATH
if [ -x "$ROOT/.venv/bin/python3" ]; then
  PY="$ROOT/.venv/bin/python3"
else
  PY="$(command -v python3)"
fi

# load API keys
if [ -f "config/secrets.env" ]; then
  # shellcheck disable=SC1091
  . "config/secrets.env"
else
  echo "Missing config/secrets.env. Run: bash scripts/configure_api_keys.sh"
  exit 1
fi

# ensure ingestor exists
if [ ! -f "ingest/ndlink_cl.py" ]; then
  echo "Missing ingest/ndlink_cl.py. Please recreate it, then rerun."
  exit 1
fi

# ensure package init (not strictly needed now, but harmless)
[ -f ingest/__init__.py ] || : > ingest/__init__.py

# run the ingestor by path (avoids module import issues)
"$PY" "$ROOT/ingest/ndlink_cl.py" \
  --m 12 \
  --out "$ROOT/data/raw/cl_strip.parquet" \
  --backfill_years 10 \
  --overlap_days 10 \
  --sleep 0.25

echo "Step 1 complete. Raw strip at: $ROOT/data/raw/cl_strip.parquet (CSV fallback if Parquet unavailable)."
