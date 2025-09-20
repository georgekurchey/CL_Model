#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
if [ -f "$ROOT/config/secrets.env" ]; then set -a; . "$ROOT/config/secrets.env"; set +a; fi
PY="$(command -v python3)"
"$PY" "$ROOT/ingest/yf_brent.py" --ticker "BZ=F" --start "2015-01-01" --out "$ROOT/data/external/brent.csv"
