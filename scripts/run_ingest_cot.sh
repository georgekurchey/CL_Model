#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
if [ -f "$ROOT/config/secrets.env" ]; then set -a; . "$ROOT/config/secrets.env"; set +a; fi
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PY="$(command -v python3)"
"$PY" "$ROOT/ingest/cftc_cot.py" --start "2015-01-01" --out "$ROOT/data/external/cot.csv"
