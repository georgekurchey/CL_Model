#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
python3 "$ROOT/scripts/generate_manifest.py" --features "$ROOT/data/proc/features.csv" --out "$ROOT/config/features_manifest.json"
echo "Manifest written -> $ROOT/config/features_manifest.json"
