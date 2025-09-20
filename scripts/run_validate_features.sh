#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
OUTJ="$ROOT/reports/feature_validation.json"
OUTH="$ROOT/reports/feature_validation.html"
python3 "$ROOT/scripts/validate_features.py"
echo "Reports -> $OUTJ ; $OUTH"
