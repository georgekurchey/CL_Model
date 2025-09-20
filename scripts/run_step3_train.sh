#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
PY="$(command -v python3)"

if [ ! -f "$ROOT/data/proc/features.csv" ]; then
  echo "Features not found at $ROOT/data/proc/features.csv. Run Step 2 first." >&2
  exit 1
fi

echo "[step3] Walk-forward training (gbt_iso)"
"$PY" "$ROOT/backtests/walkforward_iso.py" --features "$ROOT/data/proc/features.csv" --folds 5 --val_days 126 --iso_days 252 --outdir "$ROOT/preds" --report "$ROOT/reports/wf_report.html"

echo "[step3] Train full live model (gbt_iso)"
if [ -f "$ROOT/models/save_full_iso.py" ]; then
  "$PY" "$ROOT/models/save_full_iso.py" --features "$ROOT/data/proc/features.csv" --iso_days 252
else
  echo "Warning: save_full_iso.py not found; skipping full-model save."
fi

echo "Done."
echo "WF report -> $ROOT/reports/wf_report.html"
