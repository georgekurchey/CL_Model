#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
PY="$(command -v python3)"

FEAT="$ROOT/data/proc/features.csv"
if [ ! -f "$FEAT" ]; then
  echo "Features not found at $FEAT. Run Step 2 first." >&2
  exit 1
fi

echo "[step3-cqr] Walk-forward: Iso + Split-Conformal tails"
"$PY" "$ROOT/backtests/walkforward_iso_cqr.py"   --features "$FEAT"   --folds 5 --val_days 126   --n_estimators 600 --max_depth 3 --learning_rate 0.05 --min_samples_leaf 80   --calib_cfg "$ROOT/config/calibration.json"   --outdir "$ROOT/preds"   --report "$ROOT/reports/wf_cqr_report.html"

echo "WF (CQR) report -> $ROOT/reports/wf_cqr_report.html"
