#!/usr/bin/env bash
set -euo pipefail
cd /Users/georgekurchey/CL_Model
mkdir -p backtests models preds reports
[ -f backtests/__init__.py ] || : > backtests/__init__.py
[ -f models/__init__.py ] || : > models/__init__.py
rm -f /Users/georgekurchey/CL_Model/preds/preds_fold*.csv 2>/dev/null || true
python3 -m features.build_features --config /Users/georgekurchey/CL_Model/config/pipeline.json --demo
python3 -m backtests.walkforward_iso --folds 5 --target_low 0.0360 --target_high 0.9475 --cal_frac 0.60 --state_aware vol_terciles --tau_smooth 0.25 --tau_max_step 0.02 --sigma_shrink_low 0.00 --sigma_shrink_high 0.25
python3 -m backtests.scoring --input /Users/georgekurchey/CL_Model/preds --out /Users/georgekurchey/CL_Model/reports
open /Users/georgekurchey/CL_Model/reports/report.html
