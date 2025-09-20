#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m features.build_features --config config/pipeline.json --demo
python3 -m backtests.walkforward --config config/pipeline.json --folds 5
python3 -m backtests.scoring --input preds --out reports
echo "Report -> $(pwd)/reports/report.html"
