#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/georgekurchey/CL_Model"

# use your venv if present
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
PY="$(command -v python3)"

echo "[step6] generate_signal ..."
"$PY" "$ROOT/signals/generate_signal.py" --config "$ROOT/config/signal.json"

echo "[step6] paper_trade ..."
"$PY" "$ROOT/backtests/paper_trade.py"   --config "$ROOT/config/signal.json"   --out_html "$ROOT/reports/signal_dashboard.html"

echo "[step6] done."
echo "Signal   -> $ROOT/signals/live_signal.csv"
echo "History  -> $ROOT/signals/signal_history.csv"
echo "Dashboard-> $ROOT/reports/signal_dashboard.html"
