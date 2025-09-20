#!/usr/bin/env bash
set -euo pipefail
F="/Users/georgekurchey/CL_Model/scripts/run_step4_daily.sh"
if grep -q 'signals/generate_signal.py' "$F"; then
  echo "Step 6 already in run_step4_daily.sh — no changes."
  exit 0
fi
cat >> "$F" <<'APPEND'

# Step 6: signals + paper-trade
progress "Generate signal" 92
"$PY" "/Users/georgekurchey/CL_Model/signals/generate_signal.py" --config "/Users/georgekurchey/CL_Model/config/signal.json" >> "$LOG" 2>&1

progress "Update signal dashboard" 95
"$PY" "/Users/georgekurchey/CL_Model/backtests/paper_trade.py" --config "/Users/georgekurchey/CL_Model/config/signal.json" --out_html "/Users/georgekurchey/CL_Model/reports/signal_dashboard.html" >> "$LOG" 2>&1
APPEND
echo "Patched: $F"
