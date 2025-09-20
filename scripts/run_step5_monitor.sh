#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="$LOGDIR/step5_${TS}.log"
echo "[Step5] Start $(date)" | tee -a "$LOG"

# Load secrets and venv
[ -f "$ROOT/config/secrets.env" ] && . "$ROOT/config/secrets.env" || true
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
PY="$(command -v python3)"

MET="$ROOT/reports/monitor_metrics.json"

echo "[Step5] metrics.py ..." | tee -a "$LOG"
set -o pipefail
"$PY" "$ROOT/monitor/metrics.py" --config "$ROOT/config/monitor.json" --out "$MET" 2>&1 | tee -a "$LOG"

if [ ! -f "$MET" ]; then
  echo "[Step5] WARNING: metrics file not found; writing minimal placeholder." | tee -a "$LOG"
  "$PY" - <<PYEOF
from pathlib import Path
import json
Path("$MET").parent.mkdir(parents=True, exist_ok=True)
Path("$MET").write_text(json.dumps({"asof":"", "counts":{"n_realized":0},"note":"placeholder"}, indent=2))
print("Wrote placeholder ->", "$MET")
PYEOF
fi

echo "[Step5] metrics_manifest.py ..."
"$PY" "/Users/georgekurchey/CL_Model/monitor/metrics_manifest.py" >> "$LOG" 2>&1 || true
echo "[Step5] alerts.py ..." | tee -a "$LOG"
"$PY" "$ROOT/monitor/alerts.py" --config "$ROOT/config/monitor.json" --metrics "$MET" 2>&1 | tee -a "$LOG"

echo "[Step5] Done. See $LOG"
