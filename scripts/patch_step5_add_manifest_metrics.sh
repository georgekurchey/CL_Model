#!/usr/bin/env bash
set -euo pipefail
F="/Users/georgekurchey/CL_Model/scripts/run_step5_monitor.sh"
if [ ! -f "$F" ]; then
  echo "Step-5 runner not found at $F"; exit 1
fi
if grep -q "metrics_manifest.py" "$F"; then
  echo "Manifest metrics already wired into Step-5."
  exit 0
fi
BK="$F.bak.$(date +%Y%m%d_%H%M%S)"
cp "$F" "$BK"
TMP="$(mktemp)"
awk '
  /alerts.py/ && !x {
    print ""$PY" "/Users/georgekurchey/CL_Model/monitor/metrics_manifest.py" >> "$LOG" 2>&1"
    print "echo "[Step5] metrics_manifest.py ...""
    print ""$PY" "/Users/georgekurchey/CL_Model/monitor/metrics_manifest.py" >> "$LOG" 2>&1 || true"
    x=1
  } { print }' "$F" > "$TMP" && mv "$TMP" "$F"
echo "Patched Step-5 to compute manifest metrics before alerts. Backup: $BK"
