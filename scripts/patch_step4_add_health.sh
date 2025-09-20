\
#!/usr/bin/env bash
set -euo pipefail
F="/Users/georgekurchey/CL_Model/scripts/run_step4_daily.sh"
if [ ! -f "$F" ]; then
  echo "Step-4 runner not found at $F"; exit 1
fi
if grep -q "monitor/health.py" "$F"; then
  echo "Health probe already wired into Step-4."
  exit 0
fi
tmp="$(mktemp)"
awk '
  /Update live monitor|alerts\.py|signal dashboard|Done\./ && !x {
    print "progress \"Health probe\" 98"
    print "\"$PY\" \"/Users/georgekurchey/CL_Model/monitor/health.py\" >> \"$LOG\" 2>&1"
    x=1
  } { print }' "$F" > "$tmp" && mv "$tmp" "$F"
echo "Patched Step-4 to log latency/health near the end."
