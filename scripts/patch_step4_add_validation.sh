\
#!/usr/bin/env bash
set -euo pipefail
F="/Users/georgekurchey/CL_Model/scripts/run_step4_daily.sh"
if [ ! -f "$F" ]; then
  echo "Step-4 runner not found at $F"; exit 1
fi
if grep -q "scripts/validate_features.py" "$F"; then
  echo "Validation already wired into Step-4."
  exit 0
fi
tmp="$(mktemp)"
awk '
  /Train\/refresh model if stale/ && !x {
    print "progress \"Validate features\" 60"
    print "python3 \"/Users/georgekurchey/CL_Model/scripts/validate_features.py\" >> \"$LOG\" 2>&1"
    x=1
  } { print }' "$F" > "$tmp" && mv "$tmp" "$F"
echo "Patched Step-4 to run feature validation before training."
