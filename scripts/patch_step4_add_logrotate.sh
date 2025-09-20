#!/usr/bin/env bash
set -euo pipefail
F="/Users/georgekurchey/CL_Model/scripts/run_step4_daily.sh"
if [ ! -f "$F" ]; then echo "Step-4 runner not found at $F"; exit 1; fi
if grep -q "scripts/rotate_logs.sh" "$F"; then echo "Log rotation already wired."; exit 0; fi
BK="$F.bak.$(date +%Y%m%d_%H%M%S)"
cp "$F" "$BK"
TMP="$(mktemp)"
# Prepend a call to rotate logs so it happens before any logging
{ echo '/Users/georgekurchey/CL_Model/scripts/rotate_logs.sh "/Users/georgekurchey/CL_Model" "/Users/georgekurchey/CL_Model/logs/current.log" 14'; cat "$F"; } > "$TMP"
mv "$TMP" "$F"
echo "Patched Step-4 for log rotation. Backup: $BK"
