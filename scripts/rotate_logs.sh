#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/Users/georgekurchey/CL_Model}"
LOG="${2:-$ROOT/logs/current.log}"
KEEP="${3:-14}"
ARCH="$ROOT/logs/archive"
mkdir -p "$ARCH" "$(dirname "$LOG")"
if [ -f "$LOG" ]; then
  ts="$(date +"%Y%m%d_%H%M%S")"
  mv "$LOG" "$ARCH/step4_${ts}.log"
fi
# prune: keep last $KEEP most recent archives
if ls "$ARCH"/step4_*.log >/dev/null 2>&1; then
  cd "$ARCH"
  ls -1t step4_*.log | awk "NR>$KEEP" | xargs -I{} rm -f "{}" || true
fi
