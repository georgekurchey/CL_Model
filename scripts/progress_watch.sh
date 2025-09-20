#!/usr/bin/env bash
ROOT="/Users/georgekurchey/CL_Model"
LOGDIR="$ROOT/logs"
PROG="$LOGDIR/progress.json"
LOG="$LOGDIR/current.log"
echo "Watching progress at $PROG"
while true; do
  if [ -f "$PROG" ]; then
    cat "$PROG"
  else
    echo '{"stage":"(waiting)","pct":0}'
  fi
  sleep 2
done
