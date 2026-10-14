#!/bin/bash
# Kill a PID if available memory drops below a floor (prevents the Orin Nano from freezing on OOM).
# usage: tools/mem_watchdog.sh <pid> [min_available_mb]
PID=$1; MIN=${2:-500}
while kill -0 "$PID" 2>/dev/null; do
  AVAIL=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$AVAIL" -lt "$MIN" ]; then
    echo "$(date) MemAvailable ${AVAIL}MB < ${MIN}MB -> killing $PID" >&2; kill -9 "$PID"; exit 1
  fi
  sleep 2
done
