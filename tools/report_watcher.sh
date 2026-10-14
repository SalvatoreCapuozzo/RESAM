#!/bin/bash
# Regenerates docs/BENCHMARK.md whenever a competitor finishes (tables + charts, CPU only, safe
# while the next model trains); after ALL_DONE renders the inference images with every engine.
#   setsid nohup bash tools/report_watcher.sh > models/report_watcher.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
LOG=models/competitors.log
PY=.venv/bin/python
seen=-1
while true; do
  n=$(grep -c "^   done " "$LOG" 2>/dev/null)
  if [ "$n" != "$seen" ]; then
    echo "$(date '+%F %T') $n competitor(s) done -> refreshing tables/charts"
    nice -n 10 $PY -W ignore tools/make_benchmark_report.py --no-images && seen=$n
  fi
  if grep -q ALL_DONE "$LOG" 2>/dev/null; then
    echo "$(date '+%F %T') ALL_DONE -> dashboard screenshots"
    bash tools/dashboard_screenshots.sh
    echo "$(date '+%F %T') -> full report with inference images"
    $PY -W ignore tools/make_benchmark_report.py
    echo "$(date '+%F %T') report complete"
    exit 0
  fi
  sleep 60
done
