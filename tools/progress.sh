#!/bin/bash
# One-glance status of the competitor pipeline.   usage: bash tools/progress.sh
#   live:  watch -n 10 bash tools/progress.sh
cd "$(dirname "$0")/.." || exit 1
export LC_ALL=C
LOG=models/competitors.log
cur=$(grep "^=====" "$LOG" 2>/dev/null | tail -1 | tr -d '= ')
step=$(grep '^\$ ' "$LOG" | tail -1)
case "$step" in
  *train_yolo.py*) phase="training" ;;
  *"-c import json"*) phase="test-set validation" ;;
  *export_trt.py*) phase="TensorRT build (silent, 10-15 min)" ;;
  *benchmark.py*) phase="speed benchmark" ;;
  *evaluate.py*) phase="accuracy evaluation" ;;
  *) phase="-" ;;
esac
grep -q ALL_DONE "$LOG" && phase="ALL DONE"
printf "%-12s %-10s %-9s %-14s %s\n" MODEL STATE EPOCH BEST_VAL_mAP50 NOTE
for m in yolo26s yolo26n yolo11s yolov8s yolo26s-seg yolo11s-seg yolo26s-p2 rtdetr-l; do
  run=models/train/${m}_roi960_blocked; [ "$m" = yolo26s ] && run=models/train/yolo26s_roi960_blocked
  csv=$run/results.csv
  ep=$( [ -f "$csv" ] && echo "$(($(wc -l < "$csv") - 1))/40" || echo "-")
  map=$( [ -f "$csv" ] && awk -F, 'NR>1 && $8+0>m {m=$8+0} END {printf "%.3f", m}' "$csv" || echo "-")
  if [ -f results/competitors/${m}_eval_visual.json ]; then st="done"
  elif [ "$m" = "$cur" ]; then st="running"
  elif grep -q "\"$m\"" results/competitors/status.json 2>/dev/null; then st="stopped"
  else st="queued"; fi
  note=""; [ "$m" = "$cur" ] && note="$phase"
  [ "$st" = done ] && [ "${ep%%/*}" != 40 ] && note="early-stopped (no gain for 12 epochs)"
  [ "$st" = stopped ] && note=$(grep -A3 "\"$m\"" results/competitors/status.json | grep -o '"train": "[a-z0-9]*"')
  printf "%-12s %-10s %-9s %-14s %s\n" "$m" "$st" "$ep" "$map" "$note"
done
echo "RAM free: $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB (job is killed below 300)"
