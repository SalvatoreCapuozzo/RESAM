#!/bin/bash
# Full-page dashboard screenshots of live inference on this Jetson (headless Firefox).
# Replays *test-split* frames only (never seen in training), so the on-page live accuracy is honest.
#   bash tools/dashboard_screenshots.sh [engine-spec]
cd "$(dirname "$0")/.." || exit 1
ENGINE=${1:-trt:models/engines/yolo26s_roi960_int8mix.engine@crop=roi}
PY=.venv/bin/python
OUT=docs/img/dashboard; mkdir -p "$OUT"
$PY - <<'PYEOF'
import json, os
from resam.splits import frame_key
sp = json.load(open("MasterSplits/splits_blocked.json"))
scen = {"coast_orange": "IMMAGINI/Cartour delta 00.30/202109150030#82",
        "clutter_green": "RealImages#53", "traffic_orange": "RealImages#70",
        "open_sea_orange": "IMMAGINI/Cartour delta 1.30/202109150130#82"}
for name, sess in scen.items():
    d = os.path.join("models", "replay", name); os.makedirs(d, exist_ok=True)
    for f in os.listdir(d): os.remove(os.path.join(d, f))
    frames = sorted([b for b in sp["test"] if frame_key(b)[0] == sess], key=frame_key)
    for i, b in enumerate(frames):
        for ext in (".png", ".json"):
            os.symlink(os.path.abspath(b + ext), os.path.join(d, f"{i:03d}_{os.path.basename(b)}{ext}"))
    print(name, len(frames), "test frames")
PYEOF
for scen in coast_orange clutter_green traffic_orange open_sea_orange; do
  n=$(ls models/replay/$scen/*.png | wc -l)
  $PY -W ignore tools/dashboard.py --source models/replay/$scen --fps 1 --no-drop --engine "$ENGINE" \
      --show-gt --port 8095 > models/replay/$scen.log 2>&1 &
  DPID=$!
  # let it process the whole sequence once (tracks, live metrics, charts fill up)
  for i in $(seq 1 $((n + 20))); do
    sleep 1
    f=$(curl -s localhost:8095/stats.json 2>/dev/null | $PY -c "import json,sys;print(json.load(sys.stdin).get('frames_processed') or 0)" 2>/dev/null)
    [ "${f:-0}" -ge "$n" ] && break
  done
  timeout 90 firefox --headless --window-size=1600,1250 --screenshot "$PWD/$OUT/dashboard_$scen.png" \
      "http://localhost:8095/?snap=1" > /dev/null 2>&1
  kill $DPID 2>/dev/null; wait $DPID 2>/dev/null
  ls -la "$OUT/dashboard_$scen.png" 2>/dev/null || echo "screenshot failed for $scen"
done
