### trt:models/engines/yolo26n_roi960_fp16.engine@crop=roi — MasterSplits/splits_blocked.json [test], mapping `visual`

mAP50 **0.587** · mAP50-95 0.311 · mAP50 tiny/small/medium/large 0.233/0.331/0.656/0.661 · class-agnostic AP50 0.439 · best F1 0.463 @conf 0.15 · latency p50 28.3 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Target | 224 | 0.420 | 0.156 | 0.234 | 0.378 | 0.731 | 0.170 | 0.17 | 0.205 |
| Clutter | 339 | 0.265 | 0.091 | 0.232 | 0.325 | 0.481 | 0.109 | 0.50 | 0.215 |
| Land | 59 | 0.878 | 0.646 | – | 0.000 | 0.718 | 0.864 | 0.25 | 0.864 |
| Own Ship | 77 | 0.786 | 0.352 | – | 0.621 | 0.646 | 0.831 | 0.44 | 0.844 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Target | Clutter | Land | Own Ship | missed |
|---|---|---|---|---|---|
| Target | 38 | 0 | 0 | 0 | 186 |
| Clutter | 0 | 37 | 0 | 0 | 302 |
| Land | 0 | 0 | 51 | 0 | 8 |
| Own Ship | 0 | 0 | 0 | 64 | 13 |
| background | 14 | 40 | 20 | 35 | 0 |
