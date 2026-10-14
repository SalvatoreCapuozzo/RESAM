### trt:models/engines/yolo26s_roi960_int8mix.engine@crop=roi +track2 — MasterSplits/splits_blocked.json [test], mapping `visual`

mAP50 **0.558** · mAP50-95 0.289 · mAP50 tiny/small/medium/large 0.069/0.354/0.650/0.519 · class-agnostic AP50 0.333 · best F1 0.494 @conf 0.05 · latency p50 35.4 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Target | 224 | 0.362 | 0.137 | 0.119 | 0.368 | 0.630 | 0.455 | 0.75 | 0.554 |
| Clutter | 339 | 0.083 | 0.036 | 0.020 | 0.047 | 0.660 | 0.091 | 0.20 | 0.130 |
| Land | 59 | 0.904 | 0.629 | – | 0.000 | 0.915 | 0.915 | 0.06 | 0.932 |
| Own Ship | 77 | 0.883 | 0.355 | – | 1.000 | 0.922 | 0.922 | 0.07 | 0.935 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Target | Clutter | Land | Own Ship | missed |
|---|---|---|---|---|---|
| Target | 102 | 0 | 0 | 0 | 122 |
| Clutter | 0 | 31 | 0 | 0 | 308 |
| Land | 0 | 0 | 54 | 0 | 5 |
| Own Ship | 0 | 0 | 0 | 71 | 6 |
| background | 60 | 16 | 5 | 6 | 0 |
