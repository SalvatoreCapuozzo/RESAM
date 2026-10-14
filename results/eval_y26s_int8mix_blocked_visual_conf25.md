### trt:models/engines/yolo26s_roi960_int8mix.engine@crop=roi — MasterSplits/splits_blocked.json [test], mapping `visual`

mAP50 **0.583** · mAP50-95 0.299 · mAP50 tiny/small/medium/large 0.080/0.373/0.699/0.528 · class-agnostic AP50 0.375 · best F1 0.528 @conf 0.05 · latency p50 34.9 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Target | 224 | 0.431 | 0.165 | 0.130 | 0.404 | 0.587 | 0.558 | 1.10 | 0.674 |
| Clutter | 339 | 0.116 | 0.046 | 0.030 | 0.089 | 0.593 | 0.142 | 0.41 | 0.209 |
| Land | 59 | 0.904 | 0.629 | – | 0.000 | 0.915 | 0.915 | 0.06 | 0.932 |
| Own Ship | 77 | 0.883 | 0.355 | – | 1.000 | 0.922 | 0.922 | 0.07 | 0.935 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Target | Clutter | Land | Own Ship | missed |
|---|---|---|---|---|---|
| Target | 125 | 0 | 0 | 0 | 99 |
| Clutter | 0 | 48 | 0 | 0 | 291 |
| Land | 0 | 0 | 54 | 0 | 5 |
| Own Ship | 0 | 0 | 0 | 71 | 6 |
| background | 88 | 33 | 5 | 6 | 0 |
