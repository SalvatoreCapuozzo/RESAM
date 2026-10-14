### trt:models/engines/yolo26s_roi960_int8mix.engine@crop=roi — MasterSplits/splits_blocked.json [test], mapping `original`

mAP50 **0.665** · mAP50-95 0.308 · mAP50 tiny/small/medium/large 0.323/0.506/0.777/0.668 · class-agnostic AP50 0.575 · best F1 0.545 @conf 0.20 · latency p50 32.9 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship | 92 | 0.596 | 0.217 | 0.287 | 0.583 | 0.583 | 0.609 | 0.50 | 0.750 |
| Noise | 339 | 0.345 | 0.113 | 0.362 | 0.358 | 0.593 | 0.142 | 0.41 | 0.209 |
| Unknown TGT | 77 | 0.586 | 0.267 | 0.319 | 0.572 | 0.587 | 0.571 | 0.39 | 0.688 |
| Coast/Port | 59 | 0.943 | 0.663 | – | 0.000 | 0.915 | 0.915 | 0.06 | 0.932 |
| My Ship | 77 | 0.937 | 0.405 | – | 1.000 | 0.922 | 0.922 | 0.07 | 0.935 |
| AIS Lost | 55 | 0.585 | 0.180 | – | 0.524 | 0.595 | 0.455 | 0.21 | 0.527 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Ship | Noise | Unknown TGT | Coast/Port | My Ship | AIS Lost | missed |
|---|---|---|---|---|---|---|---|
| Ship | 56 | 0 | 0 | 0 | 0 | 0 | 36 |
| Noise | 0 | 48 | 0 | 0 | 0 | 0 | 291 |
| Unknown TGT | 0 | 0 | 44 | 0 | 0 | 0 | 33 |
| Coast/Port | 0 | 0 | 0 | 54 | 0 | 0 | 5 |
| My Ship | 0 | 0 | 0 | 0 | 71 | 0 | 6 |
| AIS Lost | 0 | 0 | 0 | 0 | 0 | 25 | 30 |
| background | 40 | 33 | 31 | 5 | 6 | 17 | 0 |
