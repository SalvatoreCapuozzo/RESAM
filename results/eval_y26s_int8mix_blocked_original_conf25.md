### trt:models/engines/yolo26s_roi960_int8mix.engine@crop=roi — MasterSplits/splits_blocked.json [test], mapping `original`

mAP50 **0.533** · mAP50-95 0.251 · mAP50 tiny/small/medium/large 0.123/0.363/0.680/0.528 · class-agnostic AP50 0.375 · best F1 0.528 @conf 0.05 · latency p50 34.6 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship | 92 | 0.489 | 0.184 | 0.097 | 0.491 | 0.583 | 0.609 | 0.50 | 0.750 |
| Noise | 339 | 0.116 | 0.046 | 0.030 | 0.089 | 0.593 | 0.142 | 0.41 | 0.209 |
| Unknown TGT | 77 | 0.461 | 0.205 | 0.242 | 0.435 | 0.587 | 0.571 | 0.39 | 0.688 |
| Coast/Port | 59 | 0.904 | 0.629 | – | 0.000 | 0.915 | 0.915 | 0.06 | 0.932 |
| My Ship | 77 | 0.883 | 0.355 | – | 1.000 | 0.922 | 0.922 | 0.07 | 0.935 |
| AIS Lost | 55 | 0.344 | 0.087 | – | 0.165 | 0.595 | 0.455 | 0.21 | 0.527 |

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
