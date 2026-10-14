### trt:models/engines/yolo26s_roi960_int8mix.engine@crop=roi +track2 — MasterSplits/splits_blocked.json [test], mapping `original`

mAP50 **0.488** · mAP50-95 0.235 · mAP50 tiny/small/medium/large 0.104/0.338/0.600/0.519 · class-agnostic AP50 0.333 · best F1 0.494 @conf 0.05 · latency p50 36.4 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship | 92 | 0.427 | 0.160 | 0.104 | 0.462 | 0.623 | 0.522 | 0.36 | 0.630 |
| Noise | 339 | 0.083 | 0.036 | 0.020 | 0.047 | 0.660 | 0.091 | 0.20 | 0.130 |
| Unknown TGT | 77 | 0.387 | 0.167 | 0.188 | 0.390 | 0.679 | 0.468 | 0.21 | 0.545 |
| Coast/Port | 59 | 0.904 | 0.629 | – | 0.000 | 0.915 | 0.915 | 0.06 | 0.932 |
| My Ship | 77 | 0.883 | 0.355 | – | 1.000 | 0.922 | 0.922 | 0.07 | 0.935 |
| AIS Lost | 55 | 0.245 | 0.061 | – | 0.129 | 0.562 | 0.327 | 0.17 | 0.436 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Ship | Noise | Unknown TGT | Coast/Port | My Ship | AIS Lost | missed |
|---|---|---|---|---|---|---|---|
| Ship | 48 | 0 | 0 | 0 | 0 | 0 | 44 |
| Noise | 0 | 31 | 0 | 0 | 0 | 0 | 308 |
| Unknown TGT | 0 | 0 | 36 | 0 | 0 | 0 | 41 |
| Coast/Port | 0 | 0 | 0 | 54 | 0 | 0 | 5 |
| My Ship | 0 | 0 | 0 | 0 | 71 | 0 | 6 |
| AIS Lost | 0 | 0 | 0 | 0 | 0 | 18 | 37 |
| background | 29 | 16 | 17 | 5 | 6 | 14 | 0 |
