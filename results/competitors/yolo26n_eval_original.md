### trt:models/engines/yolo26n_roi960_fp16.engine@crop=roi — MasterSplits/splits_blocked.json [test], mapping `original`

mAP50 **0.548** · mAP50-95 0.263 · mAP50 tiny/small/medium/large 0.319/0.378/0.635/0.661 · class-agnostic AP50 0.439 · best F1 0.463 @conf 0.15 · latency p50 38.1 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship | 92 | 0.394 | 0.125 | 0.409 | 0.350 | 0.769 | 0.109 | 0.04 | 0.130 |
| Noise | 339 | 0.265 | 0.091 | 0.232 | 0.325 | 0.481 | 0.109 | 0.50 | 0.215 |
| Unknown TGT | 77 | 0.474 | 0.229 | 0.314 | 0.441 | 0.774 | 0.312 | 0.09 | 0.338 |
| Coast/Port | 59 | 0.878 | 0.646 | – | 0.000 | 0.718 | 0.864 | 0.25 | 0.864 |
| My Ship | 77 | 0.786 | 0.352 | – | 0.621 | 0.646 | 0.831 | 0.44 | 0.844 |
| AIS Lost | 55 | 0.489 | 0.138 | – | 0.533 | 0.500 | 0.073 | 0.05 | 0.145 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Ship | Noise | Unknown TGT | Coast/Port | My Ship | AIS Lost | missed |
|---|---|---|---|---|---|---|---|
| Ship | 10 | 0 | 0 | 0 | 0 | 0 | 82 |
| Noise | 0 | 37 | 0 | 0 | 0 | 0 | 302 |
| Unknown TGT | 0 | 0 | 24 | 0 | 0 | 0 | 53 |
| Coast/Port | 0 | 0 | 0 | 51 | 0 | 0 | 8 |
| My Ship | 0 | 0 | 0 | 0 | 64 | 0 | 13 |
| AIS Lost | 0 | 0 | 0 | 0 | 0 | 4 | 51 |
| background | 3 | 40 | 7 | 20 | 35 | 4 | 0 |
