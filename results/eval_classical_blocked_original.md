### classical — MasterSplits/splits_blocked.json [test], mapping `original`

mAP50 **0.077** · mAP50-95 0.033 · mAP50 tiny/small/medium/large 0.000/0.000/0.000/0.173 · class-agnostic AP50 0.015 · best F1 0.060 @conf 0.40 · latency p50 7.8 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship | 92 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 9.25 | 0.228 |
| Noise | 339 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |
| Unknown TGT | 77 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |
| Coast/Port | 59 | 0.463 | 0.195 | – | 0.000 | 0.706 | 0.610 | 0.19 | 0.610 |
| My Ship | 77 | 0.000 | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |
| AIS Lost | 55 | 0.000 | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Ship | Noise | Unknown TGT | Coast/Port | My Ship | AIS Lost | missed |
|---|---|---|---|---|---|---|---|
| Ship | 0 | 0 | 0 | 0 | 0 | 0 | 92 |
| Noise | 2 | 0 | 0 | 0 | 0 | 0 | 337 |
| Unknown TGT | 8 | 0 | 0 | 0 | 0 | 0 | 69 |
| Coast/Port | 3 | 0 | 0 | 36 | 0 | 0 | 20 |
| My Ship | 1 | 0 | 0 | 8 | 0 | 0 | 68 |
| AIS Lost | 2 | 0 | 0 | 0 | 0 | 0 | 53 |
| background | 724 | 0 | 0 | 7 | 0 | 0 | 0 |
