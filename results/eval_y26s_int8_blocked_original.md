### trt:models/engines/yolo26s_roi960_int8.engine@crop=roi — MasterSplits/splits_blocked.json [test], mapping `original`

mAP50 **0.341** · mAP50-95 0.171 · mAP50 tiny/small/medium/large 0.167/0.271/0.447/0.571 · class-agnostic AP50 0.283 · best F1 0.262 @conf 0.05 · latency p50 34.8 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship | 92 | 0.188 | 0.092 | 0.351 | 0.138 | 1.000 | 0.054 | 0.00 | 0.054 |
| Noise | 339 | 0.114 | 0.038 | 0.080 | 0.109 | 0.333 | 0.003 | 0.03 | 0.009 |
| Unknown TGT | 77 | 0.139 | 0.054 | 0.069 | 0.166 | 0.143 | 0.013 | 0.07 | 0.013 |
| Coast/Port | 59 | 0.743 | 0.488 | – | 0.000 | 0.871 | 0.458 | 0.05 | 0.458 |
| My Ship | 77 | 0.705 | 0.321 | – | 1.000 | 0.917 | 0.286 | 0.03 | 0.286 |
| AIS Lost | 55 | 0.158 | 0.035 | – | 0.212 | 0.000 | 0.000 | 0.00 | 0.000 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Ship | Noise | Unknown TGT | Coast/Port | My Ship | AIS Lost | missed |
|---|---|---|---|---|---|---|---|
| Ship | 5 | 0 | 0 | 0 | 0 | 0 | 87 |
| Noise | 0 | 1 | 0 | 0 | 0 | 0 | 338 |
| Unknown TGT | 0 | 0 | 1 | 0 | 0 | 0 | 76 |
| Coast/Port | 0 | 0 | 0 | 27 | 0 | 0 | 32 |
| My Ship | 0 | 0 | 0 | 0 | 22 | 0 | 55 |
| AIS Lost | 0 | 0 | 0 | 0 | 0 | 0 | 55 |
| background | 0 | 2 | 6 | 4 | 2 | 0 | 0 |
