### trt:models/engines/yolo26s_roi960_fp16.engine@crop=roi — MasterSplits/splits_blocked.json [test], mapping `original`

mAP50 **0.641** · mAP50-95 0.304 · mAP50 tiny/small/medium/large 0.349/0.540/0.741/0.678 · class-agnostic AP50 0.570 · best F1 0.544 @conf 0.15 · latency p50 40.0 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship | 92 | 0.569 | 0.219 | 0.369 | 0.535 | 0.676 | 0.522 | 0.29 | 0.652 |
| Noise | 339 | 0.368 | 0.122 | 0.388 | 0.429 | 0.573 | 0.139 | 0.44 | 0.209 |
| Unknown TGT | 77 | 0.566 | 0.266 | 0.289 | 0.556 | 0.592 | 0.545 | 0.36 | 0.662 |
| Coast/Port | 59 | 0.935 | 0.678 | – | 0.167 | 0.898 | 0.898 | 0.07 | 0.915 |
| My Ship | 77 | 0.935 | 0.397 | – | 1.000 | 0.777 | 0.948 | 0.26 | 0.948 |
| AIS Lost | 55 | 0.475 | 0.143 | – | 0.554 | 0.475 | 0.509 | 0.39 | 0.782 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Ship | Noise | Unknown TGT | Coast/Port | My Ship | AIS Lost | missed |
|---|---|---|---|---|---|---|---|
| Ship | 48 | 0 | 0 | 0 | 0 | 0 | 44 |
| Noise | 0 | 47 | 1 | 0 | 0 | 0 | 291 |
| Unknown TGT | 0 | 0 | 42 | 0 | 0 | 0 | 35 |
| Coast/Port | 0 | 0 | 0 | 53 | 0 | 0 | 6 |
| My Ship | 0 | 0 | 0 | 0 | 73 | 0 | 4 |
| AIS Lost | 0 | 0 | 0 | 0 | 0 | 28 | 27 |
| background | 23 | 35 | 28 | 6 | 21 | 31 | 0 |
