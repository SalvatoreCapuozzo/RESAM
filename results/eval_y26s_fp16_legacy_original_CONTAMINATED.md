### trt:models/engines/yolo26s_roi960_fp16.engine@crop=roi — MasterSplits/merged_dataset_splits.json [test], mapping `original`

mAP50 **0.683** · mAP50-95 0.330 · mAP50 tiny/small/medium/large 0.402/0.611/0.723/0.789 · class-agnostic AP50 0.686 · best F1 0.645 @conf 0.25 · latency p50 41.7 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship | 103 | 0.719 | 0.268 | 0.465 | 0.677 | 0.698 | 0.650 | 0.31 | 0.786 |
| Noise | 132 | 0.467 | 0.191 | 0.252 | 0.502 | 0.706 | 0.273 | 0.16 | 0.326 |
| Unknown TGT | 91 | 0.632 | 0.255 | 0.490 | 0.620 | 0.759 | 0.484 | 0.15 | 0.549 |
| Coast/Port | 70 | 0.896 | 0.692 | – | – | 0.881 | 0.843 | 0.09 | 0.871 |
| My Ship | 93 | 0.821 | 0.378 | – | 0.752 | 0.693 | 0.849 | 0.37 | 0.946 |
| AIS Lost | 78 | 0.564 | 0.194 | – | 0.503 | 0.551 | 0.692 | 0.47 | 0.885 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Ship | Noise | Unknown TGT | Coast/Port | My Ship | AIS Lost | missed |
|---|---|---|---|---|---|---|---|
| Ship | 67 | 0 | 0 | 1 | 0 | 0 | 35 |
| Noise | 0 | 36 | 0 | 0 | 0 | 0 | 96 |
| Unknown TGT | 0 | 0 | 44 | 0 | 0 | 0 | 47 |
| Coast/Port | 0 | 0 | 0 | 59 | 0 | 0 | 11 |
| My Ship | 0 | 0 | 0 | 0 | 79 | 0 | 14 |
| AIS Lost | 0 | 0 | 0 | 1 | 0 | 54 | 23 |
| background | 29 | 15 | 14 | 6 | 35 | 44 | 0 |
