### trt:models/engines/yolo26s_roi960_fp16.engine@crop=roi — MasterSplits/splits_blocked.json [test], mapping `visual`

mAP50 **0.696** · mAP50-95 0.353 · mAP50 tiny/small/medium/large 0.349/0.531/0.766/0.678 · class-agnostic AP50 0.570 · best F1 0.544 @conf 0.15 · latency p50 41.2 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Target | 224 | 0.545 | 0.214 | 0.310 | 0.530 | 0.587 | 0.527 | 1.04 | 0.688 |
| Clutter | 339 | 0.368 | 0.122 | 0.388 | 0.429 | 0.573 | 0.139 | 0.44 | 0.209 |
| Land | 59 | 0.935 | 0.678 | – | 0.167 | 0.898 | 0.898 | 0.07 | 0.915 |
| Own Ship | 77 | 0.935 | 0.397 | – | 1.000 | 0.777 | 0.948 | 0.26 | 0.948 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Target | Clutter | Land | Own Ship | missed |
|---|---|---|---|---|---|
| Target | 118 | 0 | 0 | 0 | 106 |
| Clutter | 1 | 47 | 0 | 0 | 291 |
| Land | 0 | 0 | 53 | 0 | 6 |
| Own Ship | 0 | 0 | 0 | 73 | 4 |
| background | 82 | 35 | 6 | 21 | 0 |
