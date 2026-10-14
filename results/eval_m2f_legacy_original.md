### mask2former:vit_wradar_fold_1_20260414_152413.pth — MasterSplits/merged_dataset_splits.json [test], mapping `original`

mAP50 **0.047** · mAP50-95 0.015 · mAP50 tiny/small/medium/large 0.000/0.002/0.024/0.092 · class-agnostic AP50 0.049 · best F1 0.095 @conf 0.55 · latency p50 491.6 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship | 103 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.03 | 0.000 |
| Noise | 132 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |
| Unknown TGT | 91 | 0.007 | 0.003 | 0.000 | 0.010 | 0.035 | 0.055 | 1.45 | 0.121 |
| Coast/Port | 70 | 0.142 | 0.056 | – | – | 0.360 | 0.257 | 0.34 | 0.500 |
| My Ship | 93 | 0.135 | 0.033 | – | 0.000 | 0.538 | 0.151 | 0.13 | 0.237 |
| AIS Lost | 78 | 0.000 | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Ship | Noise | Unknown TGT | Coast/Port | My Ship | AIS Lost | missed |
|---|---|---|---|---|---|---|---|
| Ship | 0 | 0 | 7 | 0 | 0 | 0 | 96 |
| Noise | 0 | 0 | 4 | 0 | 0 | 0 | 128 |
| Unknown TGT | 0 | 0 | 5 | 0 | 0 | 0 | 86 |
| Coast/Port | 0 | 0 | 0 | 18 | 1 | 0 | 51 |
| My Ship | 0 | 0 | 9 | 0 | 14 | 0 | 70 |
| AIS Lost | 0 | 0 | 5 | 0 | 0 | 0 | 73 |
| background | 3 | 0 | 111 | 32 | 11 | 0 | 0 |

Segmentation: mIoU **0.121** · fwIoU 0.361 · foreground IoU 0.390

| class | IoU | Dice | Precision | Recall | Boundary F1 |
|---|---|---|---|---|---|
| Ship | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Noise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Unknown TGT | 0.039 | 0.075 | 0.064 | 0.090 | 0.016 |
| Coast/Port | 0.432 | 0.603 | 0.986 | 0.435 | 0.081 |
| My Ship | 0.257 | 0.408 | 0.801 | 0.274 | 0.035 |
| AIS Lost | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Object level (connected components, centre-hit): Pd 0.124, precision 0.234, FA/frame 2.36
