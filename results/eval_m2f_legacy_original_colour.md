### mask2former:vit_wradar_fold_1_20260414_152413.pth@grayscale=0 — MasterSplits/merged_dataset_splits.json [test], mapping `original`

mAP50 **0.186** · mAP50-95 0.081 · mAP50 tiny/small/medium/large 0.000/0.096/0.103/0.266 · class-agnostic AP50 0.202 · best F1 0.280 @conf 0.05 · latency p50 518.3 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship | 103 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.02 | 0.019 |
| Noise | 132 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |
| Unknown TGT | 91 | 0.052 | 0.014 | 0.000 | 0.113 | 0.108 | 0.187 | 1.50 | 0.429 |
| Coast/Port | 70 | 0.488 | 0.302 | – | – | 0.712 | 0.600 | 0.18 | 0.671 |
| My Ship | 93 | 0.578 | 0.171 | – | 0.366 | 0.650 | 0.699 | 0.37 | 0.925 |
| AIS Lost | 78 | 0.000 | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Ship | Noise | Unknown TGT | Coast/Port | My Ship | AIS Lost | missed |
|---|---|---|---|---|---|---|---|
| Ship | 0 | 0 | 5 | 1 | 4 | 0 | 93 |
| Noise | 0 | 0 | 13 | 1 | 1 | 0 | 117 |
| Unknown TGT | 0 | 0 | 17 | 0 | 0 | 0 | 74 |
| Coast/Port | 0 | 0 | 0 | 42 | 0 | 0 | 28 |
| My Ship | 0 | 0 | 0 | 0 | 65 | 0 | 28 |
| AIS Lost | 0 | 0 | 12 | 1 | 0 | 0 | 65 |
| background | 2 | 0 | 111 | 14 | 30 | 0 | 0 |

Segmentation: mIoU **0.237** · fwIoU 0.657 · foreground IoU 0.721

| class | IoU | Dice | Precision | Recall | Boundary F1 |
|---|---|---|---|---|---|
| Ship | 0.006 | 0.012 | 0.722 | 0.006 | 0.001 |
| Noise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Unknown TGT | 0.077 | 0.143 | 0.109 | 0.208 | 0.054 |
| Coast/Port | 0.778 | 0.875 | 0.933 | 0.824 | 0.123 |
| My Ship | 0.562 | 0.720 | 0.770 | 0.676 | 0.273 |
| AIS Lost | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Object level (connected components, centre-hit): Pd 0.331, precision 0.496, FA/frame 1.96
