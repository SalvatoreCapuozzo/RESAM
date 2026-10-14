### mask2former:vit_wradar_fold_1_20260414_152413.pth — MasterSplits/merged_dataset_splits.json [test], mapping `mapping_1`

mAP50 **0.073** · mAP50-95 0.023 · mAP50 tiny/small/medium/large 0.000/0.005/0.039/0.114 · class-agnostic AP50 0.049 · best F1 0.126 @conf 0.55 · latency p50 490.2 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Ship (merged) | 272 | 0.015 | 0.004 | 0.000 | 0.015 | 0.118 | 0.062 | 1.35 | 0.143 |
| Noise | 132 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |
| Coast/Port | 70 | 0.142 | 0.056 | – | – | 0.360 | 0.257 | 0.34 | 0.500 |
| My Ship | 93 | 0.135 | 0.033 | – | 0.000 | 0.538 | 0.151 | 0.13 | 0.237 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Ship (merged) | Noise | Coast/Port | My Ship | missed |
|---|---|---|---|---|---|
| Ship (merged) | 17 | 0 | 0 | 0 | 255 |
| Noise | 4 | 0 | 0 | 0 | 128 |
| Coast/Port | 0 | 0 | 18 | 1 | 51 |
| My Ship | 9 | 0 | 0 | 14 | 70 |
| background | 114 | 0 | 32 | 11 | 0 |

Segmentation: mIoU **0.190** · fwIoU 0.365 · foreground IoU 0.390

| class | IoU | Dice | Precision | Recall | Boundary F1 |
|---|---|---|---|---|---|
| Ship (merged) | 0.070 | 0.130 | 0.254 | 0.088 | 0.041 |
| Noise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Coast/Port | 0.432 | 0.603 | 0.986 | 0.435 | 0.081 |
| My Ship | 0.257 | 0.408 | 0.801 | 0.274 | 0.035 |

Object level (connected components, centre-hit): Pd 0.184, precision 0.364, FA/frame 1.86
