| engine | FPS | total p50 / p95 ms | infer p50 ms | pre / post p50 ms | power W | mJ / frame |
|---|---|---|---|---|---|---|
| classical | 146.9 | 6.5 / 8.4 | 6.4 | 0.0 / 0.0 | 7.32 | 50 |
| trt:models/engines/arch_yolo26n_960_fp16.engine@crop=roi | 32.6 | 27.5 / 39.8 | 15.9 | 9.8 / 1.5 | 9.11 | 279 |
| trt:models/engines/yolo26s_roi960_fp16.engine@crop=roi | 24.2 | 40.8 / 47.5 | 24.3 | 12.0 / 3.3 | 9.60 | 396 |
| trt:models/engines/yolo26s_roi960_int8mix.engine@crop=roi | 28.6 | 33.8 / 41.1 | 19.1 | 10.9 / 3.2 | 8.92 | 312 |
| trt:models/engines/yolo26s_roi960_int8.engine@crop=roi | 34.7 | 27.8 / 34.5 | 15.7 | 10.2 / 1.5 | 9.11 | 263 |
| trt:models/engines/arch_yolo26s-p2_960_fp16.engine@crop=roi | 17.1 | 56.8 / 64.4 | 43.1 | 12.3 / 1.7 | 9.36 | 546 |
| trt:models/engines/arch_yolo26s-seg_960_fp16.engine@crop=roi | 20.1 | 48.5 / 56.0 | 34.1 | 11.2 / 1.6 | 9.52 | 474 |
| trt:models/engines/arch_yolo26m_960_fp16.engine@crop=roi | 13.9 | 70.8 / 81.0 | 57.0 | 12.9 / 1.6 | 9.94 | 715 |
| ultralytics:models/engines/yolo26s_roi960_blocked.pt@imgsz=960,crop=roi | 11.6 | 85.7 / 95.7 | 69.1 | 14.2 / 2.1 | 10.03 | 864 |
| mask2former:vit_wradar_fold_1_20260414_152413.pth@grayscale=0 | 1.7 | 599.3 / 648.5 | 533.4 | 6.4 / 60.2 | 10.14 | 6135 |

Idle board power: 5.63 W
