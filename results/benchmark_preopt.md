| engine | FPS | total p50 / p95 ms | infer p50 ms | pre / post p50 ms | power W | mJ / frame |
|---|---|---|---|---|---|---|
| trt:models/engines/yolo26s_roi960_fp16.engine@crop=roi | 24.6 | 37.7 / 48.2 | 24.4 | 9.0 / 3.6 | 10.16 | 413 |
| trt:models/engines/yolo26s_roi960_int8mix.engine@crop=roi | 31.3 | 31.3 / 36.8 | 19.2 | 8.2 / 3.2 | 8.81 | 282 |

Idle board power: 5.51 W
