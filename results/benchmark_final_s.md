| engine | FPS | total p50 / p95 ms | infer p50 ms | pre / post p50 ms | power W | mJ / frame |
|---|---|---|---|---|---|---|
| trt:models/engines/yolo26s_roi960_fp16.engine@crop=roi | 22.9 | 41.6 / 48.0 | 24.3 | 12.1 / 3.5 | 10.05 | 439 |
| trt:models/engines/yolo26s_roi960_int8mix.engine@crop=roi | 28.8 | 34.0 / 39.6 | 19.1 | 10.8 / 3.0 | 8.98 | 312 |

Idle board power: 5.67 W
