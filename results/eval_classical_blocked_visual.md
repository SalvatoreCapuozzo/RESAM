### classical — MasterSplits/splits_blocked.json [test], mapping `visual`

mAP50 **0.118** · mAP50-95 0.049 · mAP50 tiny/small/medium/large 0.016/0.003/0.002/0.173 · class-agnostic AP50 0.015 · best F1 0.076 @conf 0.40 · latency p50 7.5 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Target | 224 | 0.007 | 0.002 | 0.032 | 0.011 | 0.014 | 0.045 | 9.12 | 0.638 |
| Clutter | 339 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |
| Land | 59 | 0.463 | 0.195 | – | 0.000 | 0.706 | 0.610 | 0.19 | 0.610 |
| Own Ship | 77 | 0.000 | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Target | Clutter | Land | Own Ship | missed |
|---|---|---|---|---|---|
| Target | 10 | 0 | 0 | 0 | 214 |
| Clutter | 2 | 0 | 0 | 0 | 337 |
| Land | 3 | 0 | 36 | 0 | 20 |
| Own Ship | 1 | 0 | 8 | 0 | 68 |
| background | 724 | 0 | 7 | 0 | 0 |
