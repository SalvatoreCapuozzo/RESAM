### classical +track2 — MasterSplits/splits_blocked.json [test], mapping `visual`

mAP50 **0.117** · mAP50-95 0.049 · mAP50 tiny/small/medium/large 0.014/0.002/0.005/0.173 · class-agnostic AP50 0.015 · best F1 0.079 @conf 0.40 · latency p50 10.0 ms

| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |
|---|---|---|---|---|---|---|---|---|---|
| Target | 224 | 0.007 | 0.002 | 0.028 | 0.010 | 0.015 | 0.031 | 5.92 | 0.545 |
| Clutter | 339 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |
| Land | 59 | 0.463 | 0.195 | – | 0.000 | 0.706 | 0.610 | 0.19 | 0.610 |
| Own Ship | 77 | 0.000 | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |

Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):

| GT \ pred | Target | Clutter | Land | Own Ship | missed |
|---|---|---|---|---|---|
| Target | 7 | 0 | 0 | 0 | 217 |
| Clutter | 2 | 0 | 0 | 0 | 337 |
| Land | 2 | 0 | 36 | 0 | 21 |
| Own Ship | 0 | 0 | 8 | 0 | 69 |
| background | 470 | 0 | 7 | 0 | 0 |
