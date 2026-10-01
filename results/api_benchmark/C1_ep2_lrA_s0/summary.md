Server: glance-C1 (C1_ep2_lrA_s0.pt) on cpu, calibrated: choice, noul, score. Metric: accuracy on the report split; KonIQ: SRCC of the expected level vs MOS. Offline = the same checkpoint's evaluation file with the same calibration.

| benchmark | n | served | offline | top-answer agreement | max abs prob diff |
|---|---|---|---|---|---|
| pope_adversarial | 3000 | 0.813 | 0.814 | 0.997 | 0.0387 |

Latency per request (ms): new_image p50 228.1 / p95 245.5 (n=500)
