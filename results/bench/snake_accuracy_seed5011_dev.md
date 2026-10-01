Same 2000 held-out boards for every model (planner games, seed 5011 = LayaStudio's test split); metrics on the 80% report split (games grouped), temperature fitted on the other 20%. Majority-move baseline: 0.283.

| model | top-1 [95% CI] | tie-aware | legal top choice | trap boards (n) | NLL / Brier / smECE as shipped | T | NLL / Brier / smECE with T |
|---|---|---|---|---|---|---|---|
| Laya-snake (text) | 0.977 [0.971, 0.985] | 0.977 | 1.000 | 0.000 (30) | 0.082 / 0.040 / 0.012 | 1.10 | 0.080 / 0.039 / 0.011 |
| Glance C5 text | 0.977 [0.971, 0.985] | 0.977 | 1.000 | 0.000 (30) | 0.099 / 0.043 / 0.012 | 1.13 | 0.096 / 0.043 / 0.010 |
| Glance C5 image+facts | 0.977 [0.971, 0.985] | 0.977 | 1.000 | 0.000 (30) | 0.099 / 0.042 / 0.016 | 0.83 | 0.099 / 0.042 / 0.011 |
| Glance C6s image+aux | 0.672 [0.646, 0.714] | 0.752 | 0.919 | 0.300 (30) | 0.805 / 0.442 / 0.025 | 0.95 | 0.808 / 0.443 / 0.028 |
| Glance C3 image+aux | 0.665 [0.629, 0.706] | 0.737 | 0.908 | 0.267 (30) | 0.837 / 0.465 / 0.026 | 1.09 | 0.835 / 0.465 / 0.024 |
