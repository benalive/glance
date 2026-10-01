Fresh test boards: 2000 boards from 101 planner games (seeds from 20011, at most 20 boards per game, 596 boards that occur in the training sets dropped). Metrics on the report split (1601 boards, 81 games); temperature fitted on the other 20% of games. CIs: game-grouped bootstrap. Majority-move baseline (report split): 0.299.

| model | top-1 [95% CI] | tie-aware | legal top choice | trap boards (n) | NLL / Brier / smECE as shipped | T | NLL / Brier / smECE with T |
|---|---|---|---|---|---|---|---|
| Greedy rule (fact lines) | 0.981 [0.973, 0.989] | 0.981 | 1.000 | 0.000 (24) | 0.259 / 0.037 / 0.019 | 2.70 | 0.114 / 0.037 / 0.000 |
| Greedy rule (planner key) | 0.985 [0.978, 0.991] | 0.985 | 1.000 | 0.000 (24) | 0.207 / 0.030 / 0.015 | 2.70 | 0.095 / 0.030 / 0.000 |
| Laya-snake (text) | 0.981 [0.973, 0.989] | 0.981 | 1.000 | 0.000 (24) | 0.067 / 0.032 / 0.010 | 0.97 | 0.067 / 0.033 / 0.009 |
| Glance C5 text | 0.981 [0.973, 0.989] | 0.981 | 1.000 | 0.000 (24) | 0.096 / 0.036 / 0.011 | 0.92 | 0.098 / 0.036 / 0.011 |
| Glance C5 image+facts | 0.981 [0.973, 0.989] | 0.981 | 1.000 | 0.000 (24) | 0.112 / 0.037 / 0.012 | 0.94 | 0.113 / 0.037 / 0.011 |
| Glance C5 image+facts (100k boards) | 0.982 [0.974, 0.989] | 0.982 | 1.000 | 0.125 (24) | 0.056 / 0.028 / 0.008 | 0.74 | 0.061 / 0.029 / 0.010 |
| Glance C6s image | 0.600 [0.578, 0.623] | 0.669 | 0.923 | 0.333 (24) | 0.996 / 0.532 / 0.037 | 1.03 | 0.996 / 0.533 / 0.039 |
| Glance C6s image+aux | 0.689 [0.665, 0.711] | 0.771 | 0.934 | 0.375 (24) | 0.777 / 0.421 / 0.028 | 1.08 | 0.777 / 0.422 / 0.032 |
| Glance C3 image+aux | 0.670 [0.648, 0.694] | 0.746 | 0.926 | 0.167 (24) | 0.861 / 0.466 / 0.024 | 1.05 | 0.858 / 0.466 / 0.024 |

Top-choice agreement on all boards:

| | Greedy rule (fact lines) | Greedy rule (planner key) | Laya-snake (text) | Glance C5 text | Glance C5 image+facts | Glance C5 image+facts (100k boards) | Glance C6s image | Glance C6s image+aux | Glance C3 image+aux |
|---|---|---|---|---|---|---|---|---|---|
| Greedy rule (fact lines) | 1.0000 | 0.9970 | 1.0000 | 1.0000 | 1.0000 | 0.9975 | 0.6005 | 0.6825 | 0.6700 |
| Greedy rule (planner key) | 0.9970 | 1.0000 | 0.9970 | 0.9970 | 0.9970 | 0.9945 | 0.6000 | 0.6830 | 0.6700 |
| Laya-snake (text) | 1.0000 | 0.9970 | 1.0000 | 1.0000 | 1.0000 | 0.9975 | 0.6005 | 0.6825 | 0.6700 |
| Glance C5 text | 1.0000 | 0.9970 | 1.0000 | 1.0000 | 1.0000 | 0.9975 | 0.6005 | 0.6825 | 0.6700 |
| Glance C5 image+facts | 1.0000 | 0.9970 | 1.0000 | 1.0000 | 1.0000 | 0.9975 | 0.6005 | 0.6825 | 0.6700 |
| Glance C5 image+facts (100k boards) | 0.9975 | 0.9945 | 0.9975 | 0.9975 | 0.9975 | 1.0000 | 0.5995 | 0.6805 | 0.6675 |
| Glance C6s image | 0.6005 | 0.6000 | 0.6005 | 0.6005 | 0.6005 | 0.5995 | 1.0000 | 0.6840 | 0.7025 |
| Glance C6s image+aux | 0.6825 | 0.6830 | 0.6825 | 0.6825 | 0.6825 | 0.6805 | 0.6840 | 1.0000 | 0.7175 |
| Glance C3 image+aux | 0.6700 | 0.6700 | 0.6700 | 0.6700 | 0.6700 | 0.6675 | 0.7025 | 0.7175 | 1.0000 |

Per-board NLL difference vs Laya-snake, as shipped (report split, paired game-grouped bootstrap):

- Glance C5 text − Laya: +0.0284 nats, 95% CI [+0.0148, +0.0442]
- Glance C5 image+facts − Laya: +0.0445 nats, 95% CI [+0.0250, +0.0689]
