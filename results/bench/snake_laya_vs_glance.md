CPU only (Apple M4 Pro, 4 threads, quiet machine). Same engine, planner, seeds and unassisted top-1 protocols for every row. A = 30 s continuous play; B = 10 games x <=500 ticks.

| model | input | params | training boards | p50 / p95 ms | decisions/s | A: food / deaths / best length | B: moves mean / median | B: score mean / best | B: legal | B: planner agreement |
|---|---|---|---|---|---|---|---|---|---|---|
| Laya-snake (mmBERT-base, LoRA) | text: ASCII board + facts | 322M | 2,339 | 40.3 / 43.6 | 24.6 | 80 / 3 / 31 | 169.2 / 161.5 | 19.8 / 33 | 0.993 | 0.979 |
| Glance C5, text | text: ASCII board + facts (same as Laya) | 40M (text path; image features constant) | 2,339 | 13.7 / 15.0 | 72.4 | 253 / 11 / 40 | 187.8 / 184.5 | 21.4 / 36 | 0.993 | 0.980 |
| Glance C5, image + facts | board image + 2 fact lines | 134M | 2,339 | 31.4 / 36.8 | 30.9 | 101 / 4 / 37 | 187.8 / 184.5 | 21.4 / 36 | 0.993 | 0.980 |
| Glance C5, image + facts | board image + 2 fact lines | 134M | 100k generated | 27.6 / 29.2 | 36.0 | 132 / 6 / 34 | 180.7 / 169.5 | 20.5 / 34 | 0.993 | 0.981 |
| Glance C6s + aux, image only | board image only | 113M | 100k generated | 16.6 / 19.3 | 58.8 | 76 / 134 / 8 | 12.7 / 9.0 | 0.5 / 2 | 0.892 | 0.478 |
| Glance C3 + aux, image only | board image only | 183M | 100k generated | 32.3 / 42.7 | 29.8 | 1 / 1 / 5 | 118.5 / 9.0 | 0.7 / 4 | 0.918 | 0.425 |

Planner (the teacher, a ceiling): B moves 418.3 mean, score 34.4, 0.1 ms per decision.
