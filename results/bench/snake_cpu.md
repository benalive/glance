CPU only, Apple M4 Pro. A = 30 s continuous play (new game on death); B = LayaStudio's 10 games x <=500 ticks, unassisted top-1 move. 'glance' rows are untrained, timed on the planner's boards (latency only); 'glance-snake' rows are Snake-tuned and play. Runs timed while another job used the machine are excluded.

| model | runtime | threads | A: decisions/s | A: p50 / p95 ms | A: food in 30 s | A: best length | A: deaths | B: moves mean | B: score mean | B: legal | B: teacher agree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Planner (ceiling) | teacher | 10 | 17809.5 | 0.1 / 0.1 | 45649 | 55 | 1045 | 418.3 | 34.4 | 0.998 | 0.998 |
| Laya-snake 322M (Snake-tuned) | laya-torch | 10 | 19.6 | 50.2 / 55.7 | 69 | 28 | 3 | 169.2 | 19.8 | 0.993 | 0.979 |
| Laya-snake 322M (Snake-tuned) | laya-torch | 4 | 23.5 | 42.0 / 45.2 | 80 | 31 | 3 | — | — | — | — |
| Laya multilingual 322M (untuned) | laya-torch | 10 | 19.8 | 49.9 / 54.2 | 0 | 4 | 595 | 1.0 | 0.0 | 0.000 | 0.000 |
| Laya English 421M (untuned) | laya-torch | 10 | 9.7 | 102.4 / 106.8 | 7 | 6 | 78 | 3.2 | 0.0 | 0.647 | 0.263 |
| Laya-snake 322M (Snake-tuned) | laya-mlx | 10 | 17.9 | 55.4 / 60.5 | 60 | 28 | 3 | 169.2 | 19.8 | 0.993 | 0.979 |
| Glance C6 6x512 (image in, untrained) | glance | 10 | 56.9 | 12.1 / 60.4 | — | — | — | — | — | — | — |
| Glance C6 6x512 (image in, untrained) | glance | 4 | 93.4 | 10.6 / 11.0 | — | — | — | — | — | — | — |
| Glance C3 12x768 (image in, untrained) | glance | 10 | 29.4 | 33.6 / 36.2 | — | — | — | — | — | — | — |
| Glance C3 12x768 (image in, untrained) | glance | 4 | 33.9 | 29.1 / 31.5 | — | — | — | — | — | — | — |
| Glance C5 fusion (image in, untrained) | glance | 10 | 34.3 | 28.6 / 32.0 | — | — | — | — | — | — | — |
| Glance C5 fusion (image in, untrained) | glance | 4 | 40.9 | 24.4 / 25.0 | — | — | — | — | — | — | — |
| C5 img+facts (Snake-tuned) [C5_img_facts] (slot readout, superseded) | glance-snake | 4 | 36.0 | 27.6 / 29.2 | 132 | 34 | 6 | 180.7 | 20.5 | 0.993 | 0.981 |
| C6 img+facts (Snake-tuned) [C6_img_facts] (slot readout, superseded) | glance-snake | 4 | 82.3 | 12.1 / 12.7 | 322 | 14 | 233 | 10.3 | 1.5 | 0.894 | 0.808 |
| C6 img (Snake-tuned) [C6_img] (slot readout, superseded) | glance-snake | 4 | 101.9 | 9.7 / 10.4 | 0 | 4 | 3058 | 1.0 | 0.0 | 0.000 | 0.000 |
| C5 img+facts (Snake-tuned) [C5_img_facts_matched] (slot readout, superseded) | glance-snake | 4 | 32.5 | 30.5 / 32.6 | 109 | 37 | 5 | 187.8 | 21.4 | 0.993 | 0.980 |
| C6s img (Snake-tuned) [C6s_img] | glance-snake | 4 | 62.5 | 15.9 / 16.8 | 39 | 6 | 292 | 6.0 | 0.1 | 0.833 | 0.200 |
| C6s img+aux (Snake-tuned) [C6s_img_aux] | glance-snake | 4 | 58.8 | 16.6 / 19.3 | 76 | 8 | 134 | 12.7 | 0.5 | 0.892 | 0.478 |
| C3 img+aux (Snake-tuned) [C3_img_aux] | glance-snake | 4 | 29.8 | 32.3 / 42.7 | 1 | 5 | 1 | 118.5 | 0.7 | 0.918 | 0.425 |
| Laya-snake 322M (Snake-tuned) | laya-torch | 4 | 24.6 | 40.3 / 43.6 | 80 | 31 | 3 | 169.2 | 19.8 | 0.993 | 0.979 |
| C5 text (Snake-tuned) [C5_text_matched] | glance-snake | 4 | 72.4 | 13.7 / 15.0 | 253 | 40 | 11 | 187.8 | 21.4 | 0.993 | 0.980 |
| C5 img+facts (Snake-tuned) [C5_img_facts_matched] (slot readout, superseded) | glance-snake | 4 | 33.5 | 29.8 / 31.0 | 112 | 37 | 5 | 187.8 | 21.4 | 0.993 | 0.980 |
| C5 img+facts (Snake-tuned) [C5_img_facts_matched] | glance-snake | 4 | 30.9 | 31.4 / 36.8 | 101 | 37 | 4 | 187.8 | 21.4 | 0.993 | 0.980 |
| Laya 421M | laya-mlx on M3 Max GPU (article) | — | 86.5 | 9 / — | 46 | 52 | — | — | — | — | — |
| jev-1.13.0 | cloud API (article, not measured) | — | 3.2 | 317 / — | 1 | 7 | — | — | — | — | — |
