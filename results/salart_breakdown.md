SalArt clean-originals test, breakdown (served path; from results/salart_clean_ai_items.jsonl).

| model | pair win rate [95% CI] | Imagen 4 flawed vs clean AUROC | specificity: real / clean AI / all unflawed |
|---|---|---|---|
| Glance C5 released (v4 seed 0, package) | 0.546 (65/119) [0.46, 0.63] | 0.461 | 0.975 / 0.218 / 0.769 |
| Glance C5 v4 seed 1 (no calibration) | 0.588 (70/119) [0.50, 0.67] | 0.515 | — |
| Glance C5 v4 seed 2 (no calibration) | 0.513 (61/119) [0.42, 0.60] | 0.478 | — |
| Glance C5 research (distilled) | 0.513 (61/119) [0.42, 0.60] | 0.476 | 0.893 / 0.017 / 0.654 |
| Glance C1 research (teacher) | 0.479 (57/119) [0.39, 0.57] | 0.460 | 0.899 / 0.025 / 0.661 |

Recall on flawed images at the served threshold, Glance C5 released (v4 seed 0, package):

- by generator: flux1_dev 0.26 (n=23); flux2_klein 0.38 (n=169); imagen4 0.75 (n=114); other 0.75 (n=16); qwen 0.36 (n=25); z_turbo 0.20 (n=90)
- by artifact type: NA 0.00 (n=3); anatomy_anomaly 0.32 (n=120); count_anomaly 0.66 (n=97); local_render_structure_anomaly 0.47 (n=57); plausibility_anomaly 0.44 (n=39); topology_anomaly 0.41 (n=121)

Recall on flawed images at the served threshold, Glance C5 research (distilled):

- by generator: flux1_dev 0.57 (n=23); flux2_klein 0.76 (n=169); imagen4 0.97 (n=114); other 0.94 (n=16); qwen 0.80 (n=25); z_turbo 0.64 (n=90)
- by artifact type: NA 0.33 (n=3); anatomy_anomaly 0.74 (n=120); count_anomaly 0.95 (n=97); local_render_structure_anomaly 0.79 (n=57); plausibility_anomaly 0.72 (n=39); topology_anomaly 0.75 (n=121)

Recall on flawed images at the served threshold, Glance C1 research (teacher):

- by generator: flux1_dev 0.70 (n=23); flux2_klein 0.79 (n=169); imagen4 0.93 (n=114); other 0.94 (n=16); qwen 0.88 (n=25); z_turbo 0.74 (n=90)
- by artifact type: NA 1.00 (n=3); anatomy_anomaly 0.80 (n=120); count_anomaly 0.95 (n=97); local_render_structure_anomaly 0.82 (n=57); plausibility_anomaly 0.72 (n=39); topology_anomaly 0.78 (n=121)
