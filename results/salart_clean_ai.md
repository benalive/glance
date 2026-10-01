SalArt-VQA Q1 on 437 flawed generated images, 318 real photos and 119 clean generated images, the originals of 119 flawed images made by inpainting a flaw (HAD-derived rows dropped). Recall and flagged shares at the served threshold (P(yes) >= 0.5).

| model | auroc_flawed_vs_real | auroc_flawed_vs_clean_ai | auroc_injected_vs_original | pairs_injected_higher | auroc_other_flawed_vs_real | recall_flawed | recall_injected | flagged_real | flagged_clean_ai |
|---|---|---|---|---|---|---|---|---|---|
| Glance C5 released (v4 seed 0, package) | 0.913 | 0.284 | 0.508 | 0.546 | 0.886 | 0.449 | 0.782 | 0.025 | 0.782 |
| Glance C5 v4 seed 1 (no calibration) | 0.570 | 0.456 | 0.509 | 0.588 | 0.550 | — | — | — | — |
| Glance C5 v4 seed 2 (no calibration) | 0.887 | 0.271 | 0.506 | 0.513 | 0.850 | — | — | — | — |
| Glance C5 research (distilled) | 0.905 | 0.335 | 0.501 | 0.513 | 0.877 | 0.792 | 0.975 | 0.107 | 0.983 |
| Glance C1 research (teacher) | 0.929 | 0.332 | 0.495 | 0.479 | 0.911 | 0.824 | 0.958 | 0.101 | 0.975 |
