Zoomed tiles, inference only, released model (data/release/glance-c5-open-v4); image score = max P(yes) over the whole image and its tiles. SalArt pairs: 119 inpainted flaws vs their clean originals. Threshold for tiles chosen so SalArt's real photos are flagged as often as with the whole image at 0.5.

| method | encodings_per_image | salart_pairs_auroc | salart_flawed_vs_real_auroc | salart_flawed_vs_clean_ai_auroc | artibench_auroc | threshold_matched_on_salart_real | salart_recall_at_threshold | real_photo_questions_flagged_at_threshold |
|---|---|---|---|---|---|---|---|---|
| whole | 1 | 0.508 | 0.913 | 0.284 | 0.667 | 0.500 | 0.449 | 0.002 |
| tiles2 | 5 | 0.508 | 0.904 | 0.285 | 0.674 | 0.568 | 0.551 | 0.006 |
| tiles3 | 10 | 0.507 | 0.906 | 0.281 | 0.639 | 0.630 | 0.522 | 0.011 |
