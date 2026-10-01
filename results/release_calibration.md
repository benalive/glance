Release package glance-c5-open-v4 (seed 0) with its own calibration: accuracy, mean confidence, smooth ECE, Brier.

| benchmark | n | accuracy | mean confidence | smECE | Brier | recall (yes) | specificity |
|---|---|---|---|---|---|---|---|
| vqav2_val_yesno | 5336 | 0.602 | 0.638 | 0.036 | 0.342 | 0.579 | 0.626 |
| pope_adversarial | 2334 | 0.769 | 0.848 | 0.079 | 0.317 | 0.831 | 0.708 |
| pope_random | 2334 | 0.887 | 0.903 | 0.019 | 0.161 | 0.834 | 0.941 |
| aokvqa_val | 927 | 0.388 | 0.538 | 0.150 | 0.750 | — | — |
| mmstar | 1191 | 0.280 | 0.394 | 0.113 | 0.764 | — | — |
| sugarcrepe_replace_rel | 1105 | 0.619 | 0.573 | 0.046 | 0.468 | — | — |
| sugarcrepe_swap_att | 541 | 0.547 | 0.548 | 0.030 | 0.485 | — | — |
| salart_q1 | 755 | 0.669 | 0.760 | 0.091 | 0.421 | 0.449 | 0.972 |
| artibench | 1000 | 0.571 | 0.868 | 0.296 | 0.686 | 0.182 | 0.960 |
| had_val_hands | 3354 | 0.492 | 0.705 | 0.213 | 0.591 | 0.294 | 0.847 |
| had_val_any | 3349 | 0.665 | 0.719 | 0.054 | 0.442 | 0.700 | 0.580 |
| richhf_test_artifacts | 318 | 0.465 | 0.811 | 0.345 | 0.851 | 0.335 | 0.984 |

Real photographs (500 COCO, 500 KonIQ; three error questions each): 23 of 3000 questions flagged (0.8%), 20 of 1000 photos flagged by at least one question (2.0%).

AUROC by seed of the released recipe (v4, seeds 0 / 1 / 2):

| benchmark | s0 | s1 | s2 |
|---|---|---|---|
| vqav2_val_yesno | 0.653 | 0.644 | 0.648 |
| pope_adversarial | 0.869 | 0.868 | 0.863 |
| pope_random | 0.964 | 0.963 | 0.965 |
| sugarcrepe_replace_rel | 0.653 | 0.599 | 0.660 |
| sugarcrepe_swap_att | 0.593 | 0.601 | 0.618 |
| salart_q1 | 0.911 | 0.551 | 0.886 |
| artibench | 0.667 | 0.677 | 0.651 |
| had_val_hands | 0.695 | 0.690 | 0.700 |
| had_val_any | 0.702 | 0.692 | 0.689 |
| richhf_test_artifacts | 0.828 | 0.812 | 0.800 |
