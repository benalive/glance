Everyday VQAv2 validation yes/no questions on images outside the training mix; accuracy on the report split after Platt scaling fitted on the calib split (chance is about 0.5: the gold answers are balanced).

| question group | n | glance C1_art2_ep2 | glance C1_ep2_lrA_s0 | glance C5_art2_ep2 | glance C5_art_ep2 | glance C5_distill_ep2 | glance C5_ep2_lrA_s0 | glance C5_gold48k_ep2 | glance C7_ep2_lrA_s0 | glance C8_ep2_lrA_s0 | qwen3vl_2b | siglip2_b16 | smolvlm_256m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 5456 | 0.680 | 0.694 | 0.537 | 0.547 | 0.568 | 0.549 | 0.561 | 0.532 | 0.569 | 0.812 | 0.488 | 0.648 |
| about people (man, woman, child, ...) | 1298 | 0.682 | 0.710 | 0.551 | 0.552 | 0.593 | 0.552 | 0.575 | 0.539 | 0.572 | 0.803 | 0.508 | 0.648 |
| 'is the man ...' | 132 | 0.629 | 0.750 | 0.553 | 0.561 | 0.568 | 0.538 | 0.591 | 0.462 | 0.561 | 0.848 | 0.485 | 0.758 |
| 'is the woman ...' | 61 | 0.623 | 0.705 | 0.443 | 0.492 | 0.590 | 0.443 | 0.508 | 0.508 | 0.590 | 0.787 | 0.443 | 0.623 |
| 'Is there a X', X a COCO name | 41 | 0.780 | 0.878 | 0.585 | 0.634 | 0.610 | 0.585 | 0.610 | 0.610 | 0.732 | 0.951 | 0.610 | 0.756 |
| 'Is there a X', other X | 782 | 0.711 | 0.729 | 0.523 | 0.555 | 0.554 | 0.535 | 0.536 | 0.540 | 0.560 | 0.849 | 0.472 | 0.673 |
| 'is this / is this a ...' | 1005 | 0.713 | 0.744 | 0.549 | 0.562 | 0.598 | 0.560 | 0.589 | 0.549 | 0.588 | 0.853 | 0.497 | 0.685 |
| 'are there ...' | 286 | 0.738 | 0.755 | 0.524 | 0.556 | 0.559 | 0.531 | 0.549 | 0.549 | 0.556 | 0.864 | 0.528 | 0.703 |
| 'does the / does this ...' | 379 | 0.620 | 0.591 | 0.483 | 0.515 | 0.517 | 0.544 | 0.530 | 0.499 | 0.533 | 0.739 | 0.438 | 0.599 |
| 'is the ...' (other) | 1067 | 0.661 | 0.669 | 0.529 | 0.529 | 0.548 | 0.541 | 0.532 | 0.515 | 0.547 | 0.800 | 0.499 | 0.664 |
| AUROC (all) | | 0.759 | 0.785 | 0.590 | 0.585 | 0.614 | 0.592 | 0.608 | 0.566 | 0.614 | 0.902 | 0.561 | 0.712 |
