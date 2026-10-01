Server: glance-C5 (glance-c5-open-v4) on cpu, calibrated: choice, noul, score. Metric: accuracy on the report split; KonIQ: SRCC of the expected level vs MOS. Offline = the same checkpoint's evaluation file with the same calibration.

| benchmark | n | served | offline | top-answer agreement | max abs prob diff |
|---|---|---|---|---|---|
| pope_adversarial | 3000 | 0.770 | 0.769 | 0.999 | 0.0124 |
| aokvqa_val | 1145 | 0.391 | 0.388 | 0.995 | 0.3959 |
| koniq_test | 2015 | 0.505 | 0.505 | 0.998 | 0.0054 |
| vqav2_val_yesno | 6813 | 0.601 | 0.601 | 0.994 | 0.0109 |
| had_val_any | 4173 | 0.666 | 0.665 | 0.999 | 0.0063 |
| salart_q1 | 755 | 0.674 | 0.672 | 0.999 | 0.0185 |

VQAv2 validation yes/no (images not in training), by phrasing; groups with n >= 40:

| group | n | accuracy | share gold yes | mean P(yes), gold yes | mean P(yes), gold no |
|---|---|---|---|---|---|
| about people (man, woman, child, ...) | 1640 | 0.607 | 0.49 | 0.53 | 0.44 |
| type: is the | 1307 | 0.592 | 0.51 | 0.52 | 0.46 |
| 'Is there a X', X not a COCO name | 972 | 0.616 | 0.53 | 0.56 | 0.42 |
| type: is this | 607 | 0.629 | 0.48 | 0.56 | 0.46 |
| type: is this a | 566 | 0.654 | 0.51 | 0.53 | 0.40 |
| type: are the | 436 | 0.626 | 0.48 | 0.55 | 0.45 |
| type: is there a | 396 | 0.621 | 0.54 | 0.56 | 0.40 |
| type: none of the above | 334 | 0.569 | 0.53 | 0.52 | 0.45 |
| type: does the | 278 | 0.558 | 0.53 | 0.52 | 0.48 |
| type: is there | 271 | 0.554 | 0.54 | 0.56 | 0.45 |
| type: is it | 269 | 0.654 | 0.55 | 0.56 | 0.37 |
| type: is | 261 | 0.536 | 0.50 | 0.50 | 0.45 |
| type: are there | 233 | 0.700 | 0.52 | 0.61 | 0.42 |
| type: are these | 224 | 0.652 | 0.49 | 0.53 | 0.41 |
| type: are | 209 | 0.545 | 0.47 | 0.51 | 0.46 |
| type: does this | 185 | 0.578 | 0.55 | 0.56 | 0.49 |
| type: is the man | 169 | 0.574 | 0.48 | 0.54 | 0.46 |
| type: do | 128 | 0.641 | 0.66 | 0.58 | 0.52 |
| type: are there any | 126 | 0.603 | 0.46 | 0.56 | 0.41 |
| type: are they | 95 | 0.695 | 0.41 | 0.54 | 0.39 |
| type: is he | 91 | 0.615 | 0.52 | 0.51 | 0.39 |
| type: has | 84 | 0.548 | 0.56 | 0.49 | 0.41 |
| type: is this an | 79 | 0.570 | 0.57 | 0.54 | 0.46 |
| type: is the woman | 74 | 0.595 | 0.64 | 0.51 | 0.43 |
| type: can you | 70 | 0.514 | 0.43 | 0.45 | 0.40 |
| type: do you | 62 | 0.742 | 0.47 | 0.65 | 0.40 |
| type: was | 57 | 0.737 | 0.40 | 0.53 | 0.38 |
| type: is that a | 56 | 0.607 | 0.45 | 0.51 | 0.36 |
| 'Is there a X', X a COCO name | 53 | 0.679 | 0.42 | 0.72 | 0.39 |
| type: is the person | 52 | 0.519 | 0.44 | 0.49 | 0.44 |
| type: is this person | 49 | 0.571 | 0.45 | 0.52 | 0.45 |
| type: could | 44 | 0.591 | 0.59 | 0.56 | 0.49 |

Latency per request (ms): new_image p50 16.6 / p95 22.3 (n=11405)
