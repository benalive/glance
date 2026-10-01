Server: glance-C5 (glance-c5-open) on cpu, calibrated: choice, noul, score. Metric: accuracy on the report split; KonIQ: SRCC of the expected level vs MOS. Offline = the same checkpoint's evaluation file with the same calibration.

| benchmark | n | served | offline | top-answer agreement | max abs prob diff |
|---|---|---|---|---|---|
| pope_adversarial | 3000 | 0.772 | 0.772 | 0.999 | 0.0111 |
| aokvqa_val | 1145 | 0.351 | 0.353 | 0.993 | 0.4453 |
| koniq_test | 2015 | 0.519 | 0.519 | 0.996 | 0.0053 |
| vqav2_val_yesno | 6813 | 0.602 | 0.602 | 0.998 | 0.0154 |
| had_val_any | 4173 | 0.691 | 0.691 | 0.999 | 0.0054 |
| salart_q1 | 755 | 0.715 | 0.715 | 1.000 | 0.0057 |

VQAv2 validation yes/no (images not in training), by phrasing; groups with n >= 40:

| group | n | accuracy | share gold yes | mean P(yes), gold yes | mean P(yes), gold no |
|---|---|---|---|---|---|
| about people (man, woman, child, ...) | 1640 | 0.595 | 0.49 | 0.54 | 0.45 |
| type: is the | 1307 | 0.575 | 0.51 | 0.52 | 0.46 |
| 'Is there a X', X not a COCO name | 972 | 0.619 | 0.53 | 0.56 | 0.42 |
| type: is this | 607 | 0.634 | 0.48 | 0.55 | 0.46 |
| type: is this a | 566 | 0.641 | 0.51 | 0.54 | 0.41 |
| type: are the | 436 | 0.610 | 0.48 | 0.57 | 0.46 |
| type: is there a | 396 | 0.606 | 0.54 | 0.56 | 0.42 |
| type: none of the above | 334 | 0.575 | 0.53 | 0.55 | 0.49 |
| type: does the | 278 | 0.568 | 0.53 | 0.53 | 0.49 |
| type: is there | 271 | 0.605 | 0.54 | 0.56 | 0.43 |
| type: is it | 269 | 0.647 | 0.55 | 0.56 | 0.38 |
| type: is | 261 | 0.571 | 0.50 | 0.52 | 0.45 |
| type: are there | 233 | 0.665 | 0.52 | 0.59 | 0.43 |
| type: are these | 224 | 0.674 | 0.49 | 0.56 | 0.42 |
| type: are | 209 | 0.541 | 0.47 | 0.52 | 0.49 |
| type: does this | 185 | 0.584 | 0.55 | 0.56 | 0.51 |
| type: is the man | 169 | 0.598 | 0.48 | 0.54 | 0.46 |
| type: do | 128 | 0.664 | 0.66 | 0.61 | 0.54 |
| type: are there any | 126 | 0.627 | 0.46 | 0.54 | 0.39 |
| type: are they | 95 | 0.642 | 0.41 | 0.58 | 0.42 |
| type: is he | 91 | 0.582 | 0.52 | 0.50 | 0.39 |
| type: has | 84 | 0.571 | 0.56 | 0.49 | 0.44 |
| type: is this an | 79 | 0.633 | 0.57 | 0.57 | 0.45 |
| type: is the woman | 74 | 0.608 | 0.64 | 0.50 | 0.44 |
| type: can you | 70 | 0.500 | 0.43 | 0.45 | 0.39 |
| type: do you | 62 | 0.726 | 0.47 | 0.64 | 0.38 |
| type: was | 57 | 0.667 | 0.40 | 0.54 | 0.38 |
| type: is that a | 56 | 0.696 | 0.45 | 0.53 | 0.32 |
| 'Is there a X', X a COCO name | 53 | 0.679 | 0.42 | 0.69 | 0.39 |
| type: is the person | 52 | 0.500 | 0.44 | 0.52 | 0.46 |
| type: is this person | 49 | 0.571 | 0.45 | 0.54 | 0.46 |
| type: could | 44 | 0.545 | 0.59 | 0.55 | 0.51 |

Latency per request (ms): new_image p50 23.1 / p95 29.2 (n=11405)
