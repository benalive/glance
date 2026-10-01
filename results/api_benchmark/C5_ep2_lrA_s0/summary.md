Server: glance-C5 (C5_ep2_lrA_s0.pt) on cpu, calibrated: choice, noul, score. Metric: accuracy on the report split; KonIQ: SRCC of the expected level vs MOS. Offline = the same checkpoint's evaluation file with the same calibration.

| benchmark | n | served | offline | top-answer agreement | max abs prob diff |
|---|---|---|---|---|---|
| pope_adversarial | 3000 | 0.790 | 0.791 | 0.999 | 0.0045 |
| aokvqa_val | 1145 | 0.434 | 0.433 | 0.992 | 0.2628 |
| mmstar | 1491 | 0.246 | 0.248 | 0.987 | 0.1021 |
| sugarcrepe_replace_rel | 1406 | 0.643 | 0.643 | 0.991 | 0.0060 |
| sugarcrepe_swap_att | 666 | 0.584 | 0.586 | 0.989 | 0.0052 |
| koniq_test | 2015 | 0.817 | 0.817 | 0.997 | 0.0052 |
| pope_random | 3000 | 0.880 | — | — | — |
| pope_popular | 3000 | 0.850 | — | — | — |
| vqav2_val_yesno | 6813 | 0.495 | — | — | — |

VQAv2 validation yes/no (images not in training), by phrasing; groups with n >= 40:

| group | n | accuracy | share gold yes | mean P(yes), gold yes | mean P(yes), gold no |
|---|---|---|---|---|---|
| about people (man, woman, child, ...) | 1640 | 0.524 | 0.49 | 0.38 | 0.37 |
| type: is the | 1307 | 0.495 | 0.51 | 0.38 | 0.37 |
| 'Is there a X', X not a COCO name | 972 | 0.499 | 0.53 | 0.39 | 0.36 |
| type: is this | 607 | 0.522 | 0.48 | 0.38 | 0.37 |
| type: is this a | 566 | 0.507 | 0.51 | 0.39 | 0.36 |
| type: are the | 436 | 0.525 | 0.48 | 0.39 | 0.37 |
| type: is there a | 396 | 0.528 | 0.54 | 0.40 | 0.35 |
| type: none of the above | 334 | 0.476 | 0.53 | 0.39 | 0.37 |
| type: does the | 278 | 0.478 | 0.53 | 0.39 | 0.38 |
| type: is there | 271 | 0.483 | 0.54 | 0.38 | 0.36 |
| type: is it | 269 | 0.472 | 0.55 | 0.38 | 0.35 |
| type: is | 261 | 0.510 | 0.50 | 0.37 | 0.37 |
| type: are there | 233 | 0.506 | 0.52 | 0.40 | 0.38 |
| type: are these | 224 | 0.518 | 0.49 | 0.39 | 0.38 |
| type: are | 209 | 0.526 | 0.47 | 0.38 | 0.37 |
| type: does this | 185 | 0.454 | 0.55 | 0.40 | 0.39 |
| type: is the man | 169 | 0.533 | 0.48 | 0.37 | 0.36 |
| type: do | 128 | 0.352 | 0.66 | 0.40 | 0.39 |
| type: are there any | 126 | 0.524 | 0.46 | 0.37 | 0.37 |
| type: are they | 95 | 0.600 | 0.41 | 0.39 | 0.35 |
| type: is he | 91 | 0.527 | 0.52 | 0.38 | 0.34 |
| type: has | 84 | 0.440 | 0.56 | 0.38 | 0.37 |
| type: is this an | 79 | 0.468 | 0.57 | 0.39 | 0.36 |
| type: is the woman | 74 | 0.392 | 0.64 | 0.38 | 0.36 |
| type: can you | 70 | 0.557 | 0.43 | 0.34 | 0.35 |
| type: do you | 62 | 0.597 | 0.47 | 0.41 | 0.35 |
| type: was | 57 | 0.596 | 0.40 | 0.38 | 0.35 |
| type: is that a | 56 | 0.571 | 0.45 | 0.39 | 0.36 |
| 'Is there a X', X a COCO name | 53 | 0.736 | 0.42 | 0.48 | 0.35 |
| type: is the person | 52 | 0.558 | 0.44 | 0.38 | 0.37 |
| type: is this person | 49 | 0.551 | 0.45 | 0.39 | 0.37 |
| type: could | 44 | 0.409 | 0.59 | 0.42 | 0.41 |

Latency per request (ms): new_image p50 24.9 / p95 31.3 (n=10268)
