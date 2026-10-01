Temperature (or Platt, yes/no only) fitted on the 20% image-grouped calib split; everything else on the report split. KonIQ accuracy is omitted (constant 'fair' scores 0.511); KonIQ SRCC is from raw logits. 'rotation Kx' = probabilities averaged over K option rotations (K forward passes).

| method | bench | readout | n | acc [95% CI] | AUROC | smECE raw | smECE cal | Brier cal | NLL cal | T | SRCC (raw) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3vl_2b | aokvqa_val | single pass | 927 | 0.748 [0.720, 0.774] | nan | 0.153 | 0.040 | 0.344 | 0.666 | 2.12 | nan |
| qwen3vl_2b | aokvqa_val | rotation Kx | 927 | 0.762 [0.734, 0.788] | nan | 0.114 | 0.029 | 0.324 | 0.623 | 1.82 | nan |
| qwen3vl_2b | koniq_test | single pass | 1637 | — | nan | 0.409 | 0.047 | 0.162 | 1.227 | 4.61 | 0.635 |
| qwen3vl_2b | pope_adversarial | single pass | 2334 | 0.840 [0.826, 0.853] | 0.916 | 0.118 | 0.021 | 0.229 | 0.366 | 3.65 | nan |
| qwen3vl_2b | pope_adversarial | single pass+platt | 2334 | 0.841 [0.827, 0.855] | 0.916 | 0.118 | 0.020 | 0.228 | 0.367 | 1.00 | nan |
| qwen3vl_2b | sugarcrepe_replace_rel | single pass | 1105 | 0.890 [0.871, 0.909] | nan | 0.055 | 0.019 | 0.167 | 0.275 | 2.01 | nan |
| qwen3vl_2b | sugarcrepe_replace_rel | rotation Kx | 1105 | 0.916 [0.898, 0.934] | nan | 0.027 | 0.032 | 0.132 | 0.231 | 1.23 | nan |
| qwen3vl_2b | sugarcrepe_swap_att | single pass | 541 | 0.939 [0.919, 0.959] | nan | 0.024 | 0.020 | 0.093 | 0.157 | 1.22 | nan |
| qwen3vl_2b | sugarcrepe_swap_att | rotation Kx | 541 | 0.967 [0.950, 0.981] | nan | 0.040 | 0.025 | 0.065 | 0.115 | 0.73 | nan |
| qwen3vl_2b | vqav2_val_yesno | single pass | 5456 | 0.808 [0.797, 0.818] | 0.900 | 0.142 | 0.014 | 0.161 | 0.456 | 3.69 | nan |
| qwen3vl_2b | vqav2_val_yesno | single pass+platt | 5456 | 0.812 [0.801, 0.823] | 0.900 | 0.142 | 0.013 | 0.155 | 0.450 | 1.00 | nan |
| siglip2_b16 | aokvqa_val | qo | 927 | 0.600 [0.567, 0.629] | nan | 0.046 | 0.035 | 0.519 | 1.013 | 1.11 | nan |
| siglip2_b16 | koniq_test | antonym | 1637 | — | nan | 0.500 | 0.199 | 0.295 | 1.607 | 14.37 | 0.188 |
| siglip2_b16 | mmstar | qo | 1191 | 0.336 [0.310, 0.363] | nan | 0.147 | 0.054 | 0.691 | 1.261 | 2.27 | nan |
| siglip2_b16 | pope_adversarial | single pass | 2334 | 0.510 [0.504, 0.516] | 0.842 | 0.470 | 0.122 | 0.484 | 0.676 | 20.00 | nan |
| siglip2_b16 | pope_adversarial | single pass+platt | 2334 | 0.771 [0.755, 0.784] | 0.842 | 0.470 | 0.039 | 0.324 | 0.500 | 1.00 | nan |
| siglip2_b16 | sugarcrepe_replace_rel | cap | 1105 | 0.692 [0.662, 0.722] | nan | 0.067 | 0.046 | 0.375 | 0.548 | 1.18 | nan |
| siglip2_b16 | sugarcrepe_swap_att | cap | 541 | 0.732 [0.697, 0.768] | nan | 0.035 | 0.027 | 0.347 | 0.511 | 1.16 | nan |
| siglip2_b16 | vqav2_val_yesno | single pass | 5456 | 0.549 [0.535, 0.563] | 0.560 | 0.050 | 0.033 | 0.362 | 0.692 | 8.75 | nan |
| siglip2_b16 | vqav2_val_yesno | single pass+platt | 5456 | 0.488 [0.474, 0.503] | 0.560 | 0.050 | 0.041 | 0.366 | 0.696 | 1.00 | nan |
| siglip2_b32 | aokvqa_val | qo | 927 | 0.589 [0.558, 0.619] | nan | 0.037 | 0.037 | 0.536 | 1.041 | 1.24 | nan |
| siglip2_b32 | koniq_test | antonym | 1637 | — | nan | 0.513 | 0.206 | 0.295 | 1.604 | 12.22 | 0.175 |
| siglip2_b32 | mmstar | qo | 1191 | 0.344 [0.318, 0.370] | nan | 0.126 | 0.043 | 0.694 | 1.265 | 2.15 | nan |
| siglip2_b32 | pope_adversarial | single pass | 2334 | 0.507 [0.501, 0.512] | 0.828 | 0.474 | 0.130 | 0.492 | 0.683 | 20.00 | nan |
| siglip2_b32 | pope_adversarial | single pass+platt | 2334 | 0.754 [0.740, 0.770] | 0.828 | 0.474 | 0.033 | 0.339 | 0.517 | 1.00 | nan |
| siglip2_b32 | sugarcrepe_replace_rel | cap | 1105 | 0.713 [0.682, 0.742] | nan | 0.037 | 0.043 | 0.365 | 0.537 | 0.95 | nan |
| siglip2_b32 | sugarcrepe_swap_att | cap | 541 | 0.769 [0.733, 0.804] | nan | 0.031 | 0.034 | 0.328 | 0.492 | 1.06 | nan |
| smolvlm_256m | aokvqa_val | single pass | 927 | 0.462 [0.430, 0.495] | nan | 0.241 | 0.034 | 0.657 | 1.230 | 2.50 | nan |
| smolvlm_256m | aokvqa_val | rotation Kx | 927 | 0.569 [0.538, 0.598] | nan | 0.033 | 0.046 | 0.566 | 1.063 | 1.15 | nan |
| smolvlm_256m | koniq_test | single pass | 1637 | — | nan | 0.517 | 0.029 | 0.231 | 1.428 | 7.40 | 0.402 |
| smolvlm_256m | mmstar | single pass | 1191 | 0.316 [0.288, 0.342] | nan | 0.337 | 0.018 | 0.725 | 1.317 | 9.88 | nan |
| smolvlm_256m | mmstar | rotation Kx | 1191 | 0.348 [0.319, 0.377] | nan | 0.071 | 0.019 | 0.702 | 1.280 | 2.10 | nan |
| smolvlm_256m | pope_adversarial | single pass | 2334 | 0.603 [0.590, 0.615] | 0.837 | 0.340 | 0.065 | 0.447 | 0.637 | 11.70 | nan |
| smolvlm_256m | pope_adversarial | single pass+platt | 2334 | 0.767 [0.752, 0.782] | 0.837 | 0.340 | 0.040 | 0.326 | 0.500 | 1.00 | nan |
| smolvlm_256m | sugarcrepe_replace_rel | single pass | 1105 | 0.535 [0.506, 0.565] | nan | 0.126 | 0.035 | 0.494 | 0.688 | 2.95 | nan |
| smolvlm_256m | sugarcrepe_replace_rel | rotation Kx | 1105 | 0.705 [0.676, 0.735] | nan | 0.150 | 0.062 | 0.410 | 0.601 | 0.30 | nan |
| smolvlm_256m | sugarcrepe_swap_att | single pass | 541 | 0.536 [0.493, 0.579] | nan | 0.103 | 0.027 | 0.497 | 0.691 | 2.41 | nan |
| smolvlm_256m | sugarcrepe_swap_att | rotation Kx | 541 | 0.621 [0.579, 0.663] | nan | 0.079 | 0.053 | 0.470 | 0.668 | 0.35 | nan |
| smolvlm_256m | vqav2_val_yesno | single pass | 5456 | 0.630 [0.616, 0.643] | 0.711 | 0.255 | 0.019 | 0.319 | 0.647 | 5.84 | nan |
| smolvlm_256m | vqav2_val_yesno | single pass+platt | 5456 | 0.648 [0.635, 0.661] | 0.711 | 0.255 | 0.014 | 0.307 | 0.635 | 1.00 | nan |
