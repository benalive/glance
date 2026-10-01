FairFace (10954 real portraits) with "Does this image contain visible generation errors?": share flagged at the served threshold (P >= 0.5) and at the threshold where the model finds 45% of SalArt's 437 flawed images (matched); AUROC of SalArt flawed images against each group's portraits.

### released C5 (v4 seed 0): SalArt recall at 0.5 = 0.339; matched threshold P >= 0.352

| group | n | flagged (served) | flagged (matched) | AUROC flawed vs group |
|---|---|---|---|---|
| all | 10954 | 0.14% | 0.37% | 0.969 |
| race: Black | 1556 | 0.32% | 1.09% | 0.952 |
| race: East Asian | 1550 | 0.00% | 0.13% | 0.968 |
| race: Indian | 1516 | 0.13% | 0.26% | 0.974 |
| race: Latino_Hispanic | 1623 | 0.00% | 0.18% | 0.978 |
| race: Middle Eastern | 1209 | 0.33% | 0.50% | 0.962 |
| race: Southeast Asian | 1415 | 0.14% | 0.35% | 0.975 |
| race: White | 2085 | 0.10% | 0.19% | 0.970 |
| gender: Female | 5162 | 0.15% | 0.29% | 0.975 |
| gender: Male | 5792 | 0.12% | 0.45% | 0.963 |
| age: 0-2 | 199 | 0.50% | 0.50% | 0.967 |
| age: 10-19 | 1181 | 0.25% | 0.68% | 0.962 |
| age: 20-29 | 3300 | 0.09% | 0.27% | 0.973 |
| age: 3-9 | 1356 | 0.22% | 0.88% | 0.954 |
| age: 30-39 | 2330 | 0.17% | 0.34% | 0.970 |
| age: 40-49 | 1353 | 0.07% | 0.15% | 0.974 |
| age: 50-59 | 796 | 0.00% | 0.00% | 0.973 |
| age: 60-69 | 321 | 0.00% | 0.31% | 0.972 |
| age: more than 70 | 118 | 0.00% | 0.00% | 0.972 |

### research C5 (distilled): SalArt recall at 0.5 = 0.762; matched threshold P >= 0.857

| group | n | flagged (served) | flagged (matched) | AUROC flawed vs group |
|---|---|---|---|---|
| all | 10954 | 2.86% | 0.06% | 0.916 |
| race: Black | 1556 | 5.40% | 0.13% | 0.899 |
| race: East Asian | 1550 | 1.48% | 0.00% | 0.931 |
| race: Indian | 1516 | 3.50% | 0.07% | 0.908 |
| race: Latino_Hispanic | 1623 | 1.97% | 0.00% | 0.917 |
| race: Middle Eastern | 1209 | 4.80% | 0.17% | 0.905 |
| race: Southeast Asian | 1415 | 1.48% | 0.07% | 0.925 |
| race: White | 2085 | 2.01% | 0.05% | 0.920 |
| gender: Female | 5162 | 2.67% | 0.08% | 0.914 |
| gender: Male | 5792 | 3.02% | 0.05% | 0.917 |
| age: 0-2 | 199 | 1.01% | 0.00% | 0.925 |
| age: 10-19 | 1181 | 2.20% | 0.00% | 0.921 |
| age: 20-29 | 3300 | 2.39% | 0.12% | 0.923 |
| age: 3-9 | 1356 | 2.95% | 0.15% | 0.911 |
| age: 30-39 | 2330 | 2.10% | 0.04% | 0.920 |
| age: 40-49 | 1353 | 2.81% | 0.00% | 0.914 |
| age: 50-59 | 796 | 4.90% | 0.00% | 0.896 |
| age: 60-69 | 321 | 8.41% | 0.00% | 0.876 |
| age: more than 70 | 118 | 11.02% | 0.00% | 0.852 |
