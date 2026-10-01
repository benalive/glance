Frozen-feature probes on SalArt's 119 inpainted pairs (5-fold CV grouped by pair). Median share of pixels changed by the inpainting: 0.124 (10-90%: 0.065-0.230).

| encoder | pooled probe AUROC, flawed vs clean | paired direction accuracy | control: other flawed vs real AUROC | median cosine (mean token) | median relative token change |
|---|---|---|---|---|---|
| C5: SigLIP2 B/32 @256 (released) | 0.562 | 0.899 | 0.981 | 0.9983 | 0.1423 |
| C1: SigLIP2 B/16 @512 (ModernVBERT) | 0.571 | 0.958 | 0.984 | 0.9991 | 0.1330 |
