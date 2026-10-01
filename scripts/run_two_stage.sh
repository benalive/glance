#!/usr/bin/env bash
# Release-candidate recipe: C5 from base weights, 2 epochs on the licence-clean general mix (clean_v2), then
# 2 epochs on the licence-clean error mix (clean_art_v2); no teacher. One run per seed given.
#   scripts/run_two_stage.sh 1 2      # -> results/r4/C5_open_2stage_s<seed>
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
free_gb() { df -g / | awk 'NR==2{print $4}'; }
( while sleep 30; do if [ "$(free_gb)" -lt 3 ]; then echo "!!! watchdog: $(free_gb) GB free, stopping"; pkill -f "train_candidate.py|eval_checkpoint.py"; fi; done ) &
WD=$!
trap 'kill $WD' EXIT
BENCHES=$(grep -o 'BENCHES=[^ ]*' scripts/run_final.sh | head -1 | cut -d= -f2)
for seed in "$@"; do
  echo ">>> general stage, seed $seed $(date) free $(free_gb) GB"
  uv run --no-sync python scripts/train_candidate.py --cand C5 --seed "$seed" --epochs 2 --out "results/r4/C5_general_v2_s$seed" \
    --data-root data/clean_v2 2>&1 | grep --line-buffered -E "'dev'|Error|Traceback"
  echo ">>> error stage, seed $seed $(date) free $(free_gb) GB"
  uv run --no-sync python scripts/train_candidate.py --cand C5 --seed "$seed" --epochs 2 --out "results/r4/C5_open_2stage_s$seed" \
    --data-root data/clean_art_v2 --init-ckpt "data/ckpt/C5_general_v2_s$seed.pt" 2>&1 | grep --line-buffered -E "eval |'dev'|Error|Traceback"
  [ -f "data/ckpt/C5_open_2stage_s$seed.pt" ] && uv run --no-sync python scripts/eval_checkpoint.py \
    --run "results/r4/C5_open_2stage_s$seed" --benches "$BENCHES" 2>&1 | grep -E "eval |Error|Traceback"
done
echo ">>> done $(date) free $(free_gb) GB"
