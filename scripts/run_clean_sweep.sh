#!/usr/bin/env bash
# Everyday-question sweep on the licence-clean mix (Phase B): each candidate from base weights, 2 epochs,
# lrA, seed 0, then the extra everyday benchmarks. A watchdog stops training if the disk runs low.
#   scripts/run_clean_sweep.sh C5 C5f6 C5u4      # -> results/r3/<cand>_clean_ep2, data/ckpt/<cand>_clean_ep2.pt
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
DATA_ROOT=${DATA_ROOT:-data/clean_v1}
EXTRA=vqav2_val_yesno,pope_random,pope_popular
free_gb() { df -g / | awk 'NR==2{print $4}'; }
( while sleep 30; do if [ "$(free_gb)" -lt 3 ]; then echo "!!! watchdog: $(free_gb) GB free, stopping"; pkill -f "train_candidate.py|eval_checkpoint.py"; fi; done ) &
WD=$!
trap 'kill $WD' EXIT
for cand in "${@:-C5 C5f6 C5u4}"; do
  out=${cand}_clean_ep2
  # a C5u<k> cache (frozen-bottom outputs) needs the disk the shared C5 cache used
  [[ $cand == C5u* ]] && rm -f "$DATA_ROOT/cache_C5_256.npy" "$DATA_ROOT/cache_C5_256.json"
  echo ">>> train $out $(date) free $(free_gb) GB"
  uv run --no-sync python scripts/train_candidate.py --cand "$cand" --epochs 2 --seed 0 --out "results/r3/$out" \
    --data-root "$DATA_ROOT" 2>&1 | grep --line-buffered -E "step [0-9]+/|eval |'dev'|Error|Traceback|error"
  [ -f "data/ckpt/$out.pt" ] && uv run --no-sync python scripts/eval_checkpoint.py --run "results/r3/$out" \
    --benches $EXTRA 2>&1 | grep -E "eval |Error|Traceback"
done
echo ">>> done $(date) free $(free_gb) GB"
