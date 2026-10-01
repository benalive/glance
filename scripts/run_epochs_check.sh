#!/usr/bin/env bash
# Does more training on the general questions recover everyday yes/no? (follow-up to the one-stage recipe)
#   (a) one stage, 4 epochs on clean_art_v2;  (b) two stages: 2 epochs on clean_v2, then 2 on clean_art_v2.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
free_gb() { df -g / | awk 'NR==2{print $4}'; }
( while sleep 30; do if [ "$(free_gb)" -lt 3 ]; then echo "!!! watchdog: $(free_gb) GB free, stopping"; pkill -f "train_candidate.py|eval_checkpoint.py"; fi; done ) &
WD=$!
trap 'kill $WD' EXIT
BENCHES=$(grep -o 'BENCHES=[^ ]*' scripts/run_final.sh | head -1 | cut -d= -f2)
run() {  # out, eval?, train args...
  local out=$1 ev=$2; shift 2
  echo ">>> train $out $(date) free $(free_gb) GB"
  uv run --no-sync python scripts/train_candidate.py --cand C5 --seed 0 --out "results/r4/$out" "$@" 2>&1 \
    | grep --line-buffered -E "eval |'dev'|Error|Traceback|error"
  [ "$ev" = 1 ] && [ -f "data/ckpt/$out.pt" ] && uv run --no-sync python scripts/eval_checkpoint.py --run "results/r4/$out" \
    --benches "$BENCHES" 2>&1 | grep -E "eval |Error|Traceback"
}
run C5_open_ep4_s0 1 --epochs 4 --data-root data/clean_art_v2
run C5_general_v2_s0 0 --epochs 2 --data-root data/clean_v2
rm -f data/clean_v2/cache_C5_256.npy data/clean_v2/cache_C5_256.json
run C5_open_2stage_s0 1 --epochs 2 --data-root data/clean_art_v2 --init-ckpt data/ckpt/C5_general_v2_s0.pt
echo ">>> done $(date) free $(free_gb) GB"
