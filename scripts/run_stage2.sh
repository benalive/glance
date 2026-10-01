#!/usr/bin/env bash
# Stage 2 of the two-stage recipe on a given error mix, from the existing stage-1 checkpoints
# (data/ckpt/C5_general_v2_s<seed>.pt); then the benchmarks and the wording probe.
#   MIX=data/clean_art_v3 TAG=open3 scripts/run_stage2.sh 0 1 2      # -> results/r4/C5_<TAG>_s<seed>
#   TRAIN_ARGS=--no-cache-features adds training options (e.g. when the feature cache would not fit on disk)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
MIX=${MIX:?set MIX}; TAG=${TAG:?set TAG}
free_gb() { df -g / | awk 'NR==2{print $4}'; }
( while sleep 30; do if [ "$(free_gb)" -lt 3 ]; then echo "!!! watchdog: $(free_gb) GB free, stopping"; pkill -f "train_candidate.py|eval_checkpoint.py"; fi; done ) &
WD=$!
trap 'kill $WD' EXIT
BENCHES=$(grep -o 'BENCHES=[^ ]*' scripts/run_final.sh | head -1 | cut -d= -f2)
RUNS=""
for seed in "$@"; do
  out=C5_${TAG}_s$seed
  echo ">>> stage 2 $out $(date) free $(free_gb) GB"
  uv run --no-sync python scripts/train_candidate.py --cand C5 --seed "$seed" --epochs 2 --out "results/r4/$out" \
    --data-root "$MIX" --init-ckpt "data/ckpt/C5_general_v2_s$seed.pt" ${TRAIN_ARGS:-} 2>&1 | grep --line-buffered -E "eval |'dev'|Error|Traceback"
  [ -f "data/ckpt/$out.pt" ] && uv run --no-sync python scripts/eval_checkpoint.py --run "results/r4/$out" \
    --benches "$BENCHES" 2>&1 | grep -E "eval |Error|Traceback"
  RUNS="$RUNS,results/r4/$out"
done
uv run --no-sync python scripts/wording_probe.py --runs "${RUNS#,}" --mix "$MIX" 2>&1 | grep -E "images:|mean AUROC|Error|Traceback"
echo ">>> done $(date) free $(free_gb) GB"
