#!/usr/bin/env bash
# Train Snake-tuned Glance models on the GPU, then play the CPU benchmark (protocols A and B),
# with our other jobs paused for exclusive GPU during training and clean CPU timing during play.
set -uo pipefail
cd "$(dirname "$0")/.."
PIDS=$(pgrep -f "scripts/run_r1.py|scripts/frontier_eval.py|glance.train_dual" || true)
echo "pausing: ${PIDS//$'\n'/ }"
[ -n "$PIDS" ] && kill -STOP $PIDS
trap '[ -n "$PIDS" ] && kill -CONT $PIDS; echo resumed' EXIT
export PYTHONDONTWRITEBYTECODE=1

for spec in "C5 img+facts" "C6 img+facts" "C6 img" "C3 img"; do
  set -- $spec
  echo ">>> train $1 $2"
  uv run --no-sync python -m glance.snake train --cand "$1" --variant "$2" --steps 3000 --batch 64 \
    2>&1 | grep --line-buffered -E "step [0-9]+/|val_move_accuracy|Error|Traceback"
done
for run in results/snake/C5_img_facts results/snake/C6_img_facts results/snake/C6_img results/snake/C3_img; do
  [ -f "$run/model.pt" ] || { echo "missing $run"; continue; }
  echo ">>> play $run"
  uv run --no-sync python -m glance.snake play --run "$run" --threads 4 2>&1 | grep --line-buffered -E '"decisions_per_s"|"food_30s"|"moves_mean"|"legal_rate"|"deaths"|Error'
done
