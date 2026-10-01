#!/usr/bin/env bash
# Snake round 2 (after the readout fix): matched-data rematch (C5 on Laya-snake's data budget) and
# pure vision with SigLIP-initialised towers, with and without auxiliary perception questions. Training shares the GPU with the R1 queue; the CPU play step
# pauses our other jobs for clean timing.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
train() { echo ">>> train $*"; uv run --no-sync python -m glance.snake train "$@" 2>&1 | \
  grep --line-buffered -E "step [0-9]+/|val_move_accuracy|Error|Traceback"; }

# (C5 matched-data rematch already trained: results/snake/C5_img_facts_matched)
train --cand C6s --variant img --steps 1500 --batch 64
train --cand C6s --variant img+aux --steps 1500 --batch 64
train --cand C3 --variant img+aux --steps 1500 --batch 64

PIDS=$(pgrep -f "scripts/run_r1.py|scripts/train_candidate.py|scripts/frontier_eval.py|glance.train_dual" || true)
[ -n "$PIDS" ] && kill -STOP $PIDS
trap '[ -n "$PIDS" ] && kill -CONT $PIDS; echo resumed' EXIT
for run in results/snake/C5_img_facts_matched results/snake/C6s_img results/snake/C6s_img_aux results/snake/C3_img_aux; do
  [ -f "$run/model.pt" ] || { echo "missing $run"; continue; }
  echo ">>> play $run"
  uv run --no-sync python -m glance.snake play --run "$run" --threads 4 2>&1 | \
    grep --line-buffered -E '"decisions_per_s"|"food_30s"|"deaths"|"moves_mean"|"legal_rate"|Error'
done
