#!/usr/bin/env bash
# Like-for-like CPU comparison: Laya-snake vs Snake-tuned Glance, same 4 threads, both protocols,
# on a quiet machine (our other jobs paused; bash so the PID list word-splits).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
OUT=results/bench/snake_cpu.jsonl
PIDS=$(pgrep -f "scripts/run_r1.py|scripts/train_candidate.py|scripts/frontier_eval.py|glance.train_dual|glance.snake train" || true)
echo "pausing: ${PIDS//$'\n'/ }"
[ -n "$PIDS" ] && kill -STOP $PIDS
trap '[ -n "$PIDS" ] && kill -CONT $PIDS; echo resumed' EXIT

echo ">>> laya-snake, torch CPU, 4 threads"
uv run --no-sync python scripts/snake_cpu_bench.py --runtime laya-torch --model madhavbiplov/laya-snake-mlx \
  --threads 4 --protocol both --out "$OUT" > /tmp/claude-501/laya_t4.log 2>&1 || tail -5 /tmp/claude-501/laya_t4.log
for run in results/snake/C5_text_matched results/snake/C5_img_facts_matched; do
  echo ">>> play $run"
  uv run --no-sync python -m glance.snake play --run "$run" --threads 4 --out "$OUT" > /tmp/claude-501/play.log 2>&1 \
    || tail -5 /tmp/claude-501/play.log
done
