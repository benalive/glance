#!/usr/bin/env bash
# CPU Snake benchmark driver: pause our own GPU/CPU jobs for clean timing, run every
# configuration in its own process (torch thread settings are per process), then resume.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/bench/snake_cpu.jsonl
rm -f "$OUT"
PIDS=$(pgrep -f "scripts/run_r1.py|scripts/frontier_eval.py|glance.train_dual" || true)
echo "pausing: ${PIDS//$'\n'/ }"
[ -n "$PIDS" ] && kill -STOP $PIDS
trap '[ -n "$PIDS" ] && kill -CONT $PIDS; echo resumed' EXIT

run() {
  echo ">>> $*"
  PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python scripts/snake_cpu_bench.py "$@" --out "$OUT" \
    > /tmp/claude-501/bench_one.log 2>&1 || { echo FAILED; tail -5 /tmp/claude-501/bench_one.log; }
}

run --runtime teacher
run --runtime laya-torch --model madhavbiplov/laya-snake-mlx --threads 10
run --runtime laya-torch --model madhavbiplov/laya-snake-mlx --threads 4 --protocol a
run --runtime laya-torch --model convaiinnovations/laya-multilingual --threads 10
run --runtime laya-torch --model convaiinnovations/laya --threads 10
run --runtime laya-mlx --model madhavbiplov/laya-snake-mlx --threads 10
for c in C6 C3 C5; do
  run --runtime glance --model "$c" --threads 10
  run --runtime glance --model "$c" --threads 4
done
wc -l "$OUT"
