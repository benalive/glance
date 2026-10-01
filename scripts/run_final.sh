#!/usr/bin/env bash
# Final recipe on the licence-clean mix v2 (data/clean_art_v2): C5 from base weights, 2 epochs, lrA,
# Qwen3-VL soft labels (alpha 0.5) on the everyday questions, three seeds; plus a no-teacher control.
# Every run is evaluated on the everyday and AI-image benchmarks.
#   scripts/run_final.sh      # -> results/r4/<run>, data/ckpt/<run>.pt
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
free_gb() { df -g / | awk 'NR==2{print $4}'; }
( while sleep 30; do if [ "$(free_gb)" -lt 3 ]; then echo "!!! watchdog: $(free_gb) GB free, stopping"; pkill -f "train_candidate.py|eval_checkpoint.py"; fi; done ) &
WD=$!
trap 'kill $WD' EXIT
BENCHES=vqav2_val_yesno,pope_random,pope_popular,had_val_hands,had_val_any,richhf_test_artifacts,richhf_test_plausibility,had_val_hands_anatomy_correct,had_val_any_looks_correct,had_val_any_anatomy_correct,richhf_test_looks_correct,had_val_any_anatomically_off,had_val_any_people_realistic,real_photos_no_errors,salart_q1,salart_q2,salart_q3,salart_q4,artibench,abench_generative
run() {  # out, train args...
  local out=$1; shift
  echo ">>> train $out $(date) free $(free_gb) GB"
  uv run --no-sync python scripts/train_candidate.py --cand C5 --epochs 2 --out "results/r4/$out" --data-root data/clean_art_v2 "$@" 2>&1 \
    | grep --line-buffered -E "step [0-9]+/|eval |'dev'|Error|Traceback|error"
  [ -f "data/ckpt/$out.pt" ] && uv run --no-sync python scripts/eval_checkpoint.py --run "results/r4/$out" \
    --benches "$BENCHES" 2>&1 | grep -E "eval |Error|Traceback"
}
for seed in 0 1 2; do
  run "C5_open_s$seed" --seed "$seed" --distill-teacher teacher_qwen3vl2b.jsonl --distill-alpha 0.5
done
run C5_open_noteacher_s0 --seed 0
echo ">>> done $(date) free $(free_gb) GB"
