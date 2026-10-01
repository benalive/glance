#!/usr/bin/env bash
# After the Qwen3-VL teacher labels exist: (B3) C5 distilled from Qwen on the licence-clean mix, and
# (C) C5 trained on the licence-clean AI-image error mix, starting from the clean C5. Both from 2 epochs,
# lrA, seed 0; evaluated on the everyday and AI-image benchmarks.
#   scripts/run_phase_bc.sh
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
free_gb() { df -g / | awk 'NR==2{print $4}'; }
( while sleep 30; do if [ "$(free_gb)" -lt 3 ]; then echo "!!! watchdog: $(free_gb) GB free, stopping"; pkill -f "train_candidate.py|eval_checkpoint.py"; fi; done ) &
WD=$!
trap 'kill $WD' EXIT
EVERYDAY=vqav2_val_yesno,pope_random,pope_popular
ART=had_val_hands,had_val_any,richhf_test_artifacts,richhf_test_plausibility,had_val_hands_anatomy_correct,had_val_any_looks_correct,had_val_any_anatomy_correct,richhf_test_looks_correct,had_val_any_anatomically_off,had_val_any_people_realistic,real_photos_no_errors
EXT=salart_q1,salart_q2,salart_q3,salart_q4,artibench,abench_generative
run() {  # out, eval benches, train args...
  local out=$1 benches=$2; shift 2
  echo ">>> train $out $(date) free $(free_gb) GB"
  uv run --no-sync python scripts/train_candidate.py --cand C5 --epochs 2 --seed 0 --out "results/r3/$out" "$@" 2>&1 \
    | grep --line-buffered -E "step [0-9]+/|eval |'dev'|Error|Traceback|error"
  [ -f "data/ckpt/$out.pt" ] && uv run --no-sync python scripts/eval_checkpoint.py --run "results/r3/$out" \
    --benches "$benches" 2>&1 | grep -E "eval |Error|Traceback"
}
run C5_qwen_clean_ep2 $EVERYDAY --data-root data/clean_v1 --distill-teacher teacher_qwen3vl2b.jsonl --distill-alpha 0.5
ln -sf ../clean_v1/teacher_qwen3vl2b.jsonl data/clean_art_v1/teacher_qwen3vl2b.jsonl
run C5_art_clean_ep2 $EVERYDAY,$ART,$EXT --data-root data/clean_art_v1 --init-ckpt data/ckpt/C5_clean_ep2.pt
echo ">>> done $(date) free $(free_gb) GB"
