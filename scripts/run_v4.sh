#!/usr/bin/env bash
# Mix v4 (v3 + real close-up portraits as 'no errors' examples): build, licence check, stage 2 for seeds
# 0-2, then export the seed-0 package and rerun the FairFace false-alarm check.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
set -e
uv run --no-sync python scripts/build_clean_artifact_mix.py --general data/clean_v2 --out data/clean_art_v4 --templated-wordings --face-negatives 2>&1 | grep -v -i warn | tail -40
uv run --no-sync python scripts/check_clean_licences.py data/clean_art_v4 | tail -2
rm -f data/clean_art_v3/cache_C5_256.npy data/clean_art_v3/cache_C5_256.json
MIX=data/clean_art_v4 TAG=open4 scripts/run_stage2.sh 0 1 2
uv run --no-sync python scripts/export_release.py --run results/r4/C5_open4_s0 --mix data/clean_art_v4 --out data/release/glance-c5-open-v4 2>&1 | grep -E "noul|choice|score|wrote|difference|Error|Traceback"
uv run --no-sync python scripts/fairness_check.py --package data/release/glance-c5-open-v4 2>&1 | grep -E "^\||^###|Error|Traceback"
echo ">>> v4 done $(date)"
