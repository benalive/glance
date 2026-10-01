#!/usr/bin/env bash
# Set up an isolated Python 3.12 environment with uv (installed to ~/.local/bin if missing), install the
# CPU build of PyTorch and the few libraries Glance needs, then run the latency benchmark.
#   ./run.sh                                       # untrained architecture from the public base models
#   ./run.sh --package /path/to/glance-c5-open-v4  # the release package, if you copied it over
#   ./run.sh --variants [--package ...]            # speed-ups (bf16, int8, torch.compile, ONNX) vs fp32
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.14.*" \
  || uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python "transformers==5.17.*" safetensors pillow numpy scipy onnx onnxruntime onnxscript
# nothing else should be running on the machine while this measures
SCRIPT=bench.py
if [ "${1:-}" = "--variants" ]; then SCRIPT=bench_variants.py; shift; fi
PYTHONPATH=. .venv/bin/python "$SCRIPT" "$@" 2>&1 | grep -v -iE "warn|UNEXPECTED|^\s*$|Notes:|can be ignored|^Key |^-+\+"
