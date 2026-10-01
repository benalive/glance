#!/usr/bin/env bash
# Build dist/glance_x86_bench.tar.gz: the Glance code the benchmark needs, bench.py, run.sh and 45
# licence-clean test photos (CC BY 2.0, attribution included). No trained weights and no training data.
#   bench/x86/make_bundle.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=dist/glance_x86_bench
rm -rf "$OUT" && mkdir -p "$OUT/glance/model" "$OUT/glance/data" "$OUT/images"
cp bench/x86/bench.py bench/x86/bench_variants.py bench/x86/run.sh "$OUT/"
cp glance/__init__.py glance/serve.py glance/metrics.py glance/onnx_export.py "$OUT/glance/"
cp glance/model/__init__.py glance/model/candidates.py glance/model/towers.py glance/model/heads.py glance/model/masks.py "$OUT/glance/model/"
cp glance/data/__init__.py glance/data/packing.py glance/data/schema.py "$OUT/glance/data/"
python3 - "$OUT" <<'EOF'
import json, shutil, sys
out = sys.argv[1]
recs = [json.loads(l) for l in open("data/clean/attribution_coco.jsonl")][:45]
for r in recs:
    shutil.copy(r["path"], f"{out}/images/{r['image_id'].replace(':', '_')}.jpg")
with open(f"{out}/images/ATTRIBUTION.jsonl", "w") as fh:
    for r in recs:
        fh.write(json.dumps({k: r[k] for k in ("image_id", "source", "license", "original")}) + "\n")
EOF
cat > "$OUT/README.md" <<'EOF'
# Glance C5 latency on an x86 CPU

On any x86-64 Linux VM with internet access (it downloads Python, PyTorch CPU and the two public base
models, about 1.5 GB in total):

    tar xzf glance_x86_bench.tar.gz && cd glance_x86_bench && ./run.sh

Takes 5-15 minutes. `./run.sh --variants` instead times speed-ups (bf16, int8, torch.compile, ONNX
Runtime) against fp32 and checks that the answers stay the same (10-25 minutes; torch.compile warms up slowly). Keep the VM otherwise idle while it runs. Paste back everything after
"=== paste everything below back ===" (also saved as result.json).

Optional: to time the actual release package instead of the same architecture built from the public base
models (identical speed), copy data/release/glance-c5-open-v4 (542 MB) to the VM and run
`./run.sh --package /path/to/glance-c5-open-v4`. If the package has image.onnx / questions.onnx, the server
path uses ONNX Runtime by default; `--runtime torch` forces PyTorch, and `./run.sh --variants --package ...`
compares both (variant onnx-full).

The test photos are COCO photos under CC BY 2.0; see images/ATTRIBUTION.jsonl.
EOF
chmod +x "$OUT/run.sh"
COPYFILE_DISABLE=1 tar --no-xattrs -czf dist/glance_x86_bench.tar.gz -C dist glance_x86_bench  # no macOS metadata (GNU tar warns on it)
ls -la dist/glance_x86_bench.tar.gz
