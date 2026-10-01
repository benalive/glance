"""Download published benchmarks for judging AI-generated images, whose papers report scores for
leading open-weight and closed models (see the paper's comparison table).

  SalArt-VQA (arXiv 2606.12671; HF salartvqa/SalArt-VQA, CC-BY-4.0): 950 images (475 with a salient
    generation artifact, 356 real COCO photos, 119 matched generated images without it); four questions
    per image (yes/no; region, box and description, each A-E with E = none).
  ArtiBench (arXiv 2602.20951; HF KRAFTON/ArtiBench, CC-BY-NC-4.0): 1,000 generated images, 500 with
    visual artifacts; yes/no. Images are taken from the repo's parquet files: 197 of the separate
    images/*.png files on the Hub are truncated (their sizes are multiples of 256 KiB and PIL cannot
    decode them), while the parquet copies are complete.
  A-Bench (arXiv 2406.03070; VLMEvalKit TSV zhangzicheng/abench_tsv): validation split with answers;
    its "generative distortion" subset (250 multiple-choice questions).

    uv run python scripts/fetch_external.py   # -> data/external/{salart,artibench,abench}/
"""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

OUT = Path(os.environ.get("GLANCE_DATA_DIR", "data")) / "external"


def main():
    snapshot_download("salartvqa/SalArt-VQA", repo_type="dataset", local_dir=OUT / "salart",
                      allow_patterns=["data/test.jsonl", "images/artifact/*", "images/clean/*", "images/q3_overlay/*"])
    print("salart done", flush=True)
    import json

    import pyarrow.parquet as pq

    img_dir = OUT / "artibench" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for shard in range(3):
        path = hf_hub_download("KRAFTON/ArtiBench", f"data/test-0000{shard}-of-00003.parquet", repo_type="dataset",
                               local_dir=OUT / "artibench")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for r in batch.to_pylist():
                (img_dir / f"{r['id']}.png").write_bytes(r["image"]["bytes"])
                labels.append({k: r[k] for k in ("id", "has_artifacts", "explanation", "bboxes")})
        os.remove(path)
    (OUT / "artibench" / "labels.json").write_text(json.dumps(labels))
    print(f"artibench: {len(labels)} images", flush=True)
    hf_hub_download("zhangzicheng/abench_tsv", "A-bench_VAL.tsv", repo_type="dataset", local_dir=OUT / "abench")
    print("abench done", flush=True)


if __name__ == "__main__":
    main()
