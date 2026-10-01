"""Download the R0 eval sets, baseline models, and candidate configs into the HF cache.

KonIQ-10k has no parquet release, so its 512x384 zip is unpacked into data/raw/koniq10k.
"""
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

DATASETS = {
    "lmms-lab-encoder/POPE": ["Full/adversarial-*.parquet"],
    "HuggingFaceM4/A-OKVQA": ["data/validation-*.parquet", "data/train-*.parquet"],
    "Lin-Chen/MMStar": ["mmstar.parquet"],
    "HuggingFaceM4/SugarCrepe_replace_rel": ["data/*.parquet"],
    "HuggingFaceM4/SugarCrepe_swap_att": ["data/*.parquet"],
    "chaofengc/IQA-PyTorch-Datasets-metainfo": ["meta_info_KonIQ10kDataset.csv"],
}
MODELS = {
    "google/siglip2-base-patch32-256": None,
    "google/siglip2-base-patch16-256": None,
    "HuggingFaceTB/SmolVLM-256M-Instruct": ["*.json", "*.txt", "model.safetensors"],
    "ModernVBERT/modernvbert": None,
    "jhu-clsp/ettin-encoder-32m": None,
    "Qwen/Qwen3-VL-2B-Instruct": None,
}

for repo, patterns in DATASETS.items():
    path = snapshot_download(repo, repo_type="dataset", allow_patterns=patterns)
    print("dataset", repo, "->", path, flush=True)

zip_path = hf_hub_download("IntMeGroup/DFBench", "koniq10k_512x384.zip", repo_type="dataset")
out = Path("data/raw/koniq10k")
if not out.exists():
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
print("koniq ->", out, sum(1 for _ in out.rglob("*.jpg")), "jpgs", flush=True)

for repo, patterns in MODELS.items():
    path = snapshot_download(repo, allow_patterns=patterns)
    print("model", repo, "->", path, flush=True)
print("DONE", flush=True)
