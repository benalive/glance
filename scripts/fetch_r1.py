"""Download the R1 training subsets: a few VQAv2 train shards and COCO-2017 detection train shards.

VQAv2 train images are COCO train2014 (disjoint from POPE's val2014 and from val2017). The COCO
detection train split is train2017, which re-uses ~35k val2014 images, so POPE's 500 image ids
are excluded downstream in glance/data/train_mix.py.
"""
from huggingface_hub import HfApi, snapshot_download

N_VQA, N_COCO = 7, 6
api = HfApi()
for repo, prefix, n in [("pingzhili/vqa_v2", "data/train-", N_VQA), ("detection-datasets/coco", "data/train-", N_COCO)]:
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(prefix))
    pick = files[:n]
    path = snapshot_download(repo, repo_type="dataset", allow_patterns=pick)
    print(repo, len(files), "train shards available; fetched", pick, "->", path, flush=True)
print("DONE", flush=True)
