"""Label every question of a training mix with a trained teacher's probabilities (for distillation).

The teacher (default: C1 fine-tuned on the error mix) answers each question of <data_root>/decisions.jsonl,
grouped by image and packed exactly as in training; its softmax probabilities (temperature 1) are written
to <data_root>/teacher_<run>.jsonl as {"uid": ..., "p": [...]}. Training then mixes them into the targets
(TrainConfig.distill_alpha); the dev split keeps its gold labels.

    uv run python scripts/teacher_label.py --data-root data/r2distill --run results/r2/C1_art2_ep2
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from glance.data.packing import collate, group_by_image
from glance.model.candidates import build_candidate, load_trained, read_state
from glance.model.heads import pad_by_question
from glance.train import DEVICE, R1Data, TrainConfig, _to, image_batch


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", required=True, help="teacher run dir (config.json; checkpoint in $GLANCE_DATA_DIR/ckpt)")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = TrainConfig(**{k: v for k, v in json.loads((run / "config.json").read_text()).items() if k in TrainConfig.__dataclass_fields__})
    cand = build_candidate(cfg.cand)
    ckpt = Path(os.environ.get("GLANCE_DATA_DIR", "data")) / "ckpt" / f"{run.name}.pt"
    load_trained(cand, read_state(ckpt), str(ckpt))
    cand = cand.to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m")
    data = R1Data(Path(a.data_root))
    decisions = data.decisions("train") + data.decisions("dev")
    seqs = group_by_image(decisions, tok, cfg.max_len, cfg.max_q_per_seq)
    seqs.sort(key=lambda s: s[0][0].meta["image_row"])  # image rows in order: sequential reads
    out_path = Path(a.data_root) / f"teacher_{run.name}.jsonl"
    done = 0
    with open(out_path, "w") as fh:
        for s in range(0, len(seqs), a.batch):
            chunk = seqs[s:s + a.batch]
            img = image_batch(cand, data, [ds[0].meta["image_row"] for ds, _ in chunk], None)
            b = _to(collate([es for _, es in chunk], tok.pad_token_id))
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=cfg.amp):
                logits = cand(cand.image_top(img), b)
            p = torch.softmax(pad_by_question(logits.float(), b["slot_q"], b["slot_k"], len(b["q_pos"]), b["k_max"]), -1).cpu()
            qi = 0
            for ds, _ in chunk:
                for d in ds:
                    fh.write(json.dumps({"uid": d.uid, "p": [round(float(x), 5) for x in p[qi, :len(d.candidates)]]}) + "\n")
                    qi += 1
            done += len(chunk)
            if (s // a.batch) % 200 == 0:
                print(f"teacher: {done}/{len(seqs)} sequences", flush=True)
    print(f"teacher: {len(decisions)} questions labelled -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
