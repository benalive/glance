"""C0d: supervised dual-encoder control (R0 eval audit: R1 needs supervised floors, not only
zero-shot ones).

Frozen SigLIP2-B/32 towers embed the image once and each "question candidate" text once (lowercased,
as SigLIP2 expects); both are cached. Only a small MLP head trains, on the same R1 mix, dev split,
loss (Brier / RPS + 0.3 CE), epochs and evaluation format as the fusion candidates. It answers "is
fusion needed?" at dual-encoder cost: with fixed candidate sets the per-question cost is one image
embedding plus a dot-product-sized head.

    uv run python -m glance.train_dual --epochs 2 --seed 0
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor

from glance.data.benchmarks import R0_BENCHMARKS
from glance.data.schema import calib_split
from glance.model.candidates import normalize
from glance.model.heads import decision_loss
from glance.train import DATA, DEVICE, R1Data

REPO = "google/siglip2-base-patch32-256"


def texts_for(d) -> list[str]:
    return [f"{d.question} {c}".lower() for c in d.candidates]


class DualHead(nn.Module):
    def __init__(self, d: int = 768, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * d + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, v, t, zs):
        """v (N, d) image embedding per candidate row, t (N, d) text embedding, zs (N,) zero-shot logit."""
        return self.mlp(torch.cat([v, t, v * t, zs[:, None]], dim=-1)).squeeze(-1)


class Encoder:
    def __init__(self):
        self.model = AutoModel.from_pretrained(REPO, dtype=torch.float32).to(DEVICE).eval()
        self.proc = AutoProcessor.from_pretrained(REPO)
        self.scale, self.bias = self.model.logit_scale.exp().item(), self.model.logit_bias.item()

    @torch.inference_mode()
    def images(self, u8: torch.Tensor) -> torch.Tensor:
        v = self.model.vision_model(pixel_values=normalize(u8).to(DEVICE)).pooler_output
        return torch.nn.functional.normalize(v, dim=-1).cpu()

    @torch.inference_mode()
    def texts(self, texts: list[str], batch: int = 256) -> torch.Tensor:
        out = []
        for s in range(0, len(texts), batch):
            ids = self.proc.tokenizer(texts[s:s + batch], padding="max_length", max_length=64, truncation=True,
                                      return_tensors="pt")["input_ids"].to(DEVICE)
            out.append(torch.nn.functional.normalize(self.model.text_model(input_ids=ids).pooler_output, dim=-1).cpu())
        return torch.cat(out)


def _rows(decisions, img_emb, txt_index, txt_emb, enc):
    """Flatten decisions into candidate rows: (v, t, zero-shot logit, question index, k)."""
    v, t, q, k, targets, is_score = [], [], [], [], [], []
    for qi, (d, iv) in enumerate(zip(decisions, img_emb)):
        for j, s in enumerate(texts_for(d)):
            v.append(iv)
            t.append(txt_emb[txt_index[s]])
            q.append(qi)
            k.append(j)
        targets.append(d.target)
        is_score.append(d.kind == "score")
    v, t = torch.stack(v), torch.stack(t)
    zs = enc.scale * (v * t).sum(-1) + enc.bias
    return v, t, zs, torch.tensor(q), torch.tensor(k), targets, torch.tensor(is_score)


def _loss_for(head, batch_q, rows):
    v, t, zs, q, k, targets, is_score = rows
    sel = torch.isin(q, batch_q)
    logits = head(v[sel].to(DEVICE), t[sel].to(DEVICE), zs[sel].to(DEVICE))
    remap = {int(x): i for i, x in enumerate(batch_q.tolist())}
    qq = torch.tensor([remap[int(x)] for x in q[sel]])
    k_max = max(len(targets[i]) for i in batch_q.tolist())
    padded = torch.full((len(batch_q), k_max), float("-inf"), device=DEVICE)
    padded[qq, k[sel]] = logits
    tg = torch.tensor([targets[i] + [0.0] * (k_max - len(targets[i])) for i in batch_q.tolist()], device=DEVICE)
    return decision_loss(padded, tg, is_score[batch_q].to(DEVICE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N train/dev decisions")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = Path(a.out or f"results/r1/C0d_ep{a.epochs:g}_lrA_s{a.seed}")
    (out / "eval").mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.time()
    enc, data = Encoder(), R1Data()
    train, dev = data.decisions("train"), data.decisions("dev")
    if a.limit:
        train, dev = train[: a.limit], dev[: a.limit // 10]

    img_cache = DATA / "cache_C0d_img.npy"
    if not img_cache.exists():
        n = len(data.offsets) - 1
        embs = torch.cat([enc.images(data.pixels_u8(list(range(s, min(n, s + 64))), 256)) for s in range(0, n, 64)])
        np.save(img_cache, embs.numpy())
    img_all = torch.from_numpy(np.load(img_cache))

    uniq = sorted({s for d in train + dev for s in texts_for(d)})
    txt_emb = enc.texts(uniq)
    txt_index = {s: i for i, s in enumerate(uniq)}
    tr_rows = _rows(train, img_all[[d.meta["image_row"] for d in train]], txt_index, txt_emb, enc)
    dv_rows = _rows(dev, img_all[[d.meta["image_row"] for d in dev]], txt_index, txt_emb, enc)

    head = DualHead().to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.05)
    steps = math.ceil(a.epochs * len(train) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    order, ptr = list(range(len(train))), len(train)
    for step in range(steps):
        if ptr + a.batch > len(order):
            random.shuffle(order)
            ptr = 0
        bq = torch.tensor(sorted(order[ptr:ptr + a.batch]))
        ptr += a.batch
        loss, _ = _loss_for(head, bq, tr_rows)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        from glance.metrics import accuracy, nll

        loss, p = _loss_for(head, torch.arange(len(dev)), dv_rows)
        ps = [p[i, :len(d.target)].cpu().numpy() for i, d in enumerate(dev)]
        devm = {"n": len(dev), "loss": float(loss), "nll": nll(ps, [d.target for d in dev]),
                "acc": accuracy(ps, [d.target for d in dev])}
        for bench, fn in R0_BENCHMARKS.items():
            items = fn()
            imgs = torch.from_numpy(np.stack([np.asarray(d.image().resize((256, 256), Image.Resampling.BICUBIC))
                                              for d in items]))
            iv = torch.cat([enc.images(imgs[s:s + 64]) for s in range(0, len(items), 64)])
            u = sorted({s for d in items for s in texts_for(d)})
            te = enc.texts(u)
            rows = _rows(items, iv, {s: i for i, s in enumerate(u)}, te, enc)
            _, pr = _loss_for(head, torch.arange(len(items)), rows)
            logp = torch.log(pr.clamp_min(1e-12)).cpu()
            with open(out / "eval" / f"C0d__{bench}.jsonl", "w") as fh:
                for i, d in enumerate(items):
                    rec = {"uid": d.uid, "image_id": d.image_id, "split": calib_split(d.image_id), "kind": d.kind,
                           "k": len(d.candidates), "target": d.target, "logits": logp[i, :len(d.candidates)].tolist()}
                    if "mos" in d.meta:
                        rec["mos"] = d.meta["mos"]
                    fh.write(json.dumps(rec) + "\n")
            print(f"[C0d] eval {bench}: {len(items)}", flush=True)
    (out / "train_summary.json").write_text(json.dumps({
        "dev": devm, "steps": steps, "epochs": a.epochs, "train_flops": 0.0, "cache_flops_one_time": 0.0,
        "wall_s": time.time() - t0, "note": "frozen SigLIP2-B/32 towers, cached embeddings; MLP head only"}, indent=1))
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    print(json.dumps(devm), flush=True)


if __name__ == "__main__":
    main()
