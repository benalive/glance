"""R1 training and evaluation for one (candidate, FLOP budget, seed).

Budget accounting: the FLOPs of an optimisation step are 3x the forward FLOPs of everything that
runs in the step (forward + backward, the standard 6ND rule; FlopCounterMode cannot see backward
ops on MPS), counted on the first steps and fitted as a * text_tokens + b * sequences; training stops when the running total reaches the budget. For
cached candidates (C1, C2, C5) the one-time cost of encoding the training images with the frozen
image path is measured and reported separately, not charged to the budget.

Sanity attacks for the training-impl audit: --sanity overfit (one batch, 300 steps),
shuffle_labels (targets permuted among same-K questions), shuffle_images (images permuted within
the batch).
"""
import io
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from glance.bench.latency import count_flops
from glance.data.benchmarks import ALL_BENCHMARKS, R0_BENCHMARKS
from glance.data.packing import collate, group_by_image
from glance.data.schema import Decision, calib_split
from glance.model.candidates import Candidate, build_candidate, load_trained, normalize, read_state
from glance.model.heads import decision_loss, pad_by_question

DATA = Path("data/r1")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class TrainConfig:
    cand: str
    budget: float
    seed: int = 0
    batch: int = 16
    lr_new: float = 3e-4
    lr_inherited: float = 5e-5
    lr_image: float = 0.0  # trainable image-encoder layers (C5u<k>); 0 = lr_inherited
    adam_eps: float = 1e-6
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    max_len: int = 320
    max_q_per_seq: int = 4
    sanity: str | None = None
    out: str = ""
    amp: bool = True
    eval_benches: str = ""  # comma-separated subset for smoke runs; empty = all R0 benchmarks
    epochs: float = 0.0  # if > 0, stop after this many passes over the training sequences instead of `budget`
    save_ckpt: bool = True  # sweep runs skip it (disk: a 12x768 checkpoint is ~0.7 GB in fp32)
    data_root: str = "data/r1"  # a mix in the build_r1_data.py layout (decisions.jsonl, images.bin, ...)
    init_ckpt: str = ""  # start from these trained weights (trainable tensors only) instead of the base models
    cache_features: bool = True  # frozen-image candidates: precompute image features once (disk) or per batch
    distill_teacher: str = ""  # teacher_<run>.jsonl in data_root (scripts/teacher_label.py)
    distill_alpha: float = 0.0  # train targets = (1 - alpha) * gold + alpha * teacher; dev keeps gold


class R1Data:
    def __init__(self, root: Path = DATA, teacher: str = "", alpha: float = 0.0):
        self.root = Path(root)
        self.alpha, self.teacher = alpha, {}
        if teacher and alpha > 0:
            for line in open(self.root / teacher):
                r = json.loads(line)
                self.teacher[r["uid"]] = (r["p"], r.get("c"))  # "c": the options the teacher saw, when recorded
        self.records = [json.loads(l) for l in open(self.root / "decisions.jsonl")]
        store = self.root / "images_256.u8.npy"
        self.images = np.load(store, mmap_mode="r") if store.exists() else None  # optional 256 px store
        if (self.root / "images.bin").exists():  # one concatenated blob (build_r1_data.py layout)
            self.offsets = np.load(self.root / "image_offsets.npy")
            self.blob_path, self.index = self.root / "images.bin", None
        else:  # images.json: {"blobs": [path, ...], "rows": [[blob, offset, length], ...]}, images left in place
            idx = json.loads((self.root / "images.json").read_text())
            self.blobs, self.index = idx["blobs"], idx["rows"]
            self.offsets = np.array([0] + [r[1] + r[2] for r in self.index], dtype=np.int64)  # fingerprint only

    @property
    def n(self) -> int:
        return len(self.index) if self.index is not None else len(self.offsets) - 1

    def image_bytes(self, row: int) -> bytes:
        if self.index is None:
            path, off, length = self.blob_path, int(self.offsets[row]), int(self.offsets[row + 1] - self.offsets[row])
        else:
            b, off, length = self.index[row]
            path = self.blobs[b]
        with open(path, "rb") as fh:
            fh.seek(off)
            return fh.read(length) if length >= 0 else fh.read()

    def decisions(self, split: str = "train") -> list[Decision]:
        """Training decisions minus audited contamination (results/r1/exclude_image_rows.json) and
        items whose candidates are empty or duplicated (unanswerable as a K-way choice). `split`
        'dev' is a 2% image-grouped holdout used for hyperparameter selection, so benchmarks are
        never used to choose anything."""
        spec_path = self.root / "exclude_image_rows.json"
        if not spec_path.exists() and self.root.resolve() == DATA.resolve():
            spec_path = Path("results/r1/exclude_image_rows.json")  # audited rows of the R1 mix
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        excluded = {r for k, v in spec.items() if k != "source" for r in v}
        out = []
        for r in self.records:
            if r["image_row"] in excluded:
                continue
            cands = [c.strip() for c in r["candidates"]]
            if any(not c for c in cands) or len(set(cands)) < len(cands):
                continue
            if (split == "dev") != _is_dev(r["image_id"].split("#")[0]):  # a derived copy ("<id>#...") follows its photo
                continue
            r = dict(r)
            row = r.pop("image_row")
            t, seen = self.teacher.get(r["uid"], (None, None)) if split == "train" else (None, None)
            if seen is not None and seen != r["candidates"]:
                t = None  # the teacher answered a different set of options (e.g. a rebuilt mix)
            if t is not None and len(t) == len(r["target"]):
                r["target"] = [(1 - self.alpha) * g + self.alpha * q for g, q in zip(r["target"], t)]
                s = sum(r["target"])
                r["target"] = [x / s for x in r["target"]]
            d = Decision(**r, image_bytes=b"")
            d.meta = {**d.meta, "image_row": row}
            out.append(d)
        return out

    def pixels_u8(self, rows: list[int], size: int) -> torch.Tensor:
        if size == 256 and self.images is not None:
            return torch.from_numpy(np.stack([self.images[r] for r in rows]))
        arrs = [np.asarray(Image.open(io.BytesIO(self.image_bytes(r))).convert("RGB").resize(
            (size, size), Image.Resampling.BICUBIC), dtype=np.uint8) for r in rows]
        return torch.from_numpy(np.stack(arrs))


def image_batch(cand: Candidate, data: "R1Data", rows: list[int], cache) -> torch.Tensor:
    """What the candidate consumes for these image rows: cached features, frozen features computed now
    (rounded to fp16 like the cache), or pixels for candidates that train their image path. For a
    candidate whose image path trains at the top (C5u<k>), this is the frozen bottom's output; the
    training step applies cand.image_top to it."""
    if cache is not None:
        return torch.from_numpy(np.ascontiguousarray(cache[rows])).to(DEVICE).float()
    px = normalize(data.pixels_u8(rows, cand.image_size)).to(DEVICE)
    if cand.cached_image:
        with torch.no_grad():
            return cand.cache_features(px).half().float()
    return px


def _is_dev(image_id: str, frac: float = 0.02) -> bool:
    import hashlib

    return int(hashlib.sha1(f"dev:{image_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def build_feature_cache(cand: Candidate, data: R1Data, batch: int = 32) -> tuple[np.ndarray, float]:
    """Frozen image path over every training image -> (N, n_tok, d) fp16 tensor, total FLOPs.

    Written to a temp file and renamed only when complete, with a sidecar recording N and a hash
    of the image offsets; a cache that does not match is rebuilt (R1 audit F5: a partial or stale
    cache used to be reused silently)."""
    import hashlib
    import os

    path = data.root / f"cache_{cand.cache_key}.npy"
    meta_path = path.with_suffix(".json")
    n = data.n
    want = {"cache_key": cand.cache_key, "size": cand.image_size, "n": n,
            "offsets_sha1": hashlib.sha1(data.offsets.tobytes()).hexdigest()}
    per_image = count_flops(lambda: cand.cache_features(normalize(data.pixels_u8([0], cand.image_size)).to(DEVICE)))
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == want:
        # The sidecar names the images, not the weights that encoded them: recompute a few rows so a
        # cache from other image-path weights, code or model revision is rebuilt, not reused (audit S1;
        # fp16 and device noise is ~0.005 rms, a stale cache ~0.4).
        cache = np.load(path, mmap_mode="r")  # memory-mapped: rows are read per batch
        probe = sorted({0, n // 2, n - 1})
        fresh = cand.cache_features(normalize(data.pixels_u8(probe, cand.image_size)).to(DEVICE)).half().float().cpu().numpy()
        if float(np.sqrt(((cache[probe].astype(np.float32) - fresh) ** 2).mean())) < 0.02:
            return cache, per_image * n
        print(f"[{cand.cid}] {path.name} does not match the current image path; rebuilding", flush=True)
    tmp = path.with_name(path.stem + ".partial.npy")
    need = n * int(np.prod(cand.cache_features(normalize(data.pixels_u8([0], cand.image_size)).to(DEVICE)).shape[1:])) * 2
    free = shutil.disk_usage(data.root).free
    if free - need < 3 * 2**30:  # leave 3 GB for checkpoints, logs and other jobs (audit S2)
        raise RuntimeError(f"feature cache {path.name} needs {need / 2**30:.1f} GB, only {free / 2**30:.1f} GB free")
    feats = None
    for s in range(0, n, batch):
        rows = list(range(s, min(n, s + batch)))
        f = cand.cache_features(normalize(data.pixels_u8(rows, cand.image_size)).to(DEVICE)).half().cpu().numpy()
        if feats is None:
            feats = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, *f.shape[1:]))
        feats[s:s + len(rows)] = f
    feats.flush()
    del feats
    os.replace(tmp, path)
    meta_path.write_text(json.dumps(want))
    return np.load(path, mmap_mode="r"), per_image * n


def _to(batch: dict) -> dict:
    return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}


def _step_loss(cand, image_input, batch, amp):
    with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=amp):
        logits = cand(cand.image_top(image_input), batch)
    padded = pad_by_question(logits.float(), batch["slot_q"], batch["slot_k"], len(batch["q_pos"]), batch["k_max"])
    return decision_loss(padded, batch["targets"], batch["is_score"])


def _shuffle_labels(batch: dict, rng: random.Random) -> dict:
    t = batch["targets"].clone()
    widths = torch.bincount(batch["slot_q"], minlength=len(t))  # candidates per question
    for kk in widths.unique().tolist():
        idx = (widths == kk).nonzero().flatten().tolist()
        perm = idx[:]
        rng.shuffle(perm)
        t[idx] = batch["targets"][perm]
    return {**batch, "targets": t}


def train(cfg: TrainConfig) -> dict:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m")
    cand = build_candidate(cfg.cand)
    if cfg.init_ckpt:
        state = read_state(cfg.init_ckpt)
        load_trained(cand, state, cfg.init_ckpt)
        # a checkpoint that writes into frozen image-path tensors (e.g. C5u4 into C5) would train the
        # tower on features the shared cache does not hold (audit S1); serving such a load is fine
        frozen = {n for n, p in cand.named_parameters() if not p.requires_grad}
        if cand.cached_image and (hit := [k for k in state if k in frozen]):
            raise ValueError(f"{cfg.init_ckpt} overwrites {len(hit)} frozen image-path tensors (e.g. {hit[0]}); "
                             f"the {cand.cache_key} feature cache would not match")
    cand = cand.to(DEVICE)
    data = R1Data(Path(cfg.data_root), cfg.distill_teacher, cfg.distill_alpha)
    seqs = group_by_image(data.decisions(), tok, cfg.max_len, cfg.max_q_per_seq)
    rows = [ds[0].meta["image_row"] for ds, _ in seqs]
    cache, cache_flops = (None, 0.0)
    if cand.cached_image:
        cand.eval()
        if cfg.cache_features:
            cache, cache_flops = build_feature_cache(cand, data)

    inh, new = cand.param_groups()
    image_ids = {id(p) for n, p in cand.named_parameters() if n.startswith("frozen_image.") and p.requires_grad}
    img, inh = [p for p in inh if id(p) in image_ids], [p for p in inh if id(p) not in image_ids]
    decay = lambda ps: [p for p in ps if p.ndim >= 2]  # noqa: E731
    no_decay = lambda ps: [p for p in ps if p.ndim < 2]  # noqa: E731
    groups = [(inh, cfg.lr_inherited), (new, cfg.lr_new), (img, cfg.lr_image or cfg.lr_inherited)]
    opt = torch.optim.AdamW([g for ps, lr in groups if ps for g in (
        {"params": decay(ps), "lr": lr, "weight_decay": cfg.weight_decay},
        {"params": no_decay(ps), "lr": lr, "weight_decay": 0.0})], betas=(0.9, 0.98), eps=cfg.adam_eps)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def image_input(idx):
        r = [rows[i] for i in idx]
        return image_batch(cand, data, r, cache)

    order = list(range(len(seqs)))
    ptr, epoch = len(order), -1

    def next_batch():
        nonlocal ptr, epoch
        if ptr + cfg.batch > len(order):
            epoch += 1
            rng.shuffle(order)
            ptr = 0
        idx = order[ptr:ptr + cfg.batch]
        ptr += cfg.batch
        return idx

    # FLOPs per step ~ a * text_tokens + b * sequences, fitted on the first 5 measured steps.
    cand.train()
    measured, fit, total_steps, nonfinite = [], None, None, 0
    consumed, step, t0 = 0.0, 0, time.time()
    fixed = next_batch() if cfg.sanity == "overfit" else None
    log = open(out / "train_log.jsonl", "w")
    epoch_steps = math.ceil(cfg.epochs * len(seqs) / cfg.batch) if cfg.epochs > 0 else None
    if epoch_steps:
        total_steps = epoch_steps  # exact schedule when matching on data rather than FLOPs

    def more() -> bool:
        if cfg.sanity == "overfit":
            return step < 300
        return step < epoch_steps if epoch_steps else consumed < cfg.budget

    while more():
        idx = fixed or next_batch()
        batch = _to(collate([seqs[i][1] for i in idx], tok.pad_token_id))
        img = image_input(idx)
        if cfg.sanity == "shuffle_labels":
            batch = _shuffle_labels(batch, rng)
        if cfg.sanity == "shuffle_images":
            img = img.roll(1, dims=0)  # a derangement: no sequence keeps its own image
        n_tok = int((batch["block_ids"] >= 0).sum())
        if len(measured) < 5:
            # FlopCounterMode has no formula for aten.linear_backward on MPS, so it counts ~1/3 of
            # a training step there (R1 audit F3); a step is 3x its counted forward (6ND rule).
            f = 3 * count_flops(lambda: _step_loss(cand, img, batch, cfg.amp))
            measured.append((n_tok, len(idx), f))
            step_flops = f
            if total_steps is None and cfg.sanity != "overfit":
                total_steps = max(1, int(cfg.budget / f))  # refined after 5 measured steps (FLOP mode)
            if len(measured) == 5:
                A = np.array([[m[0], m[1]] for m in measured], dtype=np.float64)
                fit = np.clip(np.linalg.lstsq(A, np.array([m[2] for m in measured]), rcond=None)[0], 0, None)
                mean_step = float(np.mean([m[2] for m in measured]))
                if not epoch_steps:
                    total_steps = max(1, int(cfg.budget / mean_step)) if cfg.sanity != "overfit" else 300
        else:
            step_flops = float(fit[0] * n_tok + fit[1] * len(idx))
        # schedule: linear warmup then cosine to 10% (needs total_steps; before the fit use warmup only)
        ts = total_steps or 10_000
        warm = max(1, int(cfg.warmup_frac * ts))
        scale = (step + 1) / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, ts - warm))))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * scale
        opt.zero_grad(set_to_none=True)
        loss, _ = _step_loss(cand, img, batch, cfg.amp)
        if not torch.isfinite(loss):
            nonfinite += 1
            consumed += step_flops
            step += 1
            continue
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in cand.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(gnorm):
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        consumed += step_flops
        step += 1
        if step % 25 == 0 or step == 1:
            rec = {"step": step, "epoch": epoch, "loss": float(loss), "gnorm": float(gnorm), "lr_scale": scale,
                   "flops": consumed, "seq_per_s": step * cfg.batch / (time.time() - t0)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 250 == 0 or step == 1:
                print(f"[{cfg.cand}] step {step}/{total_steps} loss {float(loss):.4f} flops {consumed:.2e} "
                      f"{rec['seq_per_s']:.1f} seq/s", flush=True)
    log.close()
    dev = dev_metrics(cand, tok, data, cfg, cache)
    summary = {"dev": dev, "nonfinite_steps": nonfinite, "steps": step, "epochs": step * cfg.batch / len(seqs), "train_flops": consumed,
               "cache_flops_one_time": cache_flops, "flop_fit_per_token_per_seq": None if fit is None else fit.tolist(),
               "wall_s": time.time() - t0, "sequences": len(seqs), "final_loss": float(loss)}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    evaluate(cand, tok, out / "eval", cfg, cfg.eval_benches.split(",") if cfg.eval_benches else None)
    if cfg.save_ckpt:  # after eval, so a failed save (e.g. a full disk) cannot cost the results
        try:
            ckpt = Path("data/ckpt") / out.name
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({n: p.detach().half().cpu() for n, p in cand.named_parameters() if p.requires_grad}, f"{ckpt}.pt")
        except OSError as e:
            print(f"[{cfg.cand}] checkpoint not saved: {e}", flush=True)
    return summary


@torch.inference_mode()
def dev_metrics(cand, tok, data: R1Data, cfg: TrainConfig, cache, batch: int = 32) -> dict:
    """Loss, NLL and accuracy on the 2% dev holdout (the only numbers used for selection)."""
    from glance.metrics import accuracy, nll

    cand.eval()
    seqs = group_by_image(data.decisions("dev"), tok, cfg.max_len, cfg.max_q_per_seq)
    ps, ys, losses = [], [], []
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        r = [ds[0].meta["image_row"] for ds, _ in chunk]
        img = image_batch(cand, data, r, cache)
        b = _to(collate([es for _, es in chunk], tok.pad_token_id))
        loss, p = _step_loss(cand, img, b, cfg.amp)
        losses.append(float(loss) * len(b["q_pos"]))
        for qi in range(len(b["q_pos"])):
            k = int((b["slot_q"] == qi).sum())
            ps.append(p[qi, :k].cpu().numpy())
            ys.append(b["targets"][qi, :k].cpu().tolist())
    cand.train()
    return {"n": len(ps), "loss": sum(losses) / len(ps), "nll": nll(ps, ys), "acc": accuracy(ps, ys)}


@torch.inference_mode()
def evaluate(cand: Candidate, tok, out_dir: Path, cfg: TrainConfig, benches=None, batch: int = 32):
    cand.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench in benches or list(R0_BENCHMARKS):
        decisions = ALL_BENCHMARKS[bench]()
        seqs = group_by_image(decisions, tok, cfg.max_len, cfg.max_q_per_seq)
        with open(out_dir / f"{cfg.cand}__{bench}.jsonl", "w") as fh:
            for s in range(0, len(seqs), batch):
                chunk = seqs[s:s + batch]
                imgs = torch.from_numpy(np.stack([
                    np.asarray(ds[0].image().resize((cand.image_size, cand.image_size), Image.Resampling.BICUBIC), dtype=np.uint8)
                    for ds, _ in chunk]))
                pixels = normalize(imgs).to(DEVICE)
                # cached candidates train on fp16-stored features; round the same way here
                image_input = cand.image_features(pixels).half().float() if cand.cached_image else pixels
                b = _to(collate([es for _, es in chunk], tok.pad_token_id))
                with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=cfg.amp):
                    logits = cand(image_input, b)
                padded = pad_by_question(logits.float(), b["slot_q"], b["slot_k"], len(b["q_pos"]), b["k_max"]).cpu()
                qi = 0
                for ds, _ in chunk:
                    for d in ds:
                        k = len(d.candidates)
                        rec = {"uid": d.uid, "image_id": d.image_id, "split": calib_split(d.image_id), "kind": d.kind,
                               "k": k, "target": d.target, "logits": padded[qi, :k].tolist()}
                        if "mos" in d.meta:
                            rec["mos"] = d.meta["mos"]
                        fh.write(json.dumps(rec) + "\n")
                        qi += 1
        print(f"[{cfg.cand}] eval {bench}: {len(decisions)} items", flush=True)
