"""Export a trained Glance run as a self-contained release package.

The package loads with no Hub download (glance.serve.Predictor accepts the directory):
  model.safetensors   every weight (frozen SigLIP2 image encoder, Ettin text encoder, fusion layers, head)
  vision_config/, text_config/, tokenizer/   configs and tokenizer of the two base models
  calibration.json    Platt scaling for yes/no, temperatures for choice and score, fitted only on the
                      held-out calibration split of the licence-clean training mix (never trained on), with
                      logits from the served path (CPU, fp32)
  config.json         candidate, source run, base-model repos and pinned revisions
  attribution.jsonl   source, licence and author of every training image
  image.onnx, questions.onnx (+ .data)   the same model for ONNX Runtime (CPU; glance.serve runtime="onnx"),
                      when onnx/onnxruntime are installed; checked against PyTorch on calibration questions
It then reloads the package offline and checks it answers exactly as the trained model does.
--add-onnx PKG adds the ONNX files to an existing package.

    uv run python scripts/export_release.py --run results/r4/C5_open3_s0 --mix data/clean_art_v3 \
        --out data/release/glance-c5-open
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from glance.data.packing import collate, group_by_image
from glance.data.schema import Decision
from glance.metrics import fit_platt, fit_temperature, nll
from glance.model.heads import pad_by_question
from glance.serve import Predictor, calibrated_probs
from glance.train import R1Data

BASE = {"vision": "google/siglip2-base-patch32-256", "text": "jhu-clsp/ettin-encoder-32m"}


def revision(repo: str) -> str:
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else "unknown"


@torch.inference_mode()
def served_logits(pred: Predictor, decisions: list[Decision], data: R1Data) -> list[np.ndarray]:
    """Raw logits per decision, through the server's own image path and packing (CPU, fp32)."""
    by_row = defaultdict(list)
    for i, d in enumerate(decisions):
        by_row[d.meta["image_row"]].append(i)
    out = [None] * len(decisions)
    for n, (row, idx) in enumerate(sorted(by_row.items())):
        feats, _ = pred._image_input(data.image_bytes(row))
        pred.cache.clear()
        ds = [decisions[i] for i in idx]
        seqs = group_by_image(ds, pred.tok, 320, 4)
        batch = collate([es for _, es in seqs], pred.tok.pad_token_id)
        logits = pred.onnx.logits(feats, batch) if pred.onnx else pred.cand(feats.expand(len(seqs), *feats.shape[1:]), batch)
        padded = pad_by_question(logits.float(), batch["slot_q"], batch["slot_k"], len(batch["q_pos"]), batch["k_max"]).numpy()
        order = [d.uid for s, _ in seqs for d in s]
        pos = {u: j for j, u in enumerate(order)}
        for i in idx:
            out[i] = padded[pos[decisions[i].uid], :len(decisions[i].candidates)].astype(np.float64)
        if (n + 1) % 1000 == 0:
            print(f"  {n + 1}/{len(by_row)} images", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--mix", help="the licence-clean mix the run trained on (its calib.jsonl is used)")
    ap.add_argument("--out")
    ap.add_argument("--add-onnx", help="an existing package: add image.onnx / questions.onnx and check them")
    a = ap.parse_args()
    if a.add_onnx:
        return add_onnx(Path(a.add_onnx), Path(json.loads((Path(a.add_onnx) / "config.json").read_text())["mix"]))
    run, mix, out = Path(a.run), Path(a.mix), Path(a.out)
    cid = json.loads((run / "config.json").read_text())["cand"]
    pred = Predictor(f"data/ckpt/{run.name}.pt", cid, None, "cpu", 4)
    data = R1Data(mix)
    decisions = []
    for line in open(mix / "calib.jsonl"):
        r = json.loads(line)
        row = r.pop("image_row")
        d = Decision(**r, image_bytes=b"")
        d.meta = {**d.meta, "image_row": row}
        decisions.append(d)
    print(f"calibration split: {len(decisions)} questions", flush=True)
    t0 = time.time()
    logits = served_logits(pred, decisions, data)
    print(f"logits in {time.time() - t0:.0f} s", flush=True)

    calib, report = {}, {}
    for kind in ("noul", "choice", "score"):
        idx = [i for i, d in enumerate(decisions) if d.kind == kind]
        z, y = [logits[i] for i in idx], [decisions[i].target for i in idx]
        if kind == "noul":
            ca, cb = fit_platt(z, y)
            calib[kind] = {"a": ca, "b": cb, "n": len(idx)}
        else:
            calib[kind] = {"temperature": fit_temperature(z, y), "n": len(idx)}
        raw = [np.exp(v - v.max()) / np.exp(v - v.max()).sum() for v in z]
        cal = [calibrated_probs(calib, kind, v) for v in z]
        report[kind] = {"n": len(idx), "nll_raw": nll(raw, y), "nll_calibrated": nll(cal, y)}
        print(f"{kind}: {calib[kind]} NLL {report[kind]['nll_raw']:.4f} -> {report[kind]['nll_calibrated']:.4f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    state = {k: v.detach().float().contiguous().clone() for k, v in pred.cand.state_dict().items()}
    save_file(state, out / "model.safetensors")
    pred.cand.frozen_image.vision.config.save_pretrained(out / "vision_config")
    pred.cand.tower.text_encoder.config.save_pretrained(out / "text_config")
    pred.tok.save_pretrained(out / "tokenizer")
    (out / "calibration.json").write_text(json.dumps(
        {"calibration": calib, "fit": report, "fitted_on": f"{mix}/calib.jsonl (held out from training)",
         "logits": "served path, CPU fp32"}, indent=1))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "config.json").write_text(json.dumps(
        {"cand": cid, "run": str(run), "mix": str(mix), "code_commit": commit, "image_size": pred.cand.image_size,
         "base_models": {k: {"repo": v, "revision": revision(v)} for k, v in BASE.items()},
         "parameters": sum(p.numel() for p in pred.cand.parameters()), "license": "Apache-2.0"}, indent=1))
    shutil.copy(mix / "attribution.jsonl", out / "attribution.jsonl")
    print(f"wrote {out} ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 2**20:.0f} MB)", flush=True)

    # reload offline and compare with the trained model on a sample of calibration questions
    os.environ["HF_HUB_OFFLINE"] = "1"
    pkg = Predictor(str(out), device="cpu", threads=4, runtime="torch")
    sample = decisions[:300]
    again = served_logits(pkg, sample, data)
    diff = max(float(np.abs(x - y).max()) for x, y in zip(again, logits[:300]))
    print(f"package vs trained model, max logit difference on {len(sample)} questions: {diff:.2e}", flush=True)
    assert diff < 1e-4, diff
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("onnxruntime not installed: no ONNX files", flush=True)
        return
    add_onnx(out, mix)


def add_onnx(pkg: Path, mix: Path, n: int = 300):
    """Export image.onnx / questions.onnx into a package and check ONNX Runtime against PyTorch."""
    from glance.onnx_export import export

    torch_pred = Predictor(str(pkg), device="cpu", threads=4, runtime="torch")
    export(torch_pred.cand, torch_pred.tok, pkg)
    onnx_pred = Predictor(str(pkg), device="cpu", threads=4, runtime="onnx")
    data = R1Data(mix)
    decisions = []
    for line in open(mix / "calib.jsonl"):
        r = json.loads(line)
        row = r.pop("image_row")
        d = Decision(**r, image_bytes=b"")
        d.meta = {**d.meta, "image_row": row}
        decisions.append(d)
        if len(decisions) == n:
            break
    a, b = served_logits(torch_pred, decisions, data), served_logits(onnx_pred, decisions, data)
    diff = max(float(np.abs(x - y).max()) for x, y in zip(a, b))
    flips = np.mean([int(np.argmax(x) != np.argmax(y)) for x, y in zip(a, b)])
    print(f"ONNX vs PyTorch on {n} calibration questions: max logit difference {diff:.2e}, top-answer changes {flips:.3f}", flush=True)
    assert diff < 0.05 and flips < 0.01, (diff, flips)
    cfg = json.loads((pkg / "config.json").read_text())
    cfg["onnx"] = {"files": ["image.onnx", "image.onnx.data", "questions.onnx", "questions.onnx.data"],
                   "max_logit_diff_vs_torch": diff, "checked_questions": n}
    (pkg / "config.json").write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
