"""Local HTTP server for a trained Glance checkpoint, plus a browser playground.

Speaks the request/response shape of TypeSafe Jev's `POST /v1/systemone` (as Laya's `laya-serve`
does), extended with an image. Standard library only, so it runs in the project environment as is.

    uv run python -m glance.serve --ckpt data/ckpt/C5_ep2_lrA_s0.pt
    # then open http://127.0.0.1:8089

Request:
    {"image": "data:image/png;base64,...",          # optional; also accepted as state.image
     "state": "optional text context",                # or {"text": "...", "image": "..."}
     "questions": {
        "cat":     {"type": "noul",   "instructions": "Is there a cat in the image?"},
        "room":    {"type": "choice", "instructions": "Which room is this?",
                    "criteria": {"kitchen": "", "bedroom": "", "office": ""}},
        "quality": {"type": "score",  "instructions": "How good is the technical quality of this image?",
                    "criteria": ["bad", "poor", "fair", "good", "excellent"]}}}

Response: {"answers": {qid: {...}}, "usage": {...}, "timing_ms": {...}, "image_cached": bool}
    choice -> {"type": "choice", "choice": key, "probabilities": {key: p}, "confidence": p_max}
    noul   -> {"type": "noul", "noul": P(yes), "probabilities": {"yes": p, "no": p}, "confidence": p_max}
    score  -> {"type": "score", "score": expected level index, "legend": {"0": level, ...},
               "probabilities": {"0": p, ...}, "confidence": p_max}

Probabilities are calibrated when the checkpoint's evaluation files are found (temperature for
choice and score, Platt scaling for yes/no, fitted on the held-out calibration split), and raw
otherwise; /health reports which. Images are kept in a small in-memory cache keyed by their
bytes, so further questions about the same image skip the image encoder. Nothing is written to
disk. Binds to 127.0.0.1 by default; set GLANCE_API_KEY to require `Authorization: Bearer <key>`.
"""
import argparse
import base64
import binascii
import hashlib
import io
import json
import math
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

STATIC = Path(__file__).parent / "static"
MAX_BODY = 25 * 1024 * 1024
BLANK = (128, 128, 128)  # stand-in image for text-only requests


def _decode_image(value: str) -> bytes:
    """A data URL or bare base64 string -> image bytes (validated by PIL)."""
    if value.startswith("data:"):
        value = value.split(",", 1)[1] if "," in value else ""
    try:
        data = base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image is not valid base64: {e}") from None
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise ValueError("image could not be decoded (send PNG, JPEG or WebP as base64 or a data URL)") from None
    return data


def _candidates(qid: str, spec: dict) -> tuple[str, list[str], list[str]]:
    """Question spec -> (kind, answer keys, candidate texts shown to the model)."""
    kind = spec.get("type")
    crit = spec.get("criteria")
    if kind == "noul":
        return kind, ["yes", "no"], ["yes", "no"]
    if kind == "choice":
        if isinstance(crit, list):
            crit = {str(c): "" for c in crit}
        if not isinstance(crit, dict) or len(crit) < 2:
            raise ValueError(f"question {qid!r}: a choice question needs 'criteria' with at least two options")
        keys = [str(k) for k in crit]
        texts = [f"{k}: {v}" if v else k for k, v in ((str(k), str(v or "").strip()) for k, v in crit.items())]
        return kind, keys, texts
    if kind == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise ValueError(f"question {qid!r}: a score question needs 'criteria' as a list of 2-10 levels, lowest first")
        levels = [str(c) for c in crit]
        return kind, [str(i) for i in range(len(levels))], levels
    raise ValueError(f"question {qid!r}: 'type' must be one of choice, noul, score")


def calibrated_probs(calib: dict, kind: str, logits: np.ndarray) -> np.ndarray:
    """Logits of one question -> probabilities: Platt for yes/no, temperature otherwise, raw if uncalibrated."""
    c = calib.get(kind)
    if kind == "noul" and c:
        p_yes = 1 / (1 + math.exp(-(c["a"] * (logits[0] - logits[1]) + c["b"])))
        return np.array([p_yes, 1 - p_yes])
    t = c["temperature"] if c else 1.0
    z = np.asarray(logits, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return e / e.sum()


class Predictor:
    """One trained candidate, its tokenizer, calibration and an image-feature cache."""

    def __init__(self, ckpt: str, cand_id: str = "C5", calib_dir: str | None = None, device: str = "auto",
                 threads: int = 4, cache_size: int = 64, notes: str = "", runtime: str = "auto"):
        """runtime: "torch", "onnx" (a release package's image.onnx and questions.onnx through ONNX Runtime,
        CPU; the PyTorch weights are then not kept in memory) or "auto" (onnx when the package has the files and
        onnxruntime is installed, else torch)."""
        from transformers import AutoTokenizer

        from glance.model.candidates import build_candidate, load_trained, read_state

        torch.set_num_threads(threads)
        self.onnx = None
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.cand_id, self.ckpt, self.notes = cand_id, ckpt, notes
        if Path(ckpt).is_dir():  # a release package (scripts/export_release.py): every weight, configs, calibration
            from safetensors.torch import load_file

            pkg = Path(ckpt)
            cfg = json.loads((pkg / "config.json").read_text())
            self.cand_id = cand_id = cfg["cand"]
            self.tok = AutoTokenizer.from_pretrained(pkg / "tokenizer")
            self.calib = json.loads((pkg / "calibration.json").read_text())["calibration"]
            has_onnx = (pkg / "image.onnx").is_file() and (pkg / "questions.onnx").is_file()
            if runtime == "onnx" and not has_onnx:
                raise ValueError(f"{pkg} has no image.onnx / questions.onnx (scripts/export_release.py --add-onnx)")
            if has_onnx and runtime in ("onnx", "auto") and self.device.type == "cpu":
                try:
                    from glance.onnx_export import OnnxRuntime

                    self.onnx = OnnxRuntime(pkg, threads)
                except ImportError:
                    if runtime == "onnx":
                        raise
            if self.onnx:  # ONNX Runtime holds the weights: no PyTorch model is built or loaded
                cand = None
                self.image_size, self.cached_image = cfg["image_size"], True
            else:
                cand = build_candidate(cand_id, base={"vision": pkg / "vision_config", "text": pkg / "text_config"})
                cand.load_state_dict(load_file(pkg / "model.safetensors"), strict=True)
        else:
            cand = build_candidate(cand_id)
            load_trained(cand, read_state(ckpt), str(ckpt))
            self.tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m")
            self.calib = self._fit_calibration(calib_dir)
        if cand is not None:
            self.image_size, self.cached_image = cand.image_size, cand.cached_image
        self.cand = cand.to(self.device).eval() if cand is not None else None
        self.runtime = "onnx" if self.onnx else "torch"
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.cache_size = cache_size
        self.lock = threading.Lock()

    @staticmethod
    def _fit_calibration(calib_dir):
        """Temperature (choice, score) and Platt (noul) from the run's evaluation logits, calib split only."""
        from glance.metrics import fit_platt, fit_temperature

        if not calib_dir or not Path(calib_dir).is_dir():
            return {}
        recs = {"choice": [], "noul": [], "score": []}
        for f in Path(calib_dir).glob("*.jsonl"):
            for line in f.open():
                r = json.loads(line)
                if r.get("split") == "calib":
                    recs[r["kind"]].append((np.array(r["logits"], dtype=np.float64), r["target"]))
        calib = {}
        for kind in ("choice", "score"):
            if recs[kind]:
                calib[kind] = {"temperature": fit_temperature([l for l, _ in recs[kind]], [y for _, y in recs[kind]]),
                               "n": len(recs[kind])}
        if recs["noul"]:
            a, b = fit_platt([l for l, _ in recs["noul"]], [y for _, y in recs["noul"]])
            calib["noul"] = {"a": a, "b": b, "n": len(recs["noul"])}
        return calib

    def _probs(self, kind: str, logits: np.ndarray) -> np.ndarray:
        return calibrated_probs(self.calib, kind, logits)

    def _image_input(self, image_bytes: bytes | None) -> tuple[torch.Tensor, bool]:
        from glance.model.candidates import normalize

        key = hashlib.sha1(image_bytes).hexdigest() if image_bytes else "blank"
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key], True
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB") if image_bytes else Image.new("RGB", (256, 256), BLANK)
        size = self.image_size
        px = np.asarray(img.resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)
        pixels = normalize(torch.from_numpy(px[None])).to(self.device)
        with torch.inference_mode():
            # cached-image candidates were trained on fp16-stored features: round the same way
            image_features = self.onnx.image_features if self.onnx else self.cand.image_features
            feats = image_features(pixels).half().float() if self.cached_image else pixels
        self.cache[key] = feats
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return feats, False

    def predict(self, questions: dict, image_bytes: bytes | None = None, state_text: str = "") -> dict:
        from glance.data.packing import collate, group_by_image
        from glance.data.schema import Decision
        from glance.model.heads import pad_by_question

        if not isinstance(questions, dict) or not questions:
            raise ValueError("'questions' must be a non-empty object {id: {type, instructions, criteria}}")
        specs, decisions = {}, []
        for qid, spec in questions.items():
            if not isinstance(spec, dict):
                raise ValueError(f"question {qid!r} must be an object")
            text = str(spec.get("instructions") or spec.get("question") or "").strip()
            if not text:
                raise ValueError(f"question {qid!r}: 'instructions' (the question text) is empty")
            kind, keys, cands = _candidates(qid, spec)
            specs[qid] = (kind, keys, cands)
            question = f"{state_text.strip()}\n{text}" if state_text and state_text.strip() else text
            decisions.append(Decision(uid=str(qid), source="serve", kind=kind, question=question, candidates=cands,
                                      target=[1 / len(cands)] * len(cands), image_bytes=b"", image_id="request"))
        t0 = time.perf_counter()
        with self.lock, torch.inference_mode():
            feats, cached = self._image_input(image_bytes)
            t_img = time.perf_counter()
            seqs = group_by_image(decisions, self.tok, 320, 4)
            batch = collate([es for _, es in seqs], self.tok.pad_token_id)
            b = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if self.onnx:
                logits = self.onnx.logits(feats, b)
            else:
                image_input = feats.expand(len(seqs), *feats.shape[1:])
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "mps"):
                    logits = self.cand(image_input, b)
            padded = pad_by_question(logits.float(), b["slot_q"], b["slot_k"], len(b["q_pos"]), b["k_max"]).cpu().numpy()
            if self.device.type == "mps":
                torch.mps.synchronize()
        t1 = time.perf_counter()
        order = [d.uid for ds, _ in seqs for d in ds]
        answers = {}
        for row, qid in zip(padded, order):
            kind, keys, cands = specs[qid]
            p = self._probs(kind, row[:len(keys)].astype(np.float64))
            probs = {k: round(float(v), 4) for k, v in zip(keys, p)}
            if kind == "choice":
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())], "probabilities": probs}
            elif kind == "noul":
                answers[qid] = {"type": "noul", "noul": round(float(p[0]), 4), "probabilities": probs}
            else:
                answers[qid] = {"type": "score", "score": round(float((np.arange(len(p)) * p).sum()), 4),
                                "legend": {k: c for k, c in zip(keys, cands)}, "probabilities": probs}
            answers[qid]["confidence"] = round(float(p.max()), 4)
        answers = {qid: answers[qid] for qid in questions}  # request order
        n_tok = int((batch["block_ids"] >= 0).sum())
        return {"answers": answers, "usage": {"input_tokens": n_tok, "output_tokens": 0},
                "image_cached": cached,
                "timing_ms": {"total": round(1000 * (t1 - t0), 2), "image": round(1000 * (t_img - t0), 2),
                              "questions": round(1000 * (t1 - t_img), 2)}}

    def info(self) -> dict:
        return {"status": "ok", "model": f"glance-{self.cand_id}", "checkpoint": Path(self.ckpt).name,
                "device": self.device.type, "runtime": self.runtime, "calibrated": sorted(self.calib), "image_cache": len(self.cache),
                "notes": self.notes}


def make_handler(pred: Predictor, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "glance-serve"

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/health":
                self._json(200, pred.info())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._json(404, {"error": "not found"})
            if api_key and self.headers.get("Authorization") != f"Bearer {api_key}":
                return self._json(401, {"error": "missing or wrong bearer token"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._json(413, {"error": f"request larger than {MAX_BODY // 2**20} MB"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                state = body.get("state") or ""
                image = body.get("image")
                if isinstance(state, dict):  # {"text": ..., "image": ..., any other fields -> JSON text}
                    image = image or state.get("image")
                    rest = {k: v for k, v in state.items() if k not in ("image", "text")}
                    state = "\n".join(p for p in (state.get("text") or "", json.dumps(rest) if rest else "") if p)
                elif not isinstance(state, str):
                    state = json.dumps(state)
                image_bytes = _decode_image(image) if image else None
                self._json(200, pred.predict(body.get("questions"), image_bytes, state))
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})

        def log_message(self, fmt, *args):  # one short line per request, no bodies
            print(f"{self.address_string()} {self.command} {self.path} {args[1] if len(args) > 1 else ''}", flush=True)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="release package dir (scripts/export_release.py), or a trained checkpoint, e.g. data/ckpt/C5_ep2_lrA_s0.pt")
    ap.add_argument("--cand", default="C5", help="candidate design the checkpoint belongs to")
    ap.add_argument("--calib", default=None, help="evaluation dir for calibration (default: results/r1/<ckpt name>/eval)")
    ap.add_argument("--host", default=os.environ.get("GLANCE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GLANCE_PORT", 8089)))
    ap.add_argument("--device", default=os.environ.get("GLANCE_DEVICE", "cpu"),
                    help="cpu (default), mps or auto. On MPS the first request with a new question length takes 150-350 ms while a kernel compiles; the CPU answers in ~5 ms per cached-image question every time")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (4 is fastest for batch 1 on an M4 Pro)")
    ap.add_argument("--runtime", default="auto", choices=["auto", "torch", "onnx"],
                    help="auto: ONNX Runtime when the release package has image.onnx / questions.onnx and onnxruntime is installed (CPU)")
    ap.add_argument("--notes", default="", help="text file with this model's measured strengths and limits, shown on the page")
    a = ap.parse_args()
    calib = a.calib or f"results/r1/{Path(a.ckpt).stem}/eval"
    print(f"loading {a.cand} from {a.ckpt} ...", flush=True)
    pred = Predictor(a.ckpt, a.cand, calib, a.device, a.threads, notes=Path(a.notes).read_text().strip() if a.notes else "",
                     runtime=a.runtime)
    pred.predict({"warmup": {"type": "noul", "instructions": "Is there a cat in the image?"}})  # compile kernels
    info = pred.info()
    try:
        server = ThreadingHTTPServer((a.host, a.port), make_handler(pred, os.environ.get("GLANCE_API_KEY")))
    except OSError as e:
        raise SystemExit(f"cannot listen on {a.host}:{a.port} ({e.strerror}); pick another port with --port") from None
    print(f"ready: {info['model']} on {info['device']} ({info['runtime']}), calibrated: {', '.join(info['calibrated']) or 'no'}", flush=True)
    print(f"open http://{a.host}:{a.port}  (API: POST http://{a.host}:{a.port}/v1/systemone)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
