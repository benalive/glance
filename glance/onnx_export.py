"""ONNX export of a C5-family candidate for ONNX Runtime serving (glance.serve.Predictor, runtime="onnx").

  image.onnx      pixels (1, 3, S, S) -> image tokens (1, 64, 768): the frozen image path, fp32
  questions.onnx  image tokens + one packed request -> one logit per answer slot: text encoder, fusion
                  layers and pointer head, with dynamic batch, length, question and slot counts

On x86 CPUs the PyTorch question path is bound by per-operator overhead (about 70 small matrix multiplies
at ~30 us each per request); ONNX Runtime fuses and runs them with less overhead. The pointer head's
per-slot token count (a bincount in PyTorch) is passed in as an input.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class QuestionPath(nn.Module):
    """Candidate.forward with the candidate-token count passed in, for export."""

    def __init__(self, cand):
        super().__init__()
        self.cand = cand

    def forward(self, image_states, input_ids, position_ids, block_ids, cand_ids, q_pos, slot_q, cand_tok_pos,
                cand_tok_slot, cand_count):
        states = self.cand.text_states(image_states, input_ids, position_ids, block_ids, cand_ids)
        head = self.cand.head
        hq = head.q(states[q_pos[:, 0], q_pos[:, 1]])
        tok = states[cand_tok_pos[:, 0], cand_tok_pos[:, 1]]
        pooled = torch.zeros(cand_count.shape[0], states.shape[-1], dtype=tok.dtype).index_add(0, cand_tok_slot, tok)
        hc = head.c(pooled / cand_count[:, None].to(tok.dtype))
        return (hc * hq[slot_q]).sum(-1) * head.scale


def question_inputs(image_states: torch.Tensor, batch: dict) -> dict:
    """ONNX feed for one packed request (collate's output) and its image tokens (1, n, d)."""
    n_slots = len(batch["slot_pos"])
    count = np.bincount(batch["cand_tok_slot"].numpy(), minlength=n_slots).clip(min=1).astype(np.float32)
    seqs = batch["input_ids"].shape[0]
    return {"image_states": image_states.expand(seqs, *image_states.shape[1:]).contiguous().numpy().astype(np.float32),
            **{k: batch[k].numpy().astype(np.int64) for k in ("input_ids", "position_ids", "block_ids", "cand_ids", "q_pos",
                                                              "slot_q", "cand_tok_pos", "cand_tok_slot")},
            "cand_count": count}


def export(cand, tok, out_dir: Path) -> None:
    """Write image.onnx and questions.onnx for a C5-family candidate into out_dir."""
    from glance.data.packing import collate, group_by_image
    from glance.data.schema import Decision

    cand = cand.eval()
    out_dir = Path(out_dir)
    size = cand.image_size
    with torch.inference_mode(False), torch.no_grad():
        image = _ImagePath(cand)
        torch.onnx.export(image, (torch.randn(1, 3, size, size),), str(out_dir / "image.onnx"),
                          input_names=["pixels"], output_names=["image_states"], dynamo=True)
        ds = [Decision(uid=f"q{i}", source="x", kind="choice" if i % 2 else "noul", question=q, candidates=c,
                       target=[1.0 / len(c)] * len(c), image_bytes=b"", image_id="x")
              for i, (q, c) in enumerate([("Is there a dog?", ["yes", "no"]), ("Which animal is it?", ["cat", "dog", "a horse"]),
                                          ("Does this image contain visible generation errors?", ["yes", "no"]),
                                          ("How good is it?", ["bad", "poor", "fair", "good", "excellent"]),
                                          ("Is anyone smiling?", ["yes", "no"])])]
        seqs = group_by_image(ds, tok, 320, 4)
        batch = collate([es for _, es in seqs], tok.pad_token_id)
        feats = image(torch.randn(1, 3, size, size))
        feed = question_inputs(feats, batch)
        args = tuple(torch.from_numpy(feed[k]) for k in _Q_INPUTS)
        B, L, NQ, NS, NT = (torch.export.Dim(n, min=1, max=4096) for n in ("batch", "length", "questions", "slots", "cand_tokens"))
        dyn = {"image_states": {0: B}, "input_ids": {0: B, 1: L}, "position_ids": {0: B, 1: L}, "block_ids": {0: B, 1: L},
               "cand_ids": {0: B, 1: L}, "q_pos": {0: NQ}, "slot_q": {0: NS}, "cand_tok_pos": {0: NT},
               "cand_tok_slot": {0: NT}, "cand_count": {0: NS}}
        torch.onnx.export(QuestionPath(cand), args, str(out_dir / "questions.onnx"), input_names=list(_Q_INPUTS),
                          output_names=["logits"], dynamic_shapes=dyn, dynamo=True)


_Q_INPUTS = ("image_states", "input_ids", "position_ids", "block_ids", "cand_ids", "q_pos", "slot_q", "cand_tok_pos",
             "cand_tok_slot", "cand_count")


class _ImagePath(nn.Module):
    def __init__(self, cand):
        super().__init__()
        self.cand = cand

    def forward(self, pixels):
        f = self.cand.frozen_image
        feats = f.bottom(pixels) if getattr(f, "top_k", 0) else f(pixels)
        return self.cand.image_top(feats)


class OnnxRuntime:
    """ONNX Runtime sessions for a package's image.onnx and questions.onnx."""

    def __init__(self, package: Path, threads: int):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.image = ort.InferenceSession(str(Path(package) / "image.onnx"), opts, providers=["CPUExecutionProvider"])
        self.questions = ort.InferenceSession(str(Path(package) / "questions.onnx"), opts, providers=["CPUExecutionProvider"])

    def image_features(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.image.run(None, {"pixels": pixels.numpy().astype(np.float32)})[0])

    def logits(self, image_states: torch.Tensor, batch: dict) -> torch.Tensor:
        return torch.from_numpy(self.questions.run(None, question_inputs(image_states, batch))[0])
