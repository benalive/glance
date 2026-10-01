"""Snake-tuned Glance: learn to play from the planner, then play the article-style CPU benchmark.

Same game and teacher as scripts/snake_cpu_bench.py (laya_mlx engine, LayaStudio planner with 20%
exploration). The board is drawn grid-aligned: 16 px per cell, the 12x8 board in the top-left of a
256x256 canvas, so each 32 px patch covers exactly 2x2 cells. Two input variants:
  img        the board image + the question (a pure vision decision);
  img+facts  the image + Laya's two fact lines (head/food positions, what is next to the head) in
             the question; Laya additionally reads the ASCII board, which the image replaces.
The four candidates are Laya's options ("UP: one cell up", ...). Loss: our decision loss (Brier +
0.3 CE) against the planner's move.
  text       exactly the text state Laya and Jev read (ASCII board + facts) and a blank image: the
             apples-to-apples input for comparing the three systems;
  img+aux    the image only, trained with 8 extra yes/no perception questions packed into the same
             sequence ("is the cell above the head free?", "is the food left of the head?", ...);
             at inference only the move question is asked.

    uv run python -m glance.snake train --cand C6 --variant img --steps 4000
    uv run python -m glance.snake play --run results/snake/C6_img
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from scripts.snake_cpu_bench import QUESTIONS, new_game, protocol_a, protocol_b, render, teacher

CELL, CANVAS = 16, 256
COLORS = {"outside": (70, 70, 80), "empty": (15, 15, 20), "body": (40, 180, 60), "head": (220, 40, 40),
          "food": (240, 210, 40)}
DIRS = list(QUESTIONS["move"]["criteria"])


def board_image(head, body, food, width=12, height=8) -> np.ndarray:
    img = np.empty((CANVAS, CANVAS, 3), dtype=np.uint8)
    img[:] = COLORS["outside"]
    img[: height * CELL, : width * CELL] = COLORS["empty"]

    def paint(cell, color):
        x, y = cell
        img[y * CELL:(y + 1) * CELL, x * CELL:(x + 1) * CELL] = color

    for c in body:
        paint(c, COLORS["body"])
    paint(food, COLORS["food"])
    paint(head, COLORS["head"])
    return img


_BLANK = np.full((CANVAS, CANVAS, 3), COLORS["outside"], dtype=np.uint8)


def facts(game) -> str:
    return "\n".join(render(game).split("\n")[:2])  # "Snake 12x8. Head ... Length n." + "Next to the head: ..."


def question_text(variant: str, context: str) -> str:
    """context = the fact lines (img+facts) or the full text state (text); unused otherwise."""
    q = QUESTIONS["move"]["instructions"]
    return f"{context}\n{q}" if variant in ("img+facts", "text") else q  # img and img+aux: no text


def context_of(variant: str, row_or_game) -> str:
    if isinstance(row_or_game, dict):
        return row_or_game["text"] if variant == "text" else row_or_game["facts"]
    return render(row_or_game) if variant == "text" else facts(row_or_game)


def image_of(variant: str, head, body, food) -> np.ndarray:
    """The board image, or a constant blank canvas for the text-only variant."""
    if variant == "text":
        return _BLANK
    return board_image(head, body, food)


def generate(n: int, seed: int = 11, explore: float = 0.2, max_ticks: int = 400) -> list[dict]:
    """Planner games with exploration (as LayaStudio's dataset), recorded as compact states."""
    rng, out, g = random.Random(seed), [], 0
    while len(out) < n:
        game, g = new_game(seed + g), g + 1
        for _ in range(max_ticks):
            if not game.alive or game.won:
                break
            best = teacher(game)
            if best is None:
                break
            legal = {m.direction: m.legal for m in game.moves()}
            (hx, hy), (fx, fy) = game.head, game.food
            out.append({"head": game.head, "body": list(game.body), "food": game.food, "facts": facts(game), "text": render(game),
                        "label": DIRS.index(best), "free": [legal[d] for d in DIRS],
                        "food_dir": [fy < hy, fy > hy, fx < hx, fx > hx]})
            options = [d for d, ok in legal.items() if ok]
            game.step(rng.choice(options) if options and rng.random() < explore else best)
            if len(out) >= n:
                break
    return out


class Encoder:
    """Question/candidate encoding for a candidate family; the question is cached when constant."""

    def __init__(self, variant):
        from transformers import AutoTokenizer

        from glance.data.schema import Decision

        self.tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m")
        self.variant = variant
        self.cands = [f"{k}: {v}" for k, v in QUESTIONS["move"]["criteria"].items()]
        self.Decision = Decision
        self._cache = {}

    AUX = ([f"Is the cell {w} the head free?" for w in ("above", "below", "left of", "right of")]
           + [f"Is the food {w} the head?" for w in ("above", "below", "left of", "right of")])

    def aux_batch(self, rows: list[dict]):
        """Move question + 8 yes/no perception questions about the same board, packed in one
        sequence (questions isolated by the mask). Used for training the img+aux variant."""
        from glance.data.packing import collate, encode_question

        if not hasattr(self, "_aux"):
            yn = lambda q: encode_question(self.tok, self.Decision("a", "snake", "noul", q, ["yes", "no"], [0.5, 0.5], b"", "s"))  # noqa: E731
            self._aux = [yn(q) for q in self.AUX]
            self._move = encode_question(self.tok, self.Decision("s", "snake", "choice", QUESTIONS["move"]["instructions"],
                                                                 self.cands, [0.25] * 4, b"", "s"))
        seqs = []
        for r in rows:
            move = type(self._move)(**{**self._move.__dict__, "target": [float(j == r["label"]) for j in range(4)]})
            aux = [type(e)(**{**e.__dict__, "target": [1.0, 0.0] if flag else [0.0, 1.0]})
                   for e, flag in zip(self._aux, r["free"] + r["food_dir"])]
            seqs.append([move] + aux)
        return collate(seqs, self.tok.pad_token_id)

    def batch(self, fact_list: list[str], labels: list[int] | None = None):
        from glance.data.packing import collate, encode_question

        seqs = []
        for i, f in enumerate(fact_list):
            q = question_text(self.variant, f)
            if q not in self._cache:
                t = [0.25] * 4
                self._cache[q] = encode_question(self.tok, self.Decision("s", "snake", "choice", q, self.cands, t, b"", "s"),
                                                 max_q=256)
            e = self._cache[q]
            if labels is not None:
                e = type(e)(**{**e.__dict__, "target": [float(j == labels[i]) for j in range(4)]})
            seqs.append([e])
        if len(self._cache) > 50_000:
            self._cache.clear()
        return collate(seqs, self.tok.pad_token_id)


def _forward(cand, pixels_u8, batch, device):
    from glance.model.candidates import normalize

    px = normalize(pixels_u8).to(device)
    # the trainable top of the image path (C5u<k>) must see gradients: not image_features (no_grad)
    image_input = cand.image_top(cand.cache_features(px).half().float()) if cand.cached_image else px
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    return cand(image_input, b), b


def train(cid: str, variant: str, steps: int, batch: int = 64, n_boards: int = 100_000, lr_new: float = 1e-3,
          lr_inh: float = 2e-4, seed: int = 0, out: Path | None = None):
    from glance.model.candidates import build_candidate
    from glance.model.heads import decision_loss, pad_by_question

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    out = out or Path(f"results/snake/{cid}_{variant.replace('+', '_')}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = generate(n_boards, seed=11)
    val = generate(2000, seed=5011)  # disjoint seeds, as LayaStudio's test split
    enc = Encoder(variant)
    cand = build_candidate(cid).to(device)
    inh, new = cand.param_groups()
    opt = torch.optim.AdamW([{"params": inh, "lr": lr_inh}, {"params": new, "lr": lr_new}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_inh, lr_new], total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed)
    log = []
    cand.train()
    for step in range(steps):
        idx = rng.integers(0, len(data), batch)
        rows = [data[i] for i in idx]
        px = torch.from_numpy(np.stack([image_of(variant, r["head"], r["body"], r["food"]) for r in rows]))
        batch_in = enc.aux_batch(rows) if variant == "img+aux" else enc.batch([context_of(variant, r) for r in rows],
                                                                                   [r["label"] for r in rows])
        logits, b = _forward(cand, px, batch_in, device)
        padded = pad_by_question(logits.float(), b["slot_q"], b["slot_k"], len(b["q_pos"]), b["k_max"])
        loss, _ = decision_loss(padded, b["targets"], b["is_score"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in cand.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            log.append({"step": step, "loss": float(loss)})
            print(f"[{cid} {variant}] step {step}/{steps} loss {float(loss):.4f}", flush=True)
    cand.eval()
    with torch.no_grad():
        correct = 0
        for s in range(0, len(val), 128):
            rows = val[s:s + 128]
            px = torch.from_numpy(np.stack([image_of(variant, r["head"], r["body"], r["food"]) for r in rows]))
            logits, b = _forward(cand, px, enc.batch([context_of(variant, r) for r in rows]), device)
            pred = pad_by_question(logits.float(), b["slot_q"], b["slot_k"], len(b["q_pos"]), b["k_max"]).argmax(-1)
            correct += int((pred.cpu() == torch.tensor([r["label"] for r in rows])).sum())
    summary = {"cand": cid, "variant": variant, "steps": steps, "batch": batch, "boards_generated": n_boards,
               "boards_seen": steps * batch, "readout": "mean-pooled", "val_move_accuracy": correct / len(val), "wall_s": time.time() - t0,
               "log": log}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    torch.save({n: p.detach().cpu() for n, p in cand.named_parameters() if p.requires_grad}, out / "model.pt")
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}), flush=True)
    return summary


def load_player(run: Path, threads: int = 4):
    """The trained model on CPU fp32, as a decide(game) -> direction callable (render + forward)."""
    from glance.model.candidates import build_candidate

    torch.set_num_threads(threads)
    s = json.loads((run / "train_summary.json").read_text())
    cand = build_candidate(s["cand"])
    cand.load_state_dict(torch.load(run / "model.pt", weights_only=True), strict=False)
    cand.eval()
    enc = Encoder(s["variant"])
    cpu = torch.device("cpu")

    variant = s["variant"]
    blank_input = None
    if variant == "text" and cand.cached_image:  # constant blank image: encode it once, as a server would
        with torch.inference_mode():
            from glance.model.candidates import normalize

            blank_input = cand.image_features(normalize(torch.from_numpy(_BLANK)[None])).float()

    @torch.inference_mode()
    def decide(game):
        batch = enc.batch([context_of(variant, game)])
        if blank_input is not None:
            b = {k: v for k, v in batch.items()}
            logits = cand(blank_input, b)
        else:
            px = torch.from_numpy(image_of(variant, game.head, list(game.body), game.food))[None]
            logits, _ = _forward(cand, px, batch, cpu)
        return DIRS[int(logits.argmax())]

    for _ in range(5):
        decide(new_game(1))
    return decide, s


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--cand", required=True)
    t.add_argument("--variant", default="img", choices=["img", "img+facts", "img+aux", "text"])
    t.add_argument("--tag", default="", help="suffix for the run directory")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--boards", type=int, default=100_000)
    p = sub.add_parser("play")
    p.add_argument("--run", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results/bench/snake_cpu.jsonl")
    a = ap.parse_args()
    if a.cmd == "train":
        out = Path(f"results/snake/{a.cand}_{a.variant.replace('+', '_')}{'_' + a.tag if a.tag else ''}")
        train(a.cand, a.variant, a.steps, a.batch, a.boards, out=out)
        return
    torch.set_num_interop_threads(1)
    decide, s = load_player(Path(a.run), a.threads)
    rec = {"runtime": "glance-snake", "model": f"{s['cand']} {s['variant']} (Snake-tuned)", "run": Path(a.run).name,
           "boards_generated": s["boards_generated"], "threads": a.threads,
           "device": "cpu", "chip": "Apple M4 Pro", "val_move_accuracy": s["val_move_accuracy"],
           "A_30s": protocol_a(decide), "B_layastudio": protocol_b(decide)}
    print(json.dumps(rec, indent=1), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
