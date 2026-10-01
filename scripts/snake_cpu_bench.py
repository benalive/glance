"""CPU-only Snake benchmark, modelled on the Sina Tech article (2026-09-23) and LayaStudio.

Game and inputs match the community benchmark exactly: the laya_mlx.snake engine (12x8 board,
initial length 4); the board text, questions and planner teacher from LayaStudio
`layastudio/snake.py` (Apache-2.0, github.com/biplovgautam/LayaStudio), reproduced below; the
model's top-1 move is executed unassisted, so an illegal move ends the game.

Protocol A, the article's "30 seconds": play continuously for 30 s of wall clock, starting the
next seeded game whenever the snake dies or reaches 500 ticks. Reports decisions, decisions/s,
p50/p95 per-decision latency, food eaten, best length and deaths.
Protocol B, LayaStudio's bench: 10 games (seeds 101..110) x up to 500 ticks. Reports moves, score,
legal-move rate, teacher agreement and median ms/decision.

Runtimes, all on CPU:
  laya-torch  the upstream PyTorch `laya` package, fp32 (as shipped)
  laya-mlx    the laya-mlx port forced onto MLX's CPU backend (the article's runtime, minus the GPU)
  glance      our candidates, torch fp32, random weights, the board rendered as a 256x256 image:
              timed on the planner's trajectory, latency only (untrained, so no score)
  jev         TypeSafe Jev over HTTPS (same text state and questions as Laya); latency is the full
              round trip from this Mac. Needs TYPESAFE_API_KEY or ~/.config/typesafe/api_key.

    uv run --extra bench python scripts/snake_cpu_bench.py --runtime laya-torch \
        --model madhavbiplov/laya-snake-mlx --threads 10 --protocol both
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

WIDTH, HEIGHT, INITIAL_LENGTH = 12, 8, 4
QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": (
            "Snake board. Pick the next move: stay inside the board, do not hit the snake, "
            "and take the shortest safe path to the food."
        ),
        "criteria": {"UP": "one cell up", "DOWN": "one cell down", "LEFT": "one cell left", "RIGHT": "one cell right"},
    }
}
ARTICLE_JEV = {"runtime": "cloud API (article, not measured)", "model": "jev-1.13.0", "decisions_per_s": 3.2,
               "p50_ms": 317, "food_30s": 1, "length": 7}
ARTICLE_LAYA_MLX_M3MAX = {"runtime": "laya-mlx on M3 Max GPU (article)", "decisions_per_s": 86.5, "p50_ms": 9,
                          "food_30s": 46, "length": 52}


# --- reproduced from LayaStudio layastudio/snake.py (Apache-2.0) ---------------------------------
def render(game):
    head, body, food = game.head, set(game.body), game.food
    rows = ["".join("H" if (x, y) == head else "F" if (x, y) == food else "o" if (x, y) in body else "."
                    for x in range(game.width)) for y in range(game.height)]
    neighbours = ", ".join(f"{m.direction} {'free' if m.legal else m.reason}" for m in game.moves())
    dx, dy = food[0] - head[0], food[1] - head[1]
    where = " and ".join(p for p in (f"{abs(dy)} {'up' if dy < 0 else 'down'}" if dy else "",
                                     f"{abs(dx)} {'left' if dx < 0 else 'right'}" if dx else "") if p)
    return (f"Snake {game.width}x{game.height}. Head {head}. Food {food} ({where or 'here'}). "
            f"Length {len(game.body)}.\n"
            f"Next to the head: {neighbours}.\n"
            "Board, top row first: . empty, H head, o body, F food.\n" + "\n".join(rows))


def free_space(game, direction):
    target = game.target(direction)
    body = list(game.body)
    blocked = set(body[:-1]) | {target}
    stack, seen = [target], {target}
    while stack:
        x, y = stack.pop()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            cell = (x + dx, y + dy)
            if 0 <= cell[0] < game.width and 0 <= cell[1] < game.height and cell not in blocked and cell not in seen:
                seen.add(cell)
                stack.append(cell)
    return len(seen)


def teacher(game):
    legal = [m for m in game.moves() if m.legal]
    if not legal:
        return None
    food, length = game.food, len(game.body)
    scored = []
    for move in legal:
        target = game.target(move.direction)
        room = free_space(game, move.direction)
        scored.append((room >= length, room, -(abs(target[0] - food[0]) + abs(target[1] - food[1])), move.direction))
    scored.sort(reverse=True)
    return scored[0][3]
# -------------------------------------------------------------------------------------------------


def new_game(seed):
    from laya_mlx.snake.game import SnakeGame

    return SnakeGame(WIDTH, HEIGHT, seed=seed, initial_length=INITIAL_LENGTH)


def board_image(game, size=256):
    """The same board as pixels for image-input models: head red, body green, food yellow."""
    from PIL import Image

    grid = np.zeros((game.height, game.width, 3), dtype=np.uint8) + 30
    for x, y in game.body:
        grid[y, x] = (40, 180, 60)
    grid[game.head[1], game.head[0]] = (220, 40, 40)
    grid[game.food[1], game.food[0]] = (240, 210, 40)
    return Image.fromarray(grid).resize((size, size), Image.Resampling.NEAREST)


JEV_URL = "https://api.typesafe.ai/v1/systemone"  # as called in Laya's feishu_zh harness
JEV_USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def jev_key() -> str:
    """TYPESAFE_API_KEY, or the first line of ~/.config/typesafe/api_key (never printed or logged)."""
    key = os.environ.get("TYPESAFE_API_KEY")
    path = Path.home() / ".config" / "typesafe" / "api_key"
    if not key and path.exists():
        key = path.read_text().strip().splitlines()[0]
    if not key:
        raise SystemExit("No Jev key: set TYPESAFE_API_KEY or write it to ~/.config/typesafe/api_key")
    return key.strip()


def jev_decider(model_id):
    """Jev over HTTPS from this Mac: same text state and question as Laya; the measured latency is
    the full round trip, which is what an agent calling the API experiences."""
    import httpx

    from laya_mlx.snake.game import DIRECTIONS

    client = httpx.Client(timeout=60, headers={"Authorization": "Bearer " + jev_key()})

    def decide(game):
        r = client.post(JEV_URL, json={"model": model_id, "state": render(game), "questions": QUESTIONS})
        r.raise_for_status()
        body = r.json()
        usage = body.get("usage", {})
        JEV_USAGE["calls"] += 1
        JEV_USAGE["input_tokens"] += usage.get("input_tokens", 0)
        JEV_USAGE["output_tokens"] += usage.get("output_tokens", 0)
        choice = body["answers"]["move"].get("choice")
        return choice if choice in DIRECTIONS else "UP"

    decide(new_game(1))  # warm the connection (TLS) once, as the local runtimes are warmed
    return decide


def load_decider(runtime, model):
    """-> decide(game) returning a direction, for the Laya runtimes."""
    from laya_mlx.snake.game import DIRECTIONS

    if runtime == "laya-torch":
        import laya

        agent = laya.load(model, device="cpu")
    elif runtime == "laya-mlx":
        import laya_mlx
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        agent = laya_mlx.load(model)
    elif runtime == "teacher":
        return lambda game: teacher(game) or "UP"
    elif runtime == "jev":
        return jev_decider(model or "jev-1.13.0")
    else:
        raise ValueError(runtime)

    def decide(game):
        choice = agent.predict(render(game), QUESTIONS)["answers"]["move"]["choice"]
        return choice if choice in DIRECTIONS else "UP"

    for _ in range(3):  # warm up, as LayaStudio does
        decide(new_game(1))
    return decide


def protocol_a(decide, seconds=30.0, max_ticks=500, seed=101):
    """Play continuously for `seconds`; a new seeded game on every death or at max_ticks."""
    lat, food, best_len, deaths, games = [], 0, 0, 0, 1
    game = new_game(seed)
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        if not game.alive or game.won or game.ticks >= max_ticks:
            deaths += not game.alive
            game, games = new_game(seed + games), games + 1
        t0 = time.perf_counter()
        move = decide(game)
        lat.append((time.perf_counter() - t0) * 1e3)
        before = game.score
        game.step(move)
        food += game.score - before
        best_len = max(best_len, len(game.body))
    deaths += not game.alive
    a = np.asarray(lat)
    return {"seconds": seconds, "decisions": len(lat), "decisions_per_s": len(lat) / seconds,
            "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "food_30s": food, "best_length": best_len, "deaths": deaths, "games": games}


def protocol_b(decide, games=10, seed=101, max_ticks=500):
    """LayaStudio `play` + `summarize`: unassisted top-1 moves, 10 games."""
    res = []
    for i in range(games):
        game, lat, legal, agreed = new_game(seed + i), [], 0, 0
        while game.alive and not game.won and game.ticks < max_ticks:
            options = {m.direction: m for m in game.moves()}
            t0 = time.perf_counter()
            move = decide(game)
            lat.append((time.perf_counter() - t0) * 1e3)
            legal += bool(options.get(move) and options[move].legal)
            agreed += move == teacher(game)
            game.step(move)
        n = max(1, len(lat))
        res.append({"moves": game.ticks, "score": game.score, "length": len(game.body), "won": game.won,
                    "legal_rate": legal / n, "teacher_agreement": agreed / n,
                    "ms_per_decision": float(np.median(lat)) if lat else None})
    moves = [r["moves"] for r in res]
    lat = [r["ms_per_decision"] for r in res if r["ms_per_decision"]]
    return {"games": games, "moves_mean": float(np.mean(moves)), "moves_median": float(np.median(moves)),
            "score_mean": float(np.mean([r["score"] for r in res])), "score_best": max(r["score"] for r in res),
            "legal_rate": float(np.mean([r["legal_rate"] for r in res])),
            "teacher_agreement": float(np.mean([r["teacher_agreement"] for r in res])),
            "ms_per_decision": float(np.median(lat)) if lat else None}


def glance_timer(cand_id):
    """Our candidate (random weights) deciding from the board image: render + preprocess +
    forward + head, every step cold (the board changes each step). Latency only."""
    import torch
    from transformers import AutoTokenizer

    from glance.data.packing import collate, encode_question
    from glance.data.schema import Decision
    from glance.model.candidates import build_candidate, normalize

    tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m")
    cand = build_candidate(cand_id).eval()
    q = QUESTIONS["move"]
    d = Decision("snake", "snake", "choice", q["instructions"], [f"{k}: {v}" for k, v in q["criteria"].items()],
                 [1.0, 0.0, 0.0, 0.0], b"", "snake")
    batch = collate([[encode_question(tok, d)]], tok.pad_token_id)  # constant question: encoded once
    dirs = list(q["criteria"])

    @torch.inference_mode()
    def decide(game):
        img = torch.from_numpy(np.asarray(board_image(game, cand.image_size)))[None]
        px = normalize(img)
        image_input = cand.image_features(px) if cand.cached_image else px
        return dirs[int(cand(image_input, batch).argmax())]

    for _ in range(5):
        decide(new_game(1))
    return decide


def timed_on_teacher_trajectory(decide, seconds=30.0, seed=101, max_ticks=500):
    """Time `decide` on the boards the planner reaches (the untrained model's own moves would end
    the game at once, which would measure nothing)."""
    lat, game, games = [], new_game(seed), 1
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        if not game.alive or game.won or game.ticks >= max_ticks:
            game, games = new_game(seed + games), games + 1
        t0 = time.perf_counter()
        decide(game)
        lat.append((time.perf_counter() - t0) * 1e3)
        game.step(teacher(game) or "UP")
    a = np.asarray(lat)
    return {"seconds": seconds, "decisions": len(lat), "decisions_per_s": len(lat) / seconds,
            "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "note": "latency only"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, choices=["laya-torch", "laya-mlx", "teacher", "glance", "jev"])
    ap.add_argument("--model", default="")
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = default)")
    ap.add_argument("--protocol", default="both", choices=["a", "b", "both"])
    ap.add_argument("--out", default="results/bench/snake_cpu.jsonl")
    a = ap.parse_args()
    import torch

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)  # Laya's own CPU finding: inter-op threads hurt batch-1 latency
    t_load = time.perf_counter()
    decide = glance_timer(a.model) if a.runtime == "glance" else load_decider(a.runtime, a.model)
    rec = {"runtime": a.runtime, "model": a.model or a.runtime, "threads": torch.get_num_threads(),
           "load_s": time.perf_counter() - t_load, "device": "cpu",
           "chip": os.popen("sysctl -n machdep.cpu.brand_string").read().strip()}
    if a.runtime == "glance":
        rec["A_30s"] = timed_on_teacher_trajectory(decide)
    else:
        if a.protocol in ("a", "both"):
            rec["A_30s"] = protocol_a(decide)
        if a.protocol in ("b", "both"):
            rec["B_layastudio"] = protocol_b(decide)
    if a.runtime == "jev":
        rec.update(device="cloud API over the network from this Mac", jev_usage=dict(JEV_USAGE))
    print(json.dumps(rec, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
