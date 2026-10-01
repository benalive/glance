"""Accuracy and calibration of every Snake model on the same fresh test boards.

Test set: planner games with 20% exploration from game seeds 20011.. (disjoint from training,
seeds 11..380; from the protocol A/B play seeds 101..; and from the development boards, seed
5011, that guided design choices). Each game is played out and at most PER_GAME boards are
sampled uniformly from it, so the set spans ~100 games rather than a handful of long ones.
Boards whose state (body, food) occurs in the 100k training set (which contains the 2,339-board
matched set as its prefix) are dropped.

Per model: top-1 accuracy vs the planner's move (game-grouped bootstrap CI), tie-aware accuracy,
legal-move rate of the top choice, trap-board accuracy, NLL / Brier / smooth ECE as shipped and
after one temperature fitted on a 20% calibration split of games. Also: two zero-parameter greedy
rules, pairwise top-choice agreement, and a paired, game-grouped bootstrap of per-board NLL
differences. Per-board probabilities are saved so every number here can be recomputed.

    uv run --extra bench python scripts/snake_accuracy.py [--reuse-probs [--rescore "model name,..."]]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

from glance.metrics import brier, fit_temperature, grouped_bootstrap_ci, nll, paired_bootstrap, smooth_ece, softmax, top_label
from scripts.snake_cpu_bench import QUESTIONS, free_space, new_game, render

DIRS = list(QUESTIONS["move"]["criteria"])
SEED, N, PER_GAME = 20011, 2000, 20
OUT = Path("results/bench")
BOARDS = OUT / "snake_test_boards.json"  # cached: generating the exclusion set is slow


def planner_scores(game):
    """LayaStudio's teacher key per legal move: (enough room, room, -distance to food)."""
    food, length = game.food, len(game.body)
    out = {}
    for m in game.moves():
        if m.legal:
            t = game.target(m.direction)
            room = free_space(game, m.direction)
            out[m.direction] = (room >= length, room, -(abs(t[0] - food[0]) + abs(t[1] - food[1])))
    return out


def key(body, food):
    return (tuple(map(tuple, body)), tuple(food))


def training_keys():
    from glance.snake import generate

    return {key(r["body"], r["food"]) for r in generate(100_000, seed=11)}


def boards(train_keys, n=N, seed=SEED, per_game=PER_GAME, explore=0.2, max_ticks=400):
    """Same trajectory generator as glance.snake.generate / LayaStudio; label = LayaStudio's
    tie-break (sort descending, first). Samples <= per_game boards per game, uniformly."""
    from glance.snake import facts

    rng, pick, out, g, dropped = random.Random(seed), random.Random(seed + 1), [], 0, 0
    while len(out) < n:
        game, g = new_game(seed + g), g + 1
        rows = []
        for _ in range(max_ticks):
            if not game.alive or game.won:
                break
            sc = planner_scores(game)
            if not sc:
                break
            best = sorted(((v, d) for d, v in sc.items()), reverse=True)[0][1]
            top = max(sc.values())
            closest = max(sc, key=lambda d: sc[d][2])
            if key(game.body, game.food) in train_keys:
                dropped += 1
            else:
                rows.append({"game": g, "text": render(game), "facts": facts(game), "head": game.head,
                             "body": list(game.body), "food": game.food, "label": DIRS.index(best),
                             "acceptable": [DIRS.index(d) for d, v in sc.items() if v == top],
                             "legal": [DIRS.index(d) for d in sc], "planner": {DIRS.index(d): v for d, v in sc.items()},
                             "trap": closest != best and sc[closest][2] > sc[best][2]})
            legal = list(sc)
            game.step(rng.choice(legal) if legal and rng.random() < explore else best)
        out += pick.sample(rows, min(per_game, len(rows)))
    return out[:n], g, dropped


def greedy_probs(rows, tie="name"):
    """Zero-parameter rule from the fact lines: among free neighbours, step closest to the food.
    Exact ties are broken as the teacher's labels are (sort descending over (key, direction name),
    i.e. the alphabetically last direction wins). tie='name': distance only, which the fact lines
    give; tie='planner': the planner's full (enough room, room, -distance) key, which they do not."""
    out = []
    for r in rows:
        sc = r["planner"]
        if tie == "name":
            k = max(sc, key=lambda j: (sc[j][2], DIRS[j]))
        else:
            k = max(sc, key=lambda j: (sc[j][2], sc[j][0], sc[j][1], DIRS[j]))
        p = np.full(4, 1e-6)
        p[k] = 1.0
        out.append(p / p.sum())
    return out


def laya_probs(rows, model="madhavbiplov/laya-snake-mlx"):
    import laya

    agent = laya.load(model, device="cpu")
    out = []
    for i, r in enumerate(rows):
        if i % 250 == 0:
            print(f"  laya {i}/{len(rows)}", flush=True)
        p = agent.predict(r["text"], QUESTIONS)["answers"]["move"]["probabilities"]
        out.append(np.array([p[d] for d in DIRS], dtype=np.float64))
    return out


def glance_probs(rows, run):
    import torch

    from glance.model.candidates import build_candidate
    from glance.snake import Encoder, _forward, context_of, image_of

    s = json.loads((Path(run) / "train_summary.json").read_text())
    cand = build_candidate(s["cand"])
    cand.load_state_dict(torch.load(Path(run) / "model.pt", weights_only=True), strict=False)
    cand.eval()
    enc, variant, out = Encoder(s["variant"]), s["variant"], []
    with torch.inference_mode():
        for i in range(0, len(rows), 64):
            chunk = rows[i:i + 64]
            px = torch.from_numpy(np.stack([image_of(variant, r["head"], r["body"], r["food"]) for r in chunk]))
            logits, _ = _forward(cand, px, enc.batch([context_of(variant, r) for r in chunk]), torch.device("cpu"))
            for row in logits.reshape(len(chunk), 4).double().numpy():
                out.append(softmax(row))
    return out


def split(rows):
    cal = [i for i, r in enumerate(rows) if r["game"] % 5 == 0]  # 20% of games, grouped
    s = set(cal)
    return cal, [i for i in range(len(rows)) if i not in s]


def evaluate(name, probs, rows):
    ys = [[1.0 if j == r["label"] else 0.0 for j in range(4)] for r in rows]
    cal, rep = split(rows)
    t = fit_temperature([np.log(np.clip(probs[i], 1e-12, 1)) for i in cal], [ys[i] for i in cal])
    P = [probs[i] for i in rep]
    Pt = [softmax(np.log(np.clip(probs[i], 1e-12, 1)), t) for i in rep]
    Y, R = [ys[i] for i in rep], [rows[i] for i in rep]
    top = [int(p.argmax()) for p in P]
    correct = [float(k == r["label"]) for k, r in zip(top, R)]
    trap = [c for c, r in zip(correct, R) if r["trap"]]
    conf, corr = top_label(P, Y)
    conf_t, corr_t = top_label(Pt, Y)
    return {"model": name, "n": len(R), "games": len({r["game"] for r in R}), "top1": float(np.mean(correct)),
            "top1_ci95": grouped_bootstrap_ci(correct, [r["game"] for r in R]),
            "tie_aware": float(np.mean([k in r["acceptable"] for k, r in zip(top, R)])),
            "legal": float(np.mean([k in r["legal"] for k, r in zip(top, R)])),
            "trap_n": len(trap), "trap_top1": float(np.mean(trap)) if trap else None,
            "nll": nll(P, Y), "brier": brier(P, Y), "smece": smooth_ece(conf, corr),
            "T": t, "nll_T": nll(Pt, Y), "brier_T": brier(Pt, Y), "smece_T": smooth_ece(conf_t, corr_t)}


def per_board_nll(probs, rows, idx):
    return [-float(np.log(max(probs[i][rows[i]["label"]], 1e-12))) for i in idx]


def main():
    import torch

    torch.set_num_threads(4)  # batch-1 CPU inference is fastest at 4 threads on this machine
    if BOARDS.exists():
        cached = json.loads(BOARDS.read_text())
        rows, n_games, dropped = cached["rows"], cached["games_played"], cached["dropped"]
        for r in rows:
            r["planner"] = {int(k): tuple(v) for k, v in r["planner"].items()}
    else:
        rows, n_games, dropped = boards(training_keys())
        BOARDS.write_text(json.dumps({"seed": SEED, "per_game": PER_GAME, "games_played": n_games, "dropped": dropped,
                                      "rows": rows}))
    cal, rep = split(rows)
    labels = np.bincount([rows[i]["label"] for i in rep], minlength=4)
    print(f"{len(rows)} boards from {len({r['game'] for r in rows})} games ({n_games} played, {dropped} training-set "
          f"boards dropped); report split {len(rep)} boards / {len({rows[i]['game'] for i in rep})} games; "
          f"{sum(rows[i]['trap'] for i in rep)} traps", flush=True)

    probs_file = OUT / "snake_accuracy_probs.jsonl"
    saved = {}
    if "--reuse-probs" in sys.argv and probs_file.exists():  # recompute metrics without re-running models
        lines = [json.loads(l) for l in probs_file.read_text().splitlines()]
        rescore = set()  # --rescore "name,name": re-run these models (e.g. after retraining) and reuse the rest
        if "--rescore" in sys.argv:
            rescore = set(sys.argv[sys.argv.index("--rescore") + 1].split(","))
        saved = {n: [np.array(l["probs"][n]) for l in lines] for n in lines[0]["probs"] if n not in rescore}
    models = {"Greedy rule (fact lines)": greedy_probs(rows, "name"),
              "Greedy rule (planner key)": greedy_probs(rows, "planner"),
              "Laya-snake (text)": saved.get("Laya-snake (text)") or laya_probs(rows)}
    for run, name in [("results/snake/C5_text_matched", "Glance C5 text"),
                      ("results/snake/C5_img_facts_matched", "Glance C5 image+facts"),
                      ("results/snake/C5_img_facts", "Glance C5 image+facts (100k boards)"),
                      ("results/snake/C6s_img", "Glance C6s image"),
                      ("results/snake/C6s_img_aux", "Glance C6s image+aux"),
                      ("results/snake/C3_img_aux", "Glance C3 image+aux")]:
        models[name] = saved.get(name) or glance_probs(rows, run)
        print("scored", name, flush=True)
    results = [evaluate(name, p, rows) for name, p in models.items()]
    for r in results:
        print(json.dumps(r), flush=True)

    names = list(models)
    agree = {a: {b: float(np.mean([models[a][i].argmax() == models[b][i].argmax() for i in range(len(rows))]))
                 for b in names} for a in names}
    groups = [rows[i]["game"] for i in rep]
    laya_nll = per_board_nll(models["Laya-snake (text)"], rows, rep)
    paired = {name: paired_bootstrap(per_board_nll(models[name], rows, rep), laya_nll, groups)
              for name in ("Glance C5 text", "Glance C5 image+facts")}
    majority = float(labels.max() / labels.sum())

    (OUT / "snake_accuracy_probs.jsonl").write_text("".join(
        json.dumps({"game": r["game"], "label": r["label"], "trap": r["trap"], "split": "calib" if r["game"] % 5 == 0 else "report",
                    "probs": {n: [round(float(x), 6) for x in models[n][i]] for n in names}}) + "\n"
        for i, r in enumerate(rows)))
    meta = {"seed": SEED, "boards": len(rows), "games": len({r["game"] for r in rows}), "per_game_cap": PER_GAME,
            "training_boards_dropped": dropped, "report_boards": len(rep), "report_games": len(set(groups)),
            "majority_baseline_report": majority}
    (OUT / "snake_accuracy.json").write_text(json.dumps({"meta": meta, "rows": results, "agreement_all_boards": agree,
                                                        "paired_nll_minus_laya_report": paired}, indent=1))
    lines = [f"Fresh test boards: {len(rows)} boards from {meta['games']} planner games (seeds from {SEED}, at most "
             f"{PER_GAME} boards per game, {dropped} boards that occur in the training sets dropped). Metrics on the "
             f"report split ({len(rep)} boards, {len(set(groups))} games); temperature fitted on the other 20% of games. "
             f"CIs: game-grouped bootstrap. Majority-move baseline (report split): {majority:.3f}.", "",
             "| model | top-1 [95% CI] | tie-aware | legal top choice | trap boards (n) | NLL / Brier / smECE as shipped "
             "| T | NLL / Brier / smECE with T |", "|" + "---|" * 8]
    for r in results:
        lo, hi = r["top1_ci95"]
        trap = f"{r['trap_top1']:.3f} ({r['trap_n']})" if r["trap_n"] else "— (0)"
        lines.append(f"| {r['model']} | {r['top1']:.3f} [{lo:.3f}, {hi:.3f}] | {r['tie_aware']:.3f} | {r['legal']:.3f} "
                     f"| {trap} | {r['nll']:.3f} / {r['brier']:.3f} / {r['smece']:.3f} "
                     f"| {r['T']:.2f} | {r['nll_T']:.3f} / {r['brier_T']:.3f} / {r['smece_T']:.3f} |")
    lines += ["", "Top-choice agreement on all boards:", "", "| | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    lines += [f"| {a} | " + " | ".join(f"{agree[a][b]:.4f}" for b in names) + " |" for a in names]
    lines += ["", "Per-board NLL difference vs Laya-snake, as shipped (report split, paired game-grouped bootstrap):", ""]
    lines += [f"- {n} − Laya: {v['diff']:+.4f} nats, 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for n, v in paired.items()]
    (OUT / "snake_accuracy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
