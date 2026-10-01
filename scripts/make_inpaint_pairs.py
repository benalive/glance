"""Paired training images for local flaws: the same clean generated image with a flaw inpainted, and with the
same region inpainted normally (the control, so a model cannot score "was inpainted" as "is flawed").

Source images: licence-clean training images judged clean by their annotators (EvalMuse-40K images with no
structural mark, from the generators whose output licences we cleared, except
Playground v2.5; ImageRewardDB images with fidelity >= 6), never calibration-split images. A box covering 8-25%
of the image, biased towards the centre, is inpainted with Stable Diffusion 1.5 inpainting (CreativeML
OpenRAIL-M, the same licence as SD 1.x outputs already in training) twice: once with a flaw prompt (a malformed
hand, a distorted face, a broken object, ...), once with the image's own prompt. Both results are pasted back
into the original through a feathered mask, so only the box changes. 10% of source images (by hash) form a
held-out test split.

    uv run --with diffusers --with accelerate python scripts/make_inpaint_pairs.py --n 3000   # -> data/clean/inpaint/
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from PIL import Image, ImageFilter

from scripts.build_clean_mix import is_calib

OUT = Path("data/clean/inpaint")
MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SKIP_GENERATORS = {"Playground_v2.5", "HunyuanDiT", "SDXL-Turbo", "SD3", "IF", "Kolors", "Dreamina_v2.0Pro", "Midjourney_v6.1"}
FLAWS = [  # (category, prompt)
    ("hand", "a deformed hand with too many fingers, fused twisted fingers"),
    ("hand", "a mangled malformed hand, extra fingers, broken knuckles"),
    ("face", "a distorted melted face, misplaced eyes, warped mouth"),
    ("face", "a disfigured asymmetric face with extra eyes"),
    ("limb", "a twisted broken limb bending the wrong way, extra arm"),
    ("limb", "extra legs growing out of the body, fused limbs"),
    ("animal", "a misshapen animal with extra legs and a warped head"),
    ("object", "a mangled warped object, broken geometry, melted shapes"),
    ("object", "an impossible distorted object fused into its surroundings"),
    ("glitch", "a garbled smeared blob of noise, glitchy corrupted texture"),
]
NEGATIVE = "clean, well formed, correct anatomy"


def split_of(image_id: str) -> str:
    return "test" if int(hashlib.sha1(image_id.encode()).hexdigest(), 16) % 10 == 0 else "train"


def sources():
    out = []
    for line in open("data/clean/labels_evalmuse.jsonl"):
        r = json.loads(line)
        if r["generator"] not in SKIP_GENERATORS and r["fidelity_label"] == [] and not is_calib(r["image_id"]):
            out.append((r["image_id"], r["path"], r["prompt"], r["generator"]))
    for line in open("data/clean/labels_imagereward.jsonl"):
        r = json.loads(line)
        if r.get("fidelity_rating", 0) >= 6 and r.get("split") == "train" and not is_calib(r["image_id"]) and len(r["prompt"]) > 3:
            out.append((r["image_id"], r["path"], r["prompt"], "SD_v1.x (ImageRewardDB)"))
    return out


def box(w: int, h: int, rng: random.Random) -> tuple[int, int, int, int]:
    area = rng.uniform(0.08, 0.25) * w * h
    ar = rng.uniform(0.6, 1.6)
    bw, bh = min(w, int((area * ar) ** 0.5)), min(h, int((area / ar) ** 0.5))
    cx = min(max(int(rng.gauss(w / 2, w / 6)), bw // 2), w - bw // 2)
    cy = min(max(int(rng.gauss(h / 2, h / 6)), bh // 2), h - bh // 2)
    return cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="source images (each gives a flawed and a control image)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
                                                          requires_safety_checker=False).to("mps")
    pipe.set_progress_bar_config(disable=True)
    rng = random.Random(a.seed)
    pool = sources()
    rng.shuffle(pool)
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "pairs.jsonl"
    done = {json.loads(l)["source_id"] for l in open(meta_path)} if meta_path.exists() else set()
    t0, made = time.time(), 0
    with open(meta_path, "a") as fh:
        for image_id, path, prompt, gen in pool[: a.n]:
            if image_id in done:
                continue
            src = Image.open(path).convert("RGB")
            s = 512 / max(src.size)
            w, h = (int(src.width * s) // 8 * 8, int(src.height * s) // 8 * 8)
            src = src.resize((w, h), Image.Resampling.BICUBIC)
            b = box(w, h, rng)
            mask = Image.new("L", (w, h), 0)
            mask.paste(255, b)
            feather = mask.filter(ImageFilter.GaussianBlur(6))
            cat, flaw = rng.choice(FLAWS)
            seed = rng.randrange(2**31)
            rec = {"source_id": image_id, "source_path": path, "generator": gen, "box": b, "size": [w, h], "flaw_category": cat,
                   "flaw_prompt": flaw, "prompt": prompt, "seed": seed, "split": split_of(image_id), "model": MODEL, "steps": a.steps}
            stem = hashlib.sha1(image_id.encode()).hexdigest()[:16]
            for kind, p, neg in (("flawed", flaw, NEGATIVE), ("control", prompt, None)):
                g = torch.Generator("cpu").manual_seed(seed)
                out = pipe(prompt=p, negative_prompt=neg, image=src, mask_image=mask, width=w, height=h,
                           num_inference_steps=a.steps, guidance_scale=7.5, generator=g).images[0]
                img = Image.composite(out.resize((w, h)), src, feather)
                dest = OUT / f"{stem}_{kind}.jpg"
                img.save(dest, quality=95)
                rec[f"{kind}_path"] = str(dest)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            made += 1
            if made % 25 == 0 or made <= 3:
                print(f"{made} pairs, {(time.time() - t0) / made:.1f} s/pair", flush=True)


if __name__ == "__main__":
    main()
