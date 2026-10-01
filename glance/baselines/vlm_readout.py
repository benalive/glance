"""Small-VLM logit readout floor (C0) and teacher proxy: one forward pass, read next-token logits.

choice: options are lettered A, B, ...; the option logit is the log-sum-exp of the log-probs of
the letter's single-token spellings ("A", " A"). We also run every cyclic rotation of the options
and average probabilities per original option (CircularEval-style debiasing) and keep both.
noul: log-sum-exp over yes/Yes/ yes/ Yes vs the no spellings. score: the digits 1-5.

Image budget is held at 64 visual tokens for every model: SmolVLM with image splitting off (one
512 px tile, 64 tokens after its 4x pixel shuffle); Qwen3-VL resized to ~256x256 pixels.

SmolVLM-256M opens every reply with " Answer: X." (checked in R0: ~100% of next-token mass sits on
" Answer"), so its assistant turn is pre-filled with " Answer:" and the letter/word is read at the
next position. Its tokenizer splits " 3" into [" ", "3"], so for score items the prefill ends in a
space and the bare digit is read (R0 eval audit: without it 48% of the mass sat on the space token
and 1990/2015 KonIQ predictions collapsed to level 3). `mass` records the probability the model puts
on valid answer tokens, so a readout that misses the format is visible.

The rotation-averaged variant ("logits_debiased") averages probabilities over the K cyclic option
orders; it is not MMBench CircularEval (which requires every rotation correct) and costs K passes.
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from glance.data.schema import Decision

LETTERS = "ABCDEFGHIJ"


def _prompt(d: Decision, order: list[int]) -> str:
    if d.kind == "choice":
        opts = "\n".join(f"{LETTERS[i]}. {d.candidates[j]}" for i, j in enumerate(order))
        return f"{d.question}\n{opts}\nAnswer with the option's letter from the given choices directly."
    if d.kind == "noul":
        return f"{d.question}\nPlease answer yes or no."
    scale = ", ".join(f"{i + 1} = {lvl}" for i, lvl in enumerate(d.candidates))
    return f"{d.question}\nRate it on a scale from 1 to {len(d.candidates)} ({scale}). Answer with a single digit."


# Assistant-turn prefill per model family and decision kind.
ANSWER_PREFIX = {"smolvlm": {"choice": " Answer:", "noul": " Answer:", "score": " Answer: "}}


class VLMReadout:
    def __init__(self, repo: str, device: torch.device, dtype=torch.float16, prefix: dict | None = None):
        default = next((v for k, v in ANSWER_PREFIX.items() if k in repo.lower()), {})
        self.answer_prefix = {**default, **(prefix or {})}
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForImageTextToText.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.device, self.dtype = device, dtype
        ip = self.processor.image_processor
        if hasattr(ip, "do_image_splitting"):  # SmolVLM / Idefics3
            ip.do_image_splitting = False
        if "qwen" in repo.lower():
            ip.size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256}
        tok = self.processor.tokenizer

        def ids(words):
            out = set()
            for w in words:
                e = tok.encode(w, add_special_tokens=False)
                if len(e) == 1:
                    out.add(e[0])
            assert out, words
            return sorted(out)

        self.letter_ids = [ids([L, " " + L]) for L in LETTERS]
        self.yes_ids = ids(["yes", "Yes", " yes", " Yes", "YES"])
        self.no_ids = ids(["no", "No", " no", " No", "NO"])
        words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.digit_ids = [ids([str(i), " " + str(i), w, " " + w, w.title(), " " + w.title()])
                          for i, w in enumerate(words, start=1)]

    def _logprobs(self, d: Decision, order: list[int]) -> torch.Tensor:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _prompt(d, order)}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True) + self.answer_prefix.get(d.kind, "")
        inputs = self.processor(text=[text], images=[[d.image()]], return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        logits = self.model(**inputs).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1).cpu()

    def _gather(self, lp: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
        return torch.stack([torch.logsumexp(lp[g], dim=0) for g in groups])

    @torch.inference_mode()
    def logits(self, d: Decision, debias: bool = True) -> dict:
        k = len(d.candidates)
        if d.kind == "noul":
            lp = self._logprobs(d, list(range(k)))
            opt = self._gather(lp, [self.yes_ids, self.no_ids])
            return {"logits": opt.tolist(), "mass": float(opt.exp().sum())}
        if d.kind == "score":
            lp = self._logprobs(d, list(range(k)))
            opt = self._gather(lp, self.digit_ids[:k])
            return {"logits": opt.tolist(), "mass": float(opt.exp().sum())}
        lp = self._logprobs(d, list(range(k)))
        raw = self._gather(lp, self.letter_ids[:k])
        out = {"logits": raw.tolist(), "mass": float(raw.exp().sum())}
        if debias:
            probs = torch.zeros(k)
            for shift in range(k):
                order = [(i + shift) % k for i in range(k)]  # position i shows candidate order[i]
                lp_s = lp if shift == 0 else self._logprobs(d, order)
                p = torch.softmax(self._gather(lp_s, self.letter_ids[:k]), dim=0)
                for pos, cand in enumerate(order):
                    probs[cand] += p[pos] / k
            out["logits_debiased"] = probs.clamp_min(1e-12).log().tolist()
        return out
