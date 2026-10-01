"""SigLIP2 zero-shot floor (C0): score candidate texts against the image.

Text is lowercased (SigLIP2 was trained on lowercased text; the loaded Gemma tokenizer ignores the
config's do_lower_case flag) and logits are taken in fp32 (fp16 produced exact ties). Several
constructions are preregistered per decision kind and all are written out as separate
"logits_<name>" variants; the summary picks one per benchmark by calib-split NLL, never by
report-split results (R0 eval audit).

choice: "qo" = question + option, "o" = option only; SugarCrepe candidates are captions ("cap").
noul: POPE-style "is there a/an X in the image?" -> "a photo of a/an x." and P(yes) = sigmoid(logit),
expressed as logits [l, 0] (Platt scaling fits the threshold); other yes/no questions fall back to
a two-way softmax over "q yes" / "q no".
score: "levels" = softmax over "a/an {level} quality photo."; "antonym" = CLIP-IQA's
"good photo." vs "bad photo." difference s, mapped to level logits k * s (monotone in s, so SRCC
depends only on s; temperature fits the slope).
"""
import re

import torch
from transformers import AutoModel, AutoProcessor

from glance.data.schema import Decision

_PRESENCE = re.compile(r"^is there an? (.+?) in the (?:image|picture|photo)\??$", re.I)


def _a(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


class SiglipZeroShot:
    def __init__(self, repo: str, device: torch.device, dtype=torch.float32):
        self.model = AutoModel.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(repo)
        self.device, self.dtype = device, dtype

    @staticmethod
    def constructions(d: Decision) -> dict[str, list[str]]:
        if d.kind == "choice":
            if d.source.startswith("sugarcrepe"):
                return {"cap": d.candidates}
            return {"qo": [f"{d.question} {c}" for c in d.candidates], "o": list(d.candidates)}
        if d.kind == "noul":
            m = _PRESENCE.match(d.question.strip())
            if m:
                return {"presence": [f"a photo of {_a(m.group(1))} {m.group(1)}."]}
            return {"yn": [f"{d.question} yes", f"{d.question} no"]}
        return {"levels": [f"{_a(lvl)} {lvl} quality photo." for lvl in d.candidates],
                "antonym": ["good photo.", "bad photo."]}

    @torch.inference_mode()
    def _score(self, image, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=[t.lower() for t in texts], images=image, padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return self.model(**inputs).logits_per_image[0].float().cpu()

    def logits(self, d: Decision) -> dict:
        image, out = d.image(), {}
        for name, texts in self.constructions(d).items():
            s = self._score(image, texts)
            if name == "presence":
                out[f"logits_{name}"] = [float(s[0]), 0.0]
            elif name == "antonym":
                diff = float(s[0] - s[1])
                out[f"logits_{name}"] = [k * diff for k in range(len(d.candidates))]
            else:
                out[f"logits_{name}"] = s.tolist()
        out["logits"] = next(iter(out.values()))  # first construction, for code that reads "logits"
        return out
