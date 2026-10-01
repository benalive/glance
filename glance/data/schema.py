"""The typed-decision record every loader, model and metric shares.

A Decision is one question about one image with an explicit candidate list. yes/no ("noul") uses
candidates ["yes", "no"]; "score" uses one candidate per ordinal level, lowest first. `target` is a
distribution over candidates: one-hot for gold labels, or a histogram (rater votes, teacher).
"""
import hashlib
import io
from dataclasses import dataclass, field

from PIL import Image

KINDS = ("choice", "noul", "score")


@dataclass
class Decision:
    uid: str
    source: str
    kind: str
    question: str
    candidates: list[str]
    target: list[float]
    image_bytes: bytes
    image_id: str  # grouping key for splits and contamination checks, e.g. "coco:310196"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.kind in KINDS, self.kind
        assert len(self.candidates) == len(self.target) >= 2, (self.uid, self.candidates, self.target)
        assert abs(sum(self.target) - 1) < 1e-4, (self.uid, self.target)

    def image(self) -> Image.Image:
        return Image.open(io.BytesIO(self.image_bytes)).convert("RGB")

    @property
    def label(self) -> int:
        return max(range(len(self.target)), key=self.target.__getitem__)


def bytes_id(data: bytes) -> str:
    return "sha1:" + hashlib.sha1(data).hexdigest()[:16]


def calib_split(image_id: str, frac: float = 0.2) -> str:
    """Deterministic image-grouped split: 'calib' (temperature fitting) or 'report'."""
    h = int(hashlib.sha1(image_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "calib" if h < frac else "report"
