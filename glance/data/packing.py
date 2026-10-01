"""Turn Decisions into packed text sequences for the encoder towers.

One question block = [CLS] question tokens, then each candidate's tokens followed by [SEP] (its
slot). Positions restart at 0 in every block, and every candidate starts at the same position,
so no candidate can be told apart by position; `cand_ids` lets the mask keep each candidate's
tokens to itself so the slots stay distinct. Several questions about the same image share one
sequence; `block_ids` keeps them apart (see glance/model/masks.py).
"""
from collections import defaultdict
from dataclasses import dataclass

import torch

from glance.data.schema import Decision


@dataclass
class EncodedQuestion:
    ids: list[int]
    pos: list[int]
    cand: list[int]  # -1 for question tokens, k for tokens of candidate k (see masks.py)
    q_offset: int  # summary token, relative to block start
    slot_offsets: list[int]  # each candidate's closing [SEP]; its span is cand_starts[k]..slot_offsets[k]
    target: list[float]
    is_score: bool
    cand_starts: list[int] | None = None

    def __len__(self):
        return len(self.ids)


def encode_question(tok, d: Decision, max_q: int = 128, max_c: int = 64) -> EncodedQuestion:
    """Caps are generous: at 48/16 the R1 audit found 13-20% of SugarCrepe captions and 5% of
    MMStar questions truncated, some to identical candidates."""
    q = tok(d.question, add_special_tokens=False)["input_ids"][:max_q]
    ids = [tok.cls_token_id] + q
    pos = list(range(len(ids)))
    cand = [-1] * len(ids)
    start, slots, starts = len(ids), [], []
    for k, c in enumerate(d.candidates):
        c_ids = tok(c, add_special_tokens=False)["input_ids"][: max_c - 1] + [tok.sep_token_id]
        starts.append(len(ids))
        ids += c_ids
        pos += list(range(start, start + len(c_ids)))
        cand += [k] * len(c_ids)
        slots.append(len(ids) - 1)
    return EncodedQuestion(ids, pos, cand, 0, slots, list(d.target), d.kind == "score", starts)


def group_by_image(decisions: list[Decision], tok, max_len: int = 320, max_q_per_seq: int = 4):
    """-> list of ([Decision, ...], [EncodedQuestion, ...]) sequences, one image per sequence."""
    by_img = defaultdict(list)
    for d in decisions:
        by_img[d.image_id].append(d)
    seqs = []
    for items in by_img.values():
        ds, es, cur_len = [], [], 0
        for d in items:
            e = encode_question(tok, d)
            if es and (cur_len + len(e) > max_len or len(es) >= max_q_per_seq):
                seqs.append((ds, es))
                ds, es, cur_len = [], [], 0
            ds.append(d)
            es.append(e)
            cur_len += len(e)
        if es:
            seqs.append((ds, es))
    return seqs


def collate(batch: list[list[EncodedQuestion]], pad_id: int):
    """batch: one list of EncodedQuestions per sequence -> tensors for tower + head."""
    lens = [sum(len(e) for e in qs) for qs in batch]
    T = max(lens)
    B = len(batch)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    pos = torch.zeros((B, T), dtype=torch.long)
    blocks = torch.full((B, T), -1, dtype=torch.long)
    cands = torch.full((B, T), -1, dtype=torch.long)
    q_pos, slot_pos, slot_q, slot_k, targets, is_score = [], [], [], [], [], []
    tok_pos, tok_slot = [], []  # every candidate token and the slot index it belongs to (mean-pooled readout)
    k_max = max(len(e.slot_offsets) for qs in batch for e in qs)
    for b, qs in enumerate(batch):
        off = 0
        for qi, e in enumerate(qs):
            n = len(e)
            ids[b, off:off + n] = torch.tensor(e.ids)
            pos[b, off:off + n] = torch.tensor(e.pos)
            blocks[b, off:off + n] = qi
            cands[b, off:off + n] = torch.tensor(e.cand)
            q_index = len(q_pos)
            q_pos.append((b, off + e.q_offset))
            for k, s in enumerate(e.slot_offsets):
                start = e.cand_starts[k] if e.cand_starts else s
                for t in range(start, s + 1):
                    tok_pos.append((b, off + t))
                    tok_slot.append(len(slot_pos))
                slot_pos.append((b, off + s))
                slot_q.append(q_index)
                slot_k.append(k)
            targets.append(e.target + [0.0] * (k_max - len(e.target)))
            is_score.append(e.is_score)
            off += n
    return {"input_ids": ids, "position_ids": pos, "block_ids": blocks, "cand_ids": cands,
            "q_pos": torch.tensor(q_pos), "slot_pos": torch.tensor(slot_pos),
            "slot_q": torch.tensor(slot_q), "slot_k": torch.tensor(slot_k),
            "cand_tok_pos": torch.tensor(tok_pos), "cand_tok_slot": torch.tensor(tok_slot),
            "targets": torch.tensor(targets), "is_score": torch.tensor(is_score), "k_max": k_max}
