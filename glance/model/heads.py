"""Decision head and losses shared by every candidate tower.

One pointer head serves all three decision kinds: each candidate is represented by the mean of
its own token states (its words plus its closing slot token), and its logit is a scaled dot
product between that projected representation and the projected summary state of its question
(the question block's first token). Reading only the shared slot token put every candidate at the
same token and position, so at initialisation the candidates were near-identical (cos 0.999996)
and gradients cancelled (~1e-6): towers whose text path was not language-pretrained (C3/C4/C6)
barely learned. Found in the Snake pure-vision investigation, 2026-09-23. yes/no uses the candidates ["yes", "no"];
score uses one candidate per level, lowest first, trained with the ranked probability score so
ordering matters. Using the same head everywhere keeps the R1 comparison about the towers.
CORN is deferred to an R3 ablation.

Loss per question = Brier (RPS for score) + 0.3 * cross-entropy, both against the target
distribution, which may be soft.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerHead(nn.Module):
    def __init__(self, d: int, d_head: int = 256):
        super().__init__()
        self.q = nn.Linear(d, d_head)
        self.c = nn.Linear(d, d_head)
        self.scale = 1.0 / math.sqrt(d_head)

    def forward(self, states, q_pos, slot_pos, slot_q, cand_tok_pos=None, cand_tok_slot=None):
        """states (B, T, d); q_pos (Nq, 2) and slot_pos (Ns, 2) hold (batch, position) indices;
        slot_q (Ns,) gives each slot's question index; cand_tok_pos (Nt, 2) / cand_tok_slot (Nt,)
        list every candidate token and its slot, for the mean-pooled candidate representation
        (without them, the slot token alone is used). Returns (Ns,) logits."""
        hq = self.q(states[q_pos[:, 0], q_pos[:, 1]])
        if cand_tok_pos is None:
            cand = states[slot_pos[:, 0], slot_pos[:, 1]]
        else:
            tok = states[cand_tok_pos[:, 0], cand_tok_pos[:, 1]]
            cand = torch.zeros(len(slot_pos), states.shape[-1], device=states.device, dtype=tok.dtype)
            cand = cand.index_add(0, cand_tok_slot, tok)
            count = torch.bincount(cand_tok_slot, minlength=len(slot_pos)).clamp_min(1).to(tok.dtype)
            cand = cand / count[:, None]
        hc = self.c(cand)
        return (hc * hq[slot_q]).sum(-1) * self.scale


def pad_by_question(logits, slot_q, slot_k, n_q: int, k_max: int):
    """Scatter flat slot logits into (Nq, k_max), with -inf where a question has fewer candidates."""
    out = logits.new_full((n_q, k_max), float("-inf"))
    out[slot_q, slot_k] = logits
    return out


def decision_loss(padded_logits, targets, is_score, ce_weight: float = 0.3):
    """padded_logits (Nq, K) with -inf padding; targets (Nq, K) distributions (zero in padding);
    is_score (Nq,) bool. Returns (mean loss, probabilities)."""
    logp = F.log_softmax(padded_logits.float(), dim=-1)
    p = logp.exp()
    ce = -(targets * logp.clamp_min(-1e4)).sum(-1)
    brier = ((p - targets) ** 2).sum(-1)
    rps = ((p.cumsum(-1) - targets.cumsum(-1)) ** 2).sum(-1)
    proper = torch.where(is_score, rps, brier)
    return (proper + ce_weight * ce).mean(), p
