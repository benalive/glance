"""Attention masks for packed [image][question blocks] sequences.

Bool masks follow SDPA: True means "may attend". Image tokens see only image tokens (deferred
fusion at k = L), so their keys/values do not depend on any question and can be cached. Text
tokens see the whole image plus their own question block, never another question.

Inside a block, `cand_ids` marks question tokens (-1) and the tokens of candidate k (k >= 0).
Question tokens see the whole block; candidate tokens see the question and only their own
candidate. Without this, every candidate's slot token would see the same set of tokens at the
same position and all slots would collapse to one state (found in R1: yes/no came out exactly
0.5). Cross-candidate information still reaches each slot through the question tokens.

`block_ids` gives the question index of every text token; -1 marks padding. Padding attends to
itself only, so SDPA never sees an all-False row (which would produce NaN).
"""
import torch


def text_block_mask(block_ids: torch.Tensor, cand_ids: torch.Tensor | None = None) -> torch.Tensor:
    """(B, T) block ids [, (B, T) candidate ids] -> (B, T, T) bool."""
    same = block_ids[:, :, None] == block_ids[:, None, :]
    if cand_ids is not None:
        ci, cj = cand_ids[:, :, None], cand_ids[:, None, :]
        same = same & ((ci < 0) | (cj < 0) | (ci == cj))
    valid = (block_ids >= 0)[:, :, None]
    eye = torch.eye(block_ids.shape[1], dtype=torch.bool, device=block_ids.device)
    return (same & valid) | eye


def joint_mask(n_img: int, block_ids: torch.Tensor, cand_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Mask for one pass over [image (n_img)][text (T)] -> (B, 1, n_img+T, n_img+T)."""
    B, T = block_ids.shape
    N = n_img + T
    m = torch.zeros(B, N, N, dtype=torch.bool, device=block_ids.device)
    m[:, :n_img, :n_img] = True
    m[:, n_img:, :n_img] = (block_ids >= 0)[:, :, None]
    m[:, n_img:, n_img:] = text_block_mask(block_ids, cand_ids)
    return m[:, None]


def with_window(mask: torch.Tensor, window: int, positions: torch.Tensor) -> torch.Tensor:
    """Restrict a (B, 1, N, N) mask to |pos_i - pos_j| <= window // 2 (ModernBERT's local layers).

    The band is on position ids, not sequence index: ModernBERT's native band is on sequence
    index, which in a packed [image][q1..q4] sequence made how much of the image a question sees
    depend on where it was packed (R1 training-impl audit, F4: the last block of C1 saw 0 of 64
    image tokens in local layers, the first block 52). With position ids (image 0..n-1, text
    n + block-local position) every block sees the same band; within one unpacked sequence the
    two definitions coincide."""
    band = (positions[:, :, None] - positions[:, None, :]).abs() <= window // 2
    return mask & band[:, None]


def modernbert_masks(mask: torch.Tensor, window: int, positions: torch.Tensor) -> dict:
    """The per-layer-type dict ModernBERT accepts, so local layers keep their pretrained window."""
    return {"full_attention": mask, "sliding_attention": with_window(mask, window, positions)}


def cached_text_mask(n_img: int, block_ids: torch.Tensor, cand_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Mask for text queries over [cached image keys][text keys] -> (B, 1, T, n_img+T)."""
    B, T = block_ids.shape
    img = (block_ids >= 0)[:, :, None].expand(B, T, n_img)
    return torch.cat([img, text_block_mask(block_ids, cand_ids)], dim=2)[:, None]
