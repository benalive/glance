"""MLX port of GlanceTower (C3/C4/C6) and PointerHead, for on-device (Apple Silicon) serving.

Parameter names mirror glance/model/towers.py so a trained torch checkpoint converts with
`from_torch`. The 32x32 patch convolution is a reshape + matmul (kernel == stride). Masks follow
glance/model/masks.py exactly (image sees image; each question sees the image and its own block;
candidate tokens see the question and their own candidate). Parity with the torch implementation
is tested in tests/test_mlx.py.
"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from glance.model.towers import TowerConfig

NEG = -3e4  # additive mask value that stays finite in fp16 (max 65504)


def gelu_tanh(x):
    return 0.5 * x * (1 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def text_block_mask(blocks: mx.array, cands: mx.array | None) -> mx.array:
    same = blocks[:, :, None] == blocks[:, None, :]
    if cands is not None:
        ci, cj = cands[:, :, None], cands[:, None, :]
        same = same & ((ci < 0) | (cj < 0) | (ci == cj))
    valid = (blocks >= 0)[:, :, None]
    eye = mx.eye(blocks.shape[1], dtype=mx.bool_)[None]
    return (same & valid) | eye


def joint_mask(n_img: int, blocks: mx.array, cands: mx.array | None) -> mx.array:
    """-> additive (B, 1, N, N) float mask."""
    B, T = blocks.shape
    img_img = mx.ones((B, n_img, n_img), dtype=mx.bool_)
    img_txt = mx.zeros((B, n_img, T), dtype=mx.bool_)
    txt_img = mx.broadcast_to((blocks >= 0)[:, :, None], (B, T, n_img))
    txt_txt = text_block_mask(blocks, cands)
    m = mx.concatenate([mx.concatenate([img_img, img_txt], 2), mx.concatenate([txt_img, txt_txt], 2)], 1)
    return mx.where(m, 0.0, NEG)[:, None]


def cached_text_mask(n_img: int, blocks: mx.array, cands: mx.array | None) -> mx.array:
    B, T = blocks.shape
    txt_img = mx.broadcast_to((blocks >= 0)[:, :, None], (B, T, n_img))
    m = mx.concatenate([txt_img, text_block_mask(blocks, cands)], 2)
    return mx.where(m, 0.0, NEG)[:, None]


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.v_proj, self.out_proj = nn.Linear(d, d), nn.Linear(d, d)

    def _split(self, x):
        B, N, _ = x.shape
        return x.reshape(B, N, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, mask=None, kv_prefix=None, context=None):
        src = x if context is None else context
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(src)), self._split(self.v_proj(src))
        own = (k, v)
        if kv_prefix is not None:
            k = mx.concatenate([kv_prefix[0], k], axis=2)
            v = mx.concatenate([kv_prefix[1], v], axis=2)
        mask = None if mask is None else mask.astype(q.dtype)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        B, H, N, Dh = o.shape
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, N, H * Dh)), own


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)

    def __call__(self, x):
        return self.fc2(gelu_tanh(self.fc1(x)))


class TowerLayer(nn.Module):
    def __init__(self, cfg: TowerConfig):
        super().__init__()
        d = cfg.width
        self.layer_norm1 = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.self_attn = Attention(d, cfg.heads)
        self.layer_norm2 = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.mlp = MLP(d, cfg.mlp_dim)
        self.layer_norm2_text = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.mlp_text = MLP(d, cfg.mlp_dim)

    def ffn_image(self, x):
        return x + self.mlp(self.layer_norm2(x))

    def ffn_text(self, x):
        return x + self.mlp_text(self.layer_norm2_text(x))

    def __call__(self, x, n_img, mask):
        a, _ = self.self_attn(self.layer_norm1(x), mask)
        x = x + a
        return mx.concatenate([self.ffn_image(x[:, :n_img]), self.ffn_text(x[:, n_img:])], axis=1)

    def forward_image(self, x):
        a, kv = self.self_attn(self.layer_norm1(x))
        return self.ffn_image(x + a), kv

    def forward_text(self, x, kv, mask):
        a, _ = self.self_attn(self.layer_norm1(x), mask, kv_prefix=kv)
        return self.ffn_text(x + a)


class GlanceTowerMLX(nn.Module):
    def __init__(self, cfg: TowerConfig):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.width, cfg.patch
        self.patch_embedding = nn.Linear(3 * p * p, d)
        self.position_embedding = nn.Embedding(cfg.n_img, d)
        self.tok_embedding = nn.Embedding(cfg.vocab_size, d)
        self.text_pos_embedding = nn.Embedding(cfg.max_text_pos, d)
        self.text_embed_norm = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.layers = [TowerLayer(cfg) for _ in range(cfg.depth)]
        self.post_layernorm = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.post_layernorm_text = nn.LayerNorm(d, eps=cfg.ln_eps)

    def embed_image(self, pixels):
        """pixels (B, 3, H, W), already normalised."""
        B, C, H, W = pixels.shape
        p = self.cfg.patch
        x = pixels.reshape(B, C, H // p, p, W // p, p).transpose(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)
        return self.patch_embedding(x) + self.position_embedding.weight[None]

    def embed_text(self, ids, pos):
        return self.text_embed_norm(self.tok_embedding(ids) + self.text_pos_embedding(pos))

    def encode_image(self, pixels):
        x, kv = self.embed_image(pixels), []
        for layer in self.layers:
            x, layer_kv = layer.forward_image(x)
            kv.append(layer_kv)
        return kv, self.post_layernorm(x)

    def encode_text(self, ids, pos, blocks, kv, cands=None):
        x = self.embed_text(ids, pos)
        mask = cached_text_mask(self.cfg.n_img, blocks, cands)
        for layer, layer_kv in zip(self.layers, kv):
            x = layer.forward_text(x, layer_kv, mask)
        return self.post_layernorm_text(x)

    def __call__(self, pixels, ids, pos, blocks, cands=None):
        n_img = self.cfg.n_img
        x = mx.concatenate([self.embed_image(pixels), self.embed_text(ids, pos)], axis=1)
        mask = joint_mask(n_img, blocks, cands)
        for layer in self.layers:
            x = layer(x, n_img, mask)
        return self.post_layernorm(x[:, :n_img]), self.post_layernorm_text(x[:, n_img:])


class PointerHeadMLX(nn.Module):
    def __init__(self, d: int, d_head: int = 256):
        super().__init__()
        self.q, self.c = nn.Linear(d, d_head), nn.Linear(d, d_head)
        self.scale = 1.0 / math.sqrt(d_head)

    def __call__(self, states, q_pos, slot_pos, slot_q, cand_tok_pos=None, cand_tok_slot=None):
        """Mean-pooled candidate readout, as glance/model/heads.py (slot token alone if no spans)."""
        hq = self.q(states[q_pos[:, 0], q_pos[:, 1]])
        if cand_tok_pos is None:
            cand = states[slot_pos[:, 0], slot_pos[:, 1]]
        else:
            tok = states[cand_tok_pos[:, 0], cand_tok_pos[:, 1]]
            n = slot_pos.shape[0]
            onehot = (cand_tok_slot[None, :] == mx.arange(n)[:, None]).astype(tok.dtype)  # (Ns, Nt)
            cand = (onehot @ tok) / mx.maximum(onehot.sum(1, keepdims=True), 1)
        hc = self.c(cand)
        return (hc * hq[slot_q]).sum(-1) * self.scale


def from_torch(state_dict: dict, prefix: str = "") -> list[tuple[str, mx.array]]:
    """torch GlanceTower/PointerHead state dict -> MLX (name, array) pairs. The patch conv weight
    (d, 3, p, p) becomes a Linear weight (d, 3*p*p) in the same (c, kh, kw) order."""
    out = []
    for name, t in state_dict.items():
        if not name.startswith(prefix):
            continue
        a = t.detach().float().cpu().numpy()
        key = name[len(prefix):]
        if key == "patch_embedding.weight":
            a = a.reshape(a.shape[0], -1)
        out.append((key, mx.array(a)))
    return out


def batch_to_mlx(batch: dict) -> dict:
    return {k: mx.array(np.asarray(v)) if hasattr(v, "shape") else v for k, v in batch.items()}
