"""Candidate towers for the Glance comparison.

GlanceTower is one bidirectional stack over [image patches][question blocks] (candidates C3, C4,
C6). Parameter names mirror SigLIP's vision encoder so C3 can inherit SigLIP2-B/32 weights
directly; C4 is the same module left at random init. Each layer shares attention across
modalities and keeps a separate FFN (and pre-FFN norm) for text tokens, VLMo-style.

FusionTower (C5) keeps two frozen towers and adds a few trainable cross-attention layers; only
those layers and the text tower run per question.

Both expose the same three calls: `encode_image` (question-independent, cacheable),
`encode_text` (per question, reads the cache), and `forward` (both in one pass). `cand_ids`
(optional) keeps each candidate's tokens to itself inside a question block; see masks.py. There are no
rotary embeddings: image tokens carry SigLIP's learned 2D positions and text tokens a learned
position that restarts in every question block, so candidates that share a start position are
interchangeable.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from glance.model.masks import cached_text_mask, joint_mask


@dataclass
class TowerConfig:
    width: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    patch: int = 32
    image_size: int = 256
    vocab_size: int = 50368
    max_text_pos: int = 256
    modality_ffn: bool = True
    ln_eps: float = 1e-6

    @property
    def n_img(self) -> int:
        return (self.image_size // self.patch) ** 2


@dataclass
class ImageCache:
    """Per-layer image keys/values plus final image states for one or more images."""

    kv: list = field(default_factory=list)  # [(k, v)] each (B, H, n_img, head_dim)
    states: torch.Tensor | None = None


def _init_weights(m: nn.Module) -> None:
    """ViT/BERT-style init (std 0.02) for anything not inherited. PyTorch's default N(0, 1) for
    nn.Embedding made the new text position embedding ~20-50x larger than the inherited token
    embeddings in C3, drowning token identity under the pre-LayerNorm sum (found in R1)."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x, mask=None, kv_prefix=None, context=None):
        """Self-attention over x; `kv_prefix` prepends cached keys/values, `context` makes it
        cross-attention. Returns (output, this call's own (k, v))."""
        src = x if context is None else context
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(src)), self._split(self.v_proj(src))
        own = (k, v)
        if kv_prefix is not None:
            k = torch.cat([kv_prefix[0], k], dim=2)
            v = torch.cat([kv_prefix[1], v], dim=2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        B, H, N, Dh = o.shape
        return self.out_proj(o.transpose(1, 2).reshape(B, N, H * Dh)), own


class TowerLayer(nn.Module):
    def __init__(self, cfg: TowerConfig):
        super().__init__()
        d = cfg.width
        self.layer_norm1 = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.self_attn = Attention(d, cfg.heads)
        self.layer_norm2 = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.mlp = MLP(d, cfg.mlp_dim)
        if cfg.modality_ffn:
            self.layer_norm2_text = nn.LayerNorm(d, eps=cfg.ln_eps)
            self.mlp_text = MLP(d, cfg.mlp_dim)
        else:
            self.layer_norm2_text, self.mlp_text = self.layer_norm2, self.mlp

    def ffn_image(self, x):
        return x + self.mlp(self.layer_norm2(x))

    def ffn_text(self, x):
        return x + self.mlp_text(self.layer_norm2_text(x))

    def forward(self, x, n_img, mask):
        a, _ = self.self_attn(self.layer_norm1(x), mask)
        x = x + a
        return torch.cat([self.ffn_image(x[:, :n_img]), self.ffn_text(x[:, n_img:])], dim=1)

    def forward_image(self, x):
        a, kv = self.self_attn(self.layer_norm1(x))
        return self.ffn_image(x + a), kv

    def forward_text(self, x, kv_prefix, mask):
        a, _ = self.self_attn(self.layer_norm1(x), mask, kv_prefix=kv_prefix)
        return self.ffn_text(x + a)


class GlanceTower(nn.Module):
    """Single bidirectional tower over [image patches][question blocks] (C3, C4, C6)."""

    def __init__(self, cfg: TowerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.width
        self.patch_embedding = nn.Conv2d(3, d, cfg.patch, stride=cfg.patch)
        self.position_embedding = nn.Embedding(cfg.n_img, d)
        self.tok_embedding = nn.Embedding(cfg.vocab_size, d)
        self.text_pos_embedding = nn.Embedding(cfg.max_text_pos, d)
        self.text_embed_norm = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.layers = nn.ModuleList(TowerLayer(cfg) for _ in range(cfg.depth))
        self.post_layernorm = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.post_layernorm_text = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.apply(_init_weights)

    def embed_image(self, pixel_values):
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return x + self.position_embedding.weight[None]

    def embed_text(self, input_ids, position_ids):
        return self.text_embed_norm(self.tok_embedding(input_ids) + self.text_pos_embedding(position_ids))

    def encode_image(self, pixel_values) -> ImageCache:
        x = self.embed_image(pixel_values)
        cache = ImageCache()
        for layer in self.layers:
            x, kv = layer.forward_image(x)
            cache.kv.append(kv)
        cache.states = self.post_layernorm(x)
        return cache

    def encode_text(self, input_ids, position_ids, block_ids, cache: ImageCache, cand_ids=None):
        x = self.embed_text(input_ids, position_ids)
        mask = cached_text_mask(self.cfg.n_img, block_ids, cand_ids)
        for layer, kv in zip(self.layers, cache.kv):
            x = layer.forward_text(x, kv, mask)
        return self.post_layernorm_text(x)

    def forward(self, pixel_values, input_ids, position_ids, block_ids, cand_ids=None):
        n_img = self.cfg.n_img
        x = torch.cat([self.embed_image(pixel_values), self.embed_text(input_ids, position_ids)], dim=1)
        mask = joint_mask(n_img, block_ids, cand_ids)
        for layer in self.layers:
            x = layer(x, n_img, mask)
        return self.post_layernorm(x[:, :n_img]), self.post_layernorm_text(x[:, n_img:])


class FusionLayer(nn.Module):
    """Question self-attention (block-diagonal), cross-attention to image tokens, then FFN."""

    def __init__(self, d: int, heads: int, mlp_dim: int):
        super().__init__()
        self.norm_self = nn.LayerNorm(d)
        self.self_attn = Attention(d, heads)
        self.norm_cross = nn.LayerNorm(d)
        self.cross_attn = Attention(d, heads)
        self.norm_ffn = nn.LayerNorm(d)
        self.mlp = MLP(d, mlp_dim)

    def forward(self, x, image, self_mask):
        x = x + self.self_attn(self.norm_self(x), self_mask)[0]
        x = x + self.cross_attn(self.norm_cross(x), context=image)[0]
        return x + self.mlp(self.norm_ffn(x))


class FusionTower(nn.Module):
    """C5: frozen image tower + small text encoder + a few new cross-attention fusion layers.

    `image_tower` maps pixel values to (B, n_img, d_img) patch states; `text_encoder` is a HF
    ModernBERT-family encoder called with input_ids and per-layer-type block-diagonal masks. In
    training (C5) the image tower is frozen and cached outside, so it may be None there.
    """

    def __init__(self, image_tower: nn.Module, text_encoder: nn.Module, d_img: int, d_text: int,
                 n_fusion: int = 3, heads: int = 6):
        super().__init__()
        self.image_tower = image_tower
        self.text_encoder = text_encoder
        self.image_proj = nn.Linear(d_img, d_text)
        self.fusion = nn.ModuleList(FusionLayer(d_text, heads, 4 * d_text) for _ in range(n_fusion))
        self.final_norm = nn.LayerNorm(d_text)

    def encode_image(self, pixel_values) -> ImageCache:
        states = self.image_tower(pixel_values=pixel_values).last_hidden_state
        return ImageCache(states=self.image_proj(states))

    def encode_text(self, input_ids, position_ids, block_ids, cache: ImageCache, cand_ids=None):
        from glance.model.masks import modernbert_masks, text_block_mask

        self_mask = text_block_mask(block_ids, cand_ids)[:, None]
        masks = modernbert_masks(self_mask, self.text_encoder.config.local_attention, position_ids)
        x = self.text_encoder(input_ids=input_ids, position_ids=position_ids,
                              attention_mask=masks).last_hidden_state
        for layer in self.fusion:
            x = layer(x, cache.states, self_mask)
        return self.final_norm(x)

    def forward(self, pixel_values, input_ids, position_ids, block_ids, cand_ids=None):
        cache = self.encode_image(pixel_values)
        return cache.states, self.encode_text(input_ids, position_ids, block_ids, cache, cand_ids)


def load_siglip_vision(tower: GlanceTower, repo: str = "google/siglip2-base-patch32-256",
                       copy_to_text: bool = True) -> list[str]:
    """Copy a SigLIP vision encoder into `tower` (C3). With `copy_to_text`, the text FFNs and
    norms start as copies of the image ones (VLMo-style warm start). Returns keys left untouched."""
    import os

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    src = load_file(os.path.join(snapshot_download(repo, allow_patterns=["*.safetensors"]), "model.safetensors"))
    state = {}
    for k, v in src.items():
        if not k.startswith("vision_model.") or ".head." in k:
            continue
        k = k.removeprefix("vision_model.").removeprefix("embeddings.").removeprefix("encoder.")
        if k.startswith("layers.") and int(k.split(".")[1]) >= tower.cfg.depth:
            continue  # a shallower tower (C6s) inherits SigLIP's first `depth` layers
        state[k] = v
        if copy_to_text and tower.cfg.modality_ffn:
            for a, b in ((".mlp.", ".mlp_text."), (".layer_norm2.", ".layer_norm2_text."), ("post_layernorm.", "post_layernorm_text.")):
                if a in k or k.startswith(a):
                    state[k.replace(a, b)] = v.clone()
    missing, unexpected = tower.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    return missing


def siglip_b32_config() -> TowerConfig:
    return TowerConfig(width=768, depth=12, heads=12, mlp_dim=3072, patch=32, image_size=256)


def tiny_cpu_config() -> TowerConfig:
    return TowerConfig(width=512, depth=6, heads=8, mlp_dim=2048, patch=32, image_size=256)
