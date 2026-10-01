"""MLX ports of ModernBERT (Ettin), the C5 fusion tower and the C1/C2 ModernVBERT text path.

ModernBERT follows transformers 5.17 `modeling_modernbert.py`: layer 0 has no attention norm,
Wqkv/Wo and the GLU MLP (Wi -> [input, gate], act(input) * gate -> Wo) are bias-free when the config
says so, and RoPE uses a per-layer-type base (global vs local). RoPE is computed from explicit
position ids, since our blocks restart positions and candidates share them. Masks are passed as a
{"full_attention", "sliding_attention"} dict of additive masks, the band keyed on position ids
(see glance/model/masks.py). Parity with the torch/HF implementations is tested in tests/test_mlx.py.
"""
import mlx.core as mx
import mlx.nn as nn

from glance.mlx.tower import NEG, Attention, MLP, text_block_mask


def _additive(m):
    return mx.where(m, 0.0, NEG)


def modernbert_masks(bool_mask: mx.array, window: int, positions: mx.array) -> dict:
    """bool (B, N, N) -> additive (B, 1, N, N) dict for global and local layers."""
    band = mx.abs(positions[:, :, None] - positions[:, None, :]) <= window // 2
    return {"full_attention": _additive(bool_mask)[:, None], "sliding_attention": _additive(bool_mask & band)[:, None]}


def rope(x, positions, inv_freq):
    """x (B, H, N, Dh); positions (B, N); non-interleaved (rotate_half) RoPE."""
    freqs = positions[:, :, None].astype(mx.float32) * inv_freq[None, None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)[:, None]
    cos, sin = mx.cos(emb), mx.sin(emb)
    half = x.shape[-1] // 2
    rot = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


class MBAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, self.heads = cfg.hidden_size, cfg.num_attention_heads
        self.head_dim = d // self.heads
        self.Wqkv = nn.Linear(d, 3 * d, bias=cfg.attention_bias)
        self.Wo = nn.Linear(d, d, bias=cfg.attention_bias)

    def __call__(self, x, mask, positions, inv_freq):
        B, N, _ = x.shape
        qkv = self.Wqkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        q, k = rope(q, positions, inv_freq), rope(k, positions, inv_freq)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask=mask.astype(q.dtype))
        return self.Wo(o.transpose(0, 2, 1, 3).reshape(B, N, -1))


class MBMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=cfg.mlp_bias)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)
        assert cfg.hidden_activation in ("gelu", "gelu_python"), cfg.hidden_activation

    def __call__(self, x):
        inp, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(inp) * gate)


class MBLayer(nn.Module):
    def __init__(self, cfg, idx, attention_type):
        super().__init__()
        norm = lambda: nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)  # noqa: E731
        if idx != 0:
            self.attn_norm = norm()
        self.has_attn_norm = idx != 0
        self.attn = MBAttention(cfg)
        self.mlp_norm = norm()
        self.mlp = MBMLP(cfg)
        self.attention_type = attention_type

    def __call__(self, x, masks, positions, inv_freqs):
        h = self.attn_norm(x) if self.has_attn_norm else x
        x = x + self.attn(h, masks[self.attention_type], positions, inv_freqs[self.attention_type])
        return x + self.mlp(self.mlp_norm(x))


class MBEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)


class ModernBertMLX(nn.Module):
    """`layer_ids` keeps a subset of layers with their original index and type (C2 pruning)."""

    def __init__(self, cfg, layer_ids=None):
        super().__init__()
        layer_ids = list(range(cfg.num_hidden_layers)) if layer_ids is None else layer_ids
        self.embeddings = MBEmbeddings(cfg)
        self.layers = [MBLayer(cfg, i, cfg.layer_types[i]) for i in layer_ids]
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        dim = cfg.hidden_size // cfg.num_attention_heads
        self._inv_freqs = {t: 1.0 / (cfg.rope_parameters[t]["rope_theta"] ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
                           for t in set(cfg.layer_types)}
        self.window = cfg.local_attention

    def __call__(self, masks, positions, input_ids=None, inputs_embeds=None):
        x = self.embeddings.tok_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        x = self.embeddings.norm(x)
        for layer in self.layers:
            x = layer(x, masks, positions, self._inv_freqs)
        return self.final_norm(x)


class FusionLayerMLX(nn.Module):
    def __init__(self, d, heads, mlp_dim):
        super().__init__()
        self.norm_self, self.self_attn = nn.LayerNorm(d), Attention(d, heads)
        self.norm_cross, self.cross_attn = nn.LayerNorm(d), Attention(d, heads)
        self.norm_ffn, self.mlp = nn.LayerNorm(d), MLP(d, mlp_dim)

    def __call__(self, x, image, self_mask):
        x = x + self.self_attn(self.norm_self(x), self_mask)[0]
        x = x + self.cross_attn(self.norm_cross(x), None, kv_prefix=None, context=image)[0]
        return x + self.mlp(self.norm_ffn(x))


class FusionTowerMLX(nn.Module):
    """C5 per-question path: Ettin text encoder + fusion layers over cached image states."""

    def __init__(self, text_cfg, d_img=768, n_fusion=3, heads=6):
        super().__init__()
        d = text_cfg.hidden_size
        self.text_encoder = ModernBertMLX(text_cfg)
        self.image_proj = nn.Linear(d_img, d)
        self.fusion = [FusionLayerMLX(d, heads, 4 * d) for _ in range(n_fusion)]
        self.final_norm = nn.LayerNorm(d)

    def project_image(self, image_states):
        return self.image_proj(image_states)

    def encode_text(self, ids, pos, blocks, image, cands=None):
        m = text_block_mask(blocks, cands)
        x = self.text_encoder(modernbert_masks(m, self.text_encoder.window, pos), pos, input_ids=ids)
        self_mask = _additive(m)[:, None]
        for layer in self.fusion:
            x = layer(x, image, self_mask)
        return self.final_norm(x)
