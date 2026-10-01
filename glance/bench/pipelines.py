"""Real-shape builders for every candidate and reference, for latency measurement.

Weights are random (latency does not depend on their values), and the architecture comes from
the actual config. Each builder returns a Pipeline with a `cold` call (image + questions) and a
`cached` call (questions only, image work reused). The workload is n_q questions of 48 text
tokens each: a 24-token instruction plus K=4 candidates of 6 tokens.
"""
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoConfig, AutoModel

from glance.model.masks import text_block_mask
from glance.model.towers import FusionTower, GlanceTower, siglip_b32_config, tiny_cpu_config

Q_LEN, K, C_LEN = 24, 4, 6


@dataclass
class Pipeline:
    name: str
    cold: Callable
    cached: Callable
    depth_cold: int
    depth_cached: int
    bytes_cold: float
    bytes_cached: float
    params: int
    note: str = ""


def text_workload(n_q: int, vocab: int, device):
    g = torch.Generator().manual_seed(n_q)
    ids, pos, blocks = [], [], []
    for qi in range(n_q):
        ids += torch.randint(5, vocab - 5, (Q_LEN + K * C_LEN,), generator=g).tolist()
        pos += list(range(Q_LEN)) + [Q_LEN + j for _ in range(K) for j in range(C_LEN)]
        blocks += [qi] * (Q_LEN + K * C_LEN)
    t = lambda x: torch.tensor([x], device=device)
    return t(ids), t(pos), t(blocks)


def _bytes(modules, dtype) -> float:
    """Weight bytes streamed per call; embedding tables are excluded (only a few rows are read)."""
    size = torch.tensor([], dtype=dtype).element_size()
    seen, n = set(), 0
    for m in modules:
        for sub in m.modules():
            if isinstance(sub, torch.nn.Embedding):
                seen.update(id(p) for p in sub.parameters())
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
    return float(n * size)


def _nparams(m) -> int:
    return sum(p.numel() for p in m.parameters())


def glance_tower(name, cfg, n_q, device, dtype) -> Pipeline:
    model = GlanceTower(cfg).to(device, dtype).eval()
    pixels = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device, dtype=dtype)
    ids, pos, blocks = text_workload(n_q, cfg.vocab_size, device)
    with torch.inference_mode():
        cache = model.encode_image(pixels)
    text_side = [m for layer in model.layers for m in (layer.layer_norm1, layer.self_attn, layer.layer_norm2_text, layer.mlp_text)]
    return Pipeline(
        name, cold=lambda: model(pixels, ids, pos, blocks),
        cached=lambda: model.encode_text(ids, pos, blocks, cache),
        depth_cold=cfg.depth, depth_cached=cfg.depth,
        bytes_cold=_bytes([model], dtype), bytes_cached=_bytes(text_side, dtype), params=_nparams(model))


def fusion_tower(n_q, device, dtype) -> Pipeline:
    from transformers import SiglipVisionConfig, SiglipVisionModel

    vcfg = SiglipVisionConfig(hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                              num_attention_heads=12, image_size=256, patch_size=32, vision_use_head=False)
    tcfg = AutoConfig.from_pretrained("jhu-clsp/ettin-encoder-32m")
    model = FusionTower(SiglipVisionModel(vcfg), AutoModel.from_config(tcfg), d_img=768, d_text=tcfg.hidden_size,
                        n_fusion=3, heads=6).to(device, dtype).eval()
    pixels = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
    ids, pos, blocks = text_workload(n_q, tcfg.vocab_size, device)
    with torch.inference_mode():
        cache = model.encode_image(pixels)
    text_side = [model.text_encoder, model.fusion, model.final_norm]
    return Pipeline(
        "C5 fusion (B/32 + Ettin-32M + 3 fusion)", cold=lambda: model(pixels, ids, pos, blocks),
        cached=lambda: model.encode_text(ids, pos, blocks, cache),
        depth_cold=12 + tcfg.num_hidden_layers + 3, depth_cached=tcfg.num_hidden_layers + 3,
        bytes_cold=_bytes([model], dtype), bytes_cached=_bytes(text_side, dtype), params=_nparams(model))


def modernvbert(n_q, device, dtype, text_layers=None, res=512) -> Pipeline:
    cfg = AutoConfig.from_pretrained("ModernVBERT/modernvbert")
    if text_layers:
        cfg.text_config.num_hidden_layers = text_layers
    model = AutoModel.from_config(cfg).to(device, dtype).eval()
    tm = model.text_model
    pixels = torch.randn(1, 3, res, res, device=device, dtype=dtype)
    ids, pos, blocks = text_workload(n_q, cfg.text_config.vocab_size, device)

    def image_tokens():
        feats = model.vision_model(pixel_values=pixels, interpolate_pos_encoding=res != 512).last_hidden_state
        return model.connector(feats)

    with torch.inference_mode():
        img = image_tokens()
    n_img = img.shape[1]
    all_blocks = torch.cat([torch.full((1, n_img), 10_000, device=device), blocks], 1)
    mask = text_block_mask(all_blocks)[:, None]  # image block + isolated questions (latency is mask-independent)
    all_pos = torch.cat([torch.arange(n_img, device=device)[None], pos + n_img], 1)

    def text(img_tokens):
        x = torch.cat([img_tokens, tm.get_input_embeddings()(ids)], 1)
        return tm(inputs_embeds=x, attention_mask=mask, position_ids=all_pos).last_hidden_state

    L = cfg.text_config.num_hidden_layers
    name = f"C1 ModernVBERT @{res}px" if not text_layers else f"C2 ModernVBERT text {L}L @{res}px"
    return Pipeline(
        name, cold=lambda: text(image_tokens()), cached=lambda: text(img),
        depth_cold=12 + L, depth_cached=L, bytes_cold=_bytes([model], dtype), bytes_cached=_bytes([tm], dtype),
        params=_nparams(model), note=f"{n_img} image tokens; cached path still runs them through the text stack")


def smolvlm_causal(n_q, device, dtype) -> Pipeline:
    from transformers import Idefics3ForConditionalGeneration

    cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    model = Idefics3ForConditionalGeneration(cfg).to(device, dtype).eval()
    inner, tm = model.model, model.model.text_model
    pixels = torch.randn(1, 3, 512, 512, device=device, dtype=dtype)
    ids, _, _ = text_workload(n_q, cfg.text_config.vocab_size, device)

    def image_tokens():
        return inner.connector(inner.vision_model(pixel_values=pixels).last_hidden_state)

    with torch.inference_mode():
        img = image_tokens()
        prefix = tm(inputs_embeds=img, use_cache=True).past_key_values

    def cold():
        x = torch.cat([image_tokens(), tm.get_input_embeddings()(ids)], 1)
        return model.lm_head(tm(inputs_embeds=x).last_hidden_state[:, -1])

    def cached():
        out = tm(inputs_embeds=tm.get_input_embeddings()(ids), past_key_values=prefix, use_cache=True)
        prefix.crop(-ids.shape[1])
        return model.lm_head(out.last_hidden_state[:, -1])

    L = cfg.text_config.num_hidden_layers
    return Pipeline(
        "C0 SmolVLM-256M causal readout", cold=cold, cached=cached, depth_cold=12 + L, depth_cached=L,
        bytes_cold=_bytes([model], dtype), bytes_cached=_bytes([tm, model.lm_head], dtype), params=_nparams(model),
        note="questions encoded jointly (laya-style); cached = image KV prefix in the LM")


def siglip_dual(n_q, device, dtype) -> Pipeline:
    from transformers import SiglipModel

    cfg = AutoConfig.from_pretrained("google/siglip2-base-patch32-256")
    model = SiglipModel(cfg).to(device, dtype).eval()
    pixels = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
    texts = torch.randint(5, cfg.text_config.vocab_size - 5, (n_q * K, 64), device=device)

    def image():
        return model.vision_model(pixel_values=pixels).pooler_output

    def text():
        return model.text_model(input_ids=texts).pooler_output

    return Pipeline(
        "C0 SigLIP2-B/32 dual encoder", cold=lambda: (image(), text()), cached=text,
        depth_cold=24, depth_cached=12, bytes_cold=_bytes([model], dtype),
        bytes_cached=_bytes([model.text_model], dtype), params=_nparams(model),
        note="candidate texts padded to 64 tokens; fixed label sets could cache these too")


def qwen3vl_teacher(n_q, device, dtype) -> Pipeline:
    from transformers import Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    model = Qwen3VLForConditionalGeneration(cfg).to(device, dtype).eval()
    inner, lm = model.model, model.model.language_model
    grid = torch.tensor([[1, 16, 16]], device=device)  # 256 px, patch 16 -> 64 tokens after 2x2 merge
    pixels = torch.randn(256, 3 * 2 * 16 * 16, device=device, dtype=dtype)
    ids, _, _ = text_workload(n_q, cfg.text_config.vocab_size, device)

    def image_tokens():
        out = inner.visual(pixels, grid_thw=grid)
        return (out[0] if isinstance(out, tuple) else getattr(out, "pooler_output", out.last_hidden_state))[None]

    def cold():
        x = torch.cat([image_tokens(), lm.get_input_embeddings()(ids)], 1)
        return model.lm_head(lm(inputs_embeds=x).last_hidden_state[:, -1])

    with torch.inference_mode():
        img = image_tokens()

    def cached():
        x = torch.cat([img, lm.get_input_embeddings()(ids)], 1)
        return model.lm_head(lm(inputs_embeds=x).last_hidden_state[:, -1])

    L = cfg.text_config.num_hidden_layers
    return Pipeline(
        "teacher Qwen3-VL-2B readout", cold=cold, cached=cached, depth_cold=cfg.vision_config.depth + L, depth_cached=L,
        bytes_cold=_bytes([model], dtype), bytes_cached=_bytes([lm, model.lm_head], dtype), params=_nparams(model),
        note=f"{img.shape[1]} image tokens; deepstack features ignored")


def build_all(n_q: int, device, dtype, include_teacher: bool = True):
    yield lambda: glance_tower("C3/C4 single tower 12x768 (B/32 shape)", siglip_b32_config(), n_q, device, dtype)
    yield lambda: glance_tower("C6 tiny tower 6x512", tiny_cpu_config(), n_q, device, dtype)
    yield lambda: fusion_tower(n_q, device, dtype)
    yield lambda: modernvbert(n_q, device, dtype)
    yield lambda: modernvbert(n_q, device, dtype, res=256)
    yield lambda: modernvbert(n_q, device, dtype, text_layers=11)
    yield lambda: siglip_dual(n_q, device, dtype)
    yield lambda: smolvlm_causal(n_q, device, dtype)
    if include_teacher:
        yield lambda: qwen3vl_teacher(n_q, device, dtype)
