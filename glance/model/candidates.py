"""Build each R1 candidate as tower + pointer head behind one interface.

Every candidate turns (image input, packed text) into per-token text states; the shared
PointerHead turns slot states into decision logits. What differs is the tower, what is inherited,
what is frozen, and what the image input is:

  C1  ModernVBERT (SigLIP2-B/16 @512 -> 64 tokens -> Ettin-150M, 22L). Vision + connector frozen,
      so the image input is the cached connector output. Text stack trains. Quality reference.
  C2  C1 with the text stack pruned to its 11 even layers, fed 256 px (16 image tokens; reshaped
      in R0 to meet the CPU budget). Cached image tokens; text stack trains.
  C3  GlanceTower 12x768 seeded from SigLIP2-B/32 (vision) and ModernVBERT's text embeddings;
      everything trains; image input is pixels.
  C4  C3's shape at random init; everything trains.
  C5  Frozen SigLIP2-B/32 (cached 64x768) + Ettin-32M (trains) + 3 new fusion layers.
      C5u<k>: SigLIP's top k layers train too (training caches the frozen bottom's output; served
      and evaluated like C5, one image encoding per image). C5f<n>: n fusion layers.
  C6  GlanceTower 6x512 at random init; everything trains.

The image always attends only to itself (deferred fusion at k = L), so every candidate's image
work is question-independent.
"""
import re

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from glance.model.heads import PointerHead
from glance.model.masks import joint_mask, modernbert_masks
from glance.model.towers import (FusionTower, GlanceTower, ImageCache, TowerConfig, load_siglip_vision,
                                 siglip_b32_config, tiny_cpu_config)

MEAN, STD = 0.5, 0.5  # SigLIP / ModernVBERT normalisation


def normalize(images_u8: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float in SigLIP's range."""
    return (images_u8.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD


class VBertText(nn.Module):
    """ModernVBERT's text stack over [image tokens][question blocks] with isolation masks."""

    def __init__(self, text_model, keep_layers=None):
        super().__init__()
        if keep_layers is not None:
            text_model.layers = nn.ModuleList(text_model.layers[i] for i in keep_layers)
            text_model.config.num_hidden_layers = len(keep_layers)
            if getattr(text_model.config, "layer_types", None):
                text_model.config.layer_types = [text_model.config.layer_types[i] for i in keep_layers]
        self.text_model = text_model
        self.window = text_model.config.local_attention

    def forward(self, img_tokens, input_ids, position_ids, block_ids, cand_ids=None):
        n_img = img_tokens.shape[1]
        emb = self.text_model.get_input_embeddings()(input_ids)
        x = torch.cat([img_tokens.to(emb.dtype), emb], dim=1)
        img_pos = torch.arange(n_img, device=input_ids.device)[None].expand(input_ids.shape[0], -1)
        pos = torch.cat([img_pos, position_ids + n_img], dim=1)
        masks = modernbert_masks(joint_mask(n_img, block_ids, cand_ids), self.window, pos)
        return self.text_model(inputs_embeds=x, attention_mask=masks, position_ids=pos).last_hidden_state[:, n_img:]


class Candidate(nn.Module):
    def __init__(self, cid, width, image_size, cached_image, tower, frozen_image=None):
        super().__init__()
        self.cid, self.image_size, self.cached_image = cid, image_size, cached_image
        self.tower = tower
        self.frozen_image = frozen_image  # used to build / recompute the image cache
        if frozen_image is not None:
            frozen_image.requires_grad_(False)
        self.head = PointerHead(width)
        self.inherited: set[str] = set()  # parameter names that start from pretrained weights
        self.cache_key = f"{cid}_{image_size}"  # training feature cache file; candidates that cache the same thing share it

    @torch.no_grad()
    def cache_features(self, pixels: torch.Tensor) -> torch.Tensor:
        """The frozen part of the image path, which training caches per image (stored as fp16)."""
        f = self.frozen_image
        return f.bottom(pixels) if getattr(f, "top_k", 0) else f(pixels)

    def image_top(self, x: torch.Tensor) -> torch.Tensor:
        """The trainable top of the image path, applied to cached features during training; the
        identity when the whole image path is frozen."""
        f = self.frozen_image
        return f.top(x) if getattr(f, "top_k", 0) else x

    @torch.no_grad()
    def image_features(self, pixels: torch.Tensor) -> torch.Tensor:
        """Whole image path for cached candidates: pixels -> tokens the tower consumes. The frozen
        part is rounded to fp16 first, exactly as training reads it from the cache."""
        return self.image_top(self.cache_features(pixels).half().float())

    def text_states(self, image_input, input_ids, position_ids, block_ids, cand_ids):
        if self.cid in ("C3", "C4", "C6", "C6s"):
            _, txt = self.tower(image_input, input_ids, position_ids, block_ids, cand_ids)
            return txt
        if isinstance(self.tower, FusionTower):
            return self.tower.encode_text(input_ids, position_ids, block_ids,
                                          ImageCache(states=self.tower.image_proj(image_input)), cand_ids)
        return self.tower(image_input, input_ids, position_ids, block_ids, cand_ids)

    def forward(self, image_input, batch):
        states = self.text_states(image_input, batch["input_ids"], batch["position_ids"], batch["block_ids"],
                                  batch["cand_ids"])
        return self.head(states, batch["q_pos"], batch["slot_pos"], batch["slot_q"],
                         batch.get("cand_tok_pos"), batch.get("cand_tok_slot"))

    def param_groups(self):
        inh, new = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                (inh if n in self.inherited else new).append(p)
        return inh, new


class EarlyFusionSmall(nn.Module):
    """C7: frozen image tokens projected into a small text encoder's input, which then attends to them
    in every layer (C1's early fusion, at C5's size): [image tokens][question blocks], joint mask."""

    def __init__(self, text_model, d_img: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_img, text_model.config.hidden_size), nn.LayerNorm(text_model.config.hidden_size))
        self.text = VBertText(text_model)

    def forward(self, img_tokens, input_ids, position_ids, block_ids, cand_ids=None):
        return self.text(self.proj(img_tokens), input_ids, position_ids, block_ids, cand_ids)


class VBertImage(nn.Module):
    def __init__(self, vbert):
        super().__init__()
        self.vision, self.connector = vbert.vision_model, vbert.connector

    def forward(self, pixels):
        return self.connector(self.vision(pixel_values=pixels, interpolate_pos_encoding=pixels.shape[-1] != 512).last_hidden_state)


class SiglipImage(nn.Module):
    """SigLIP vision encoder -> patch tokens (post-layernorm). With top_k > 0 the path splits into
    `bottom` (embeddings + the first L - top_k layers) and `top` (the last top_k layers + the final
    layernorm), so the top can train on cached bottom outputs."""

    def __init__(self, vision, top_k: int = 0):
        super().__init__()
        self.vision, self.top_k = vision, top_k

    def forward(self, pixels):
        return self.vision(pixel_values=pixels).last_hidden_state

    def bottom(self, pixels):
        h = self.vision.embeddings(pixels)
        for layer in self.vision.encoder.layers[:len(self.vision.encoder.layers) - self.top_k]:
            h = layer(h, None)
        return h

    def top(self, h):
        for layer in self.vision.encoder.layers[len(self.vision.encoder.layers) - self.top_k:]:
            h = layer(h, None)
        return self.vision.post_layernorm(h)


def read_state(path) -> dict:
    """A trained-tensors checkpoint as a state dict: .safetensors (the published format) or a torch .pt
    state dict, read with weights_only=True so no pickled code runs."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_trained(cand: Candidate, state: dict, name: str = "checkpoint") -> Candidate:
    """Load a trained-tensors state dict (train.py saves only the tensors that require grad) and refuse
    a mismatch either way: a trainable tensor the state lacks, or a state tensor the candidate has no
    place for. Plain load_state_dict(strict=False) drops the latter silently, so a checkpoint loaded
    under the wrong candidate id would run with some of its trained weights missing."""
    params = dict(cand.named_parameters())
    missing = [n for n, p in params.items() if p.requires_grad and n not in state]
    unexpected = [k for k in state if k not in params and k not in dict(cand.named_buffers())]
    if missing or unexpected:
        raise ValueError(f"{name} does not match candidate {cand.cid}: {len(missing)} trainable tensors missing"
                         f"{' (e.g. ' + missing[0] + ')' if missing else ''}, {len(unexpected)} unexpected"
                         f"{' (e.g. ' + unexpected[0] + ')' if unexpected else ''}")
    cand.load_state_dict(state, strict=False)
    return cand


C9_LAYERS = [0, 4, 8, 12, 16, 21]  # ModernVBERT text layers C9 keeps (spread over the stack, last one kept)


def c9_init_from_c1(state: dict) -> dict:
    """A trained C1 state dict with its text layers renumbered to C9's kept subset."""
    out = {}
    for k, v in state.items():
        if k.startswith("tower.text_model.layers."):
            i = int(k.split(".")[3])
            if i not in C9_LAYERS:
                continue
            k = k.replace(f"tower.text_model.layers.{i}.", f"tower.text_model.layers.{C9_LAYERS.index(i)}.", 1)
        out[k] = v
    return out


def _mark(cand: Candidate, prefix: str, names):
    cand.inherited |= {f"{prefix}{n}" for n in names}


def build_candidate(cid: str, pretrained: bool = True, base: dict | None = None) -> Candidate:
    """`base` (C5 family only): {"vision": dir, "text": dir} of saved configs; the base models are then
    built empty from those configs, with no Hub download, for a release package that holds every weight."""
    if cid in ("C1", "C2", "C9"):  # C9: C1 cut to C5's cost: 192 px (9 image tokens), 6 of 22 text layers
        vbert = AutoModel.from_pretrained("ModernVBERT/modernvbert") if pretrained else \
            AutoModel.from_config(AutoConfig.from_pretrained("ModernVBERT/modernvbert"))
        keep = {"C1": None, "C2": list(range(0, 22, 2)), "C9": C9_LAYERS}[cid]
        text = VBertText(vbert.text_model, keep)
        cand = Candidate(cid, 768, {"C1": 512, "C2": 256, "C9": 192}[cid], True, text, VBertImage(vbert))
        _mark(cand, "tower.", [n for n, _ in text.named_parameters()])
        return cand
    if cid in ("C3", "C4", "C6", "C6s"):
        cfg = tiny_cpu_config() if cid == "C6" else siglip_b32_config()
        if cid == "C6s":  # half-depth single tower that keeps the visual prior: SigLIP2-B/32's first 6 layers
            cfg = TowerConfig(width=768, depth=6, heads=12, mlp_dim=3072, patch=32, image_size=256)
        tower = GlanceTower(cfg)
        cand = Candidate(cid, cfg.width, cfg.image_size, False, tower)
        if cid in ("C3", "C6s") and pretrained:
            missing = load_siglip_vision(tower)
            vbert_text = AutoModel.from_pretrained("ModernVBERT/modernvbert").text_model
            with torch.no_grad():
                tower.tok_embedding.weight.copy_(vbert_text.embeddings.tok_embeddings.weight[: cfg.vocab_size])
                tower.text_embed_norm.weight.copy_(vbert_text.embeddings.norm.weight)
                if vbert_text.embeddings.norm.bias is not None:
                    tower.text_embed_norm.bias.copy_(vbert_text.embeddings.norm.bias)
            # The text FFNs/norms start as copies of the vision ones but serve a new modality, so
            # they train at the new-parameter LR (R1 audit F6); the plan calls them "text FFN new".
            copies = ("mlp_text", "layer_norm2_text", "post_layernorm_text")
            inherited = [n for n, _ in tower.named_parameters()
                         if (n not in missing or n.startswith(("tok_embedding", "text_embed_norm")))
                         and not any(c in n for c in copies)]
            _mark(cand, "tower.", inherited)
        return cand
    if cid == "C7":  # early fusion at C5's size: SigLIP2-B/32 @256 (frozen, 64 tokens) into every Ettin-32M layer
        from transformers import SiglipVisionModel

        vision = SiglipVisionModel.from_pretrained("google/siglip2-base-patch32-256")
        text = AutoModel.from_pretrained("jhu-clsp/ettin-encoder-32m")
        tower = EarlyFusionSmall(text, d_img=768)
        cand = Candidate(cid, text.config.hidden_size, 256, True, tower, SiglipImage(vision))
        _mark(cand, "tower.text.text_model.", [n for n, _ in text.named_parameters()])
        return cand
    c5 = re.fullmatch(r"C5(?:u(\d+))?(?:f(\d+))?", cid)
    if c5 or cid == "C8":  # C8: C5 with SigLIP2-B/16 @256 (256 image tokens instead of 64)
        # C5 variants: C5u<k> trains SigLIP's top k layers (the inference shape is C5's: the image is
        # still encoded once and cached); C5f<n> has n fusion layers instead of 3.
        from transformers import SiglipVisionModel

        top_k, n_fusion = (int(c5.group(1) or 0), int(c5.group(2) or 3)) if c5 else (0, 3)
        if not (0 <= top_k <= 12 and n_fusion >= 1):
            raise ValueError(f"{cid}: need 0 <= k <= 12 SigLIP layers to train and at least 1 fusion layer")
        if base:
            from transformers import SiglipVisionConfig

            vision = SiglipVisionModel(SiglipVisionConfig.from_pretrained(base["vision"]))
            text = AutoModel.from_config(AutoConfig.from_pretrained(base["text"]))
        else:
            vision = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-256" if cid == "C8" else
                                                       "google/siglip2-base-patch32-256")
            text = AutoModel.from_pretrained("jhu-clsp/ettin-encoder-32m")
        fusion = FusionTower(vision, text, d_img=768, d_text=text.config.hidden_size, n_fusion=n_fusion, heads=6)
        fusion.image_tower = None  # frozen and cached outside the trainable tower
        cand = Candidate(cid, text.config.hidden_size, 256, True, fusion, SiglipImage(vision, top_k))
        _mark(cand, "tower.text_encoder.", [n for n, _ in text.named_parameters()])
        if top_k:
            layers = vision.encoder.layers
            trainable = [f"encoder.layers.{i}.{n}" for i in range(len(layers) - top_k, len(layers))
                         for n, _ in layers[i].named_parameters()] + [f"post_layernorm.{n}" for n, _ in vision.post_layernorm.named_parameters()]
            params = dict(vision.named_parameters())
            for n in trainable:
                params[n].requires_grad_(True)
            _mark(cand, "frozen_image.vision.", trainable)
            cand.cache_key = f"C5u{top_k}_256"  # cached: the output of the frozen bottom layers
        elif cid != "C8":
            cand.cache_key = "C5_256"  # every C5f<n> caches the same final SigLIP tokens as C5
        return cand
    raise ValueError(cid)
