import pytest
import torch
from transformers import AutoConfig, AutoModel

from glance.model.candidates import VBertImage, VBertText, build_candidate
from glance.model.masks import joint_mask, modernbert_masks


def _vbert_config():
    try:
        return AutoConfig.from_pretrained("ModernVBERT/modernvbert", local_files_only=True)
    except Exception:
        pytest.skip("ModernVBERT config not cached")


def test_vbert_text_path_matches_modernvbert_forward():
    cfg = _vbert_config()
    torch.manual_seed(0)
    vbert = AutoModel.from_config(cfg).eval()
    image = VBertImage(vbert).eval()
    text = VBertText(vbert.text_model).eval()
    pixels = torch.randn(1, 3, 512, 512)
    ids = torch.randint(100, 5000, (1, 12))
    pos = torch.arange(12)[None]
    blocks = torch.zeros(1, 12, dtype=torch.long)
    with torch.no_grad():
        img_tokens = image(pixels)
        ours = text(img_tokens, ids, pos, blocks)
        n_img = img_tokens.shape[1]
        full_ids = torch.cat([torch.full((1, n_img), cfg.image_token_id), ids], 1)
        masks = modernbert_masks(joint_mask(n_img, blocks), cfg.text_config.local_attention,
                                 torch.arange(n_img + ids.shape[1])[None])
        ref = vbert(input_ids=full_ids, pixel_values=pixels[:, None], attention_mask=masks).last_hidden_state[:, n_img:]
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_pruned_text_stack_keeps_even_layers():
    cfg = _vbert_config()
    vbert = AutoModel.from_config(cfg)
    layers = list(vbert.text_model.layers)
    text = VBertText(vbert.text_model, keep_layers=list(range(0, 22, 2)))
    assert len(text.text_model.layers) == 11
    assert all(a is b for a, b in zip(text.text_model.layers, layers[::2]))


@pytest.mark.parametrize("cid", ["C1", "C2", "C3", "C4", "C5", "C6", "C5u4", "C5f6"])
def test_candidates_build_and_run(cid):
    _vbert_config()
    from transformers import AutoTokenizer

    from glance.data.packing import collate, encode_question
    from glance.data.schema import Decision

    tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m", local_files_only=True)
    cand = build_candidate(cid).eval()
    d = Decision("u", "t", "choice", "what color is the car?", ["red", "blue", "green"], [1.0, 0.0, 0.0], b"", "i")
    batch = collate([[encode_question(tok, d)]], tok.pad_token_id)
    pixels = torch.randn(1, 3, cand.image_size, cand.image_size)
    with torch.no_grad():
        image_input = cand.image_features(pixels) if cand.cached_image else pixels
        logits = cand(image_input, batch)
    assert logits.shape == (3,) and torch.isfinite(logits).all()
    inh, new = cand.param_groups()
    n_inh, n_new = sum(p.numel() for p in inh), sum(p.numel() for p in new)
    if cid in ("C4", "C6"):
        assert n_inh == 0
    else:
        assert n_inh > 0
    assert n_new > 0  # at least the head
    print(cid, f"inherited-trainable {n_inh / 1e6:.1f}M new-trainable {n_new / 1e6:.1f}M")


def test_packing_order_does_not_change_answers_c2():
    """R1 audit F4: with the local window keyed on sequence index, a question packed later saw
    less of the image in ModernBERT's local layers. Keyed on position ids, order is irrelevant."""
    cfg = _vbert_config()
    torch.manual_seed(0)
    vbert = AutoModel.from_config(cfg).eval()
    text = VBertText(vbert.text_model, keep_layers=list(range(0, 22, 2))).eval()
    img = torch.randn(1, 64, 768)
    qa = torch.randint(100, 5000, (40,))
    qb = torch.randint(100, 5000, (90,))

    def run(first, second):
        ids = torch.cat([first, second])[None]
        pos = torch.cat([torch.arange(len(first)), torch.arange(len(second))])[None]
        blocks = torch.cat([torch.zeros(len(first)), torch.ones(len(second))]).long()[None]
        with torch.no_grad():
            return text(img, ids, pos, blocks)[0]

    ab, ba = run(qa, qb), run(qb, qa)
    torch.testing.assert_close(ab[:40], ba[90:], atol=1e-4, rtol=1e-4)


def test_load_trained_refuses_mismatched_checkpoints():
    from glance.model.candidates import load_trained

    cand = build_candidate("C5")
    state = {n: p.detach().clone() for n, p in cand.named_parameters() if p.requires_grad}
    load_trained(build_candidate("C5"), state)  # a matching checkpoint loads
    with pytest.raises(ValueError, match="1 unexpected"):
        load_trained(build_candidate("C5"), {**state, "tower.fusion.3.mlp.fc1.weight": torch.zeros(1)})
    first = next(iter(state))
    with pytest.raises(ValueError, match="1 trainable tensors missing"):
        load_trained(build_candidate("C5"), {k: v for k, v in state.items() if k != first})


def _c5_batch():
    from transformers import AutoTokenizer

    from glance.data.packing import collate, encode_question
    from glance.data.schema import Decision

    tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m", local_files_only=True)
    ds = [Decision("u", "t", "noul", "Is there a dog in the image?", ["yes", "no"], [1.0, 0.0], b"", "i"),
          Decision("v", "t", "choice", "what color is the car?", ["red", "blue", "green"], [1.0, 0.0, 0.0], b"", "i")]
    return collate([[encode_question(tok, d) for d in ds]], tok.pad_token_id)


def test_siglip_split_reproduces_the_full_encoder():
    from transformers import SiglipVisionModel

    from glance.model.candidates import SiglipImage

    vision = SiglipVisionModel.from_pretrained("google/siglip2-base-patch32-256").eval()
    px = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        full = vision(pixel_values=px).last_hidden_state
        for k in (1, 4, 8):
            split = SiglipImage(vision, k)
            assert torch.allclose(split.top(split.bottom(px)), full, atol=1e-4), k


def test_c5u4_trains_only_the_top_image_layers_at_the_inherited_rate():
    cand = build_candidate("C5u4")
    trainable = [n for n, p in cand.named_parameters() if p.requires_grad and n.startswith("frozen_image.")]
    layers = sorted({int(n.split(".")[4]) for n in trainable if ".encoder.layers." in n})
    assert layers == [8, 9, 10, 11]
    assert any(n.startswith("frozen_image.vision.post_layernorm.") for n in trainable)
    assert not any(".embeddings." in n or ".head." in n for n in trainable)
    assert set(trainable) <= cand.inherited  # inherited learning rate, not the new-layer rate
    assert cand.cache_key == "C5u4_256" and build_candidate("C5f6").cache_key == "C5_256"
    assert len(build_candidate("C5f6").tower.fusion) == 6 and len(build_candidate("C5").tower.fusion) == 3


def test_c5u4_training_step_reaches_the_top_layers_only():
    from glance.data.packing import collate  # noqa: F401  (batch helper imports it)
    from glance.model.heads import decision_loss, pad_by_question

    cand = build_candidate("C5u4").train()
    batch = _c5_batch()
    px = torch.randn(1, 3, 256, 256)
    cached = cand.cache_features(px).half().float()
    logits = cand(cand.image_top(cached), batch)
    padded = pad_by_question(logits, batch["slot_q"], batch["slot_k"], len(batch["q_pos"]), batch["k_max"])
    loss, _ = decision_loss(padded, batch["targets"], batch["is_score"])
    loss.backward()
    grads = {n: p.grad for n, p in cand.named_parameters() if n.startswith("frozen_image.vision.encoder.layers.")}
    assert grads["frozen_image.vision.encoder.layers.11.mlp.fc2.weight"].abs().sum() > 0
    assert grads["frozen_image.vision.encoder.layers.0.mlp.fc2.weight"] is None


def test_c5u4_checkpoint_round_trip_and_under_the_c5_id():
    from glance.model.candidates import load_trained

    cand = build_candidate("C5u4").eval()
    with torch.no_grad():  # perturb the trainable top so a silent fallback to pretrained weights would show
        cand.frozen_image.vision.encoder.layers[11].mlp.fc2.weight.add_(0.05)
    state = {n: p.detach().clone() for n, p in cand.named_parameters() if p.requires_grad}
    batch, px = _c5_batch(), torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        want = cand(cand.image_features(px), batch)
        again = load_trained(build_candidate("C5u4").eval(), state)
        assert torch.equal(again(again.image_features(px), batch), want)
        # the trained top layers exist (frozen) in plain C5, so they load there too, and serve identically
        as_c5 = load_trained(build_candidate("C5").eval(), state)
        assert torch.allclose(as_c5(as_c5.image_features(px), batch), want, atol=1e-5)
