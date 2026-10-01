import torch

from glance.model.masks import joint_mask
from glance.model.towers import GlanceTower, TowerConfig

TINY = TowerConfig(width=64, depth=3, heads=4, mlp_dim=128, patch=32, image_size=64, vocab_size=100, max_text_pos=32)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    pixels = torch.randn(1, 3, 64, 64, generator=g)
    # Two questions: q0 = 3 tokens + two 2-token candidates sharing start position 3; q1 = 4 tokens.
    ids = torch.randint(0, 100, (1, 11), generator=g)
    pos = torch.tensor([[0, 1, 2, 3, 4, 3, 4, 0, 1, 2, 3]])
    blocks = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
    return pixels, ids, pos, blocks


def _model():
    torch.manual_seed(0)
    return GlanceTower(TINY).eval()


def test_mask_isolates_questions_and_shields_image():
    m = joint_mask(4, torch.tensor([[0, 0, 1, -1]]))[0, 0]
    assert m[:4, :4].all() and not m[:4, 4:].any()  # image sees only image
    assert m[4:6, :4].all() and m[4, 5] and not m[4, 6]  # q0 sees image + own block, not q1
    assert not m[7, :7].any() and m[7, 7]  # padding sees only itself


def test_cached_path_matches_joint_pass():
    model = _model()
    pixels, ids, pos, blocks = _inputs()
    with torch.no_grad():
        img_joint, txt_joint = model(pixels, ids, pos, blocks)
        cache = model.encode_image(pixels)
        txt_cached = model.encode_text(ids, pos, blocks, cache)
    torch.testing.assert_close(cache.states, img_joint, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(txt_cached, txt_joint, atol=1e-5, rtol=1e-5)


def test_questions_do_not_see_each_other():
    model = _model()
    pixels, ids, pos, blocks = _inputs()
    ids2 = ids.clone()
    ids2[0, 7:] = (ids2[0, 7:] + 1) % 100  # change only question 1
    with torch.no_grad():
        _, a = model(pixels, ids, pos, blocks)
        _, b = model(pixels, ids2, pos, blocks)
    torch.testing.assert_close(a[0, :7], b[0, :7])
    assert not torch.allclose(a[0, 7:], b[0, 7:])


def test_image_states_ignore_questions():
    model = _model()
    pixels, ids, pos, blocks = _inputs()
    with torch.no_grad():
        a, _ = model(pixels, ids, pos, blocks)
        b, _ = model(pixels, (ids + 7) % 100, pos, blocks)
    torch.testing.assert_close(a, b)


def test_candidates_sharing_positions_are_interchangeable():
    model = _model()
    pixels, ids, pos, blocks = _inputs()
    swapped = ids.clone()
    swapped[0, 3:5], swapped[0, 5:7] = ids[0, 5:7], ids[0, 3:5]
    with torch.no_grad():
        _, a = model(pixels, ids, pos, blocks)
        _, b = model(pixels, swapped, pos, blocks)
    torch.testing.assert_close(a[0, 3:5], b[0, 5:7], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(a[0, 5:7], b[0, 3:5], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(a[0, :3], b[0, :3], atol=1e-5, rtol=1e-5)


def test_padding_is_finite_and_inert():
    model = _model()
    pixels, ids, pos, blocks = _inputs()
    ids_p = torch.cat([ids, torch.zeros(1, 3, dtype=torch.long)], 1)
    pos_p = torch.cat([pos, torch.zeros(1, 3, dtype=torch.long)], 1)
    blocks_p = torch.cat([blocks, torch.full((1, 3), -1)], 1)
    with torch.no_grad():
        _, a = model(pixels, ids, pos, blocks)
        _, b = model(pixels, ids_p, pos_p, blocks_p)
    assert torch.isfinite(b).all()
    torch.testing.assert_close(a, b[:, :11], atol=1e-5, rtol=1e-5)


def test_siglip_b32_inheritance_reproduces_siglip():
    import pytest

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download("google/siglip2-base-patch32-256", local_files_only=True)
    except Exception:
        pytest.skip("SigLIP2-B/32 weights not cached")
    from transformers import SiglipVisionModel

    from glance.model.towers import load_siglip_vision, siglip_b32_config

    ref = SiglipVisionModel.from_pretrained(path, attn_implementation="sdpa").eval()
    tower = GlanceTower(siglip_b32_config()).eval()
    missing = load_siglip_vision(tower)
    assert all(k.startswith(("tok_embedding", "text_pos_embedding", "text_embed_norm")) for k in missing), missing
    pixels = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        want = ref(pixel_values=pixels).last_hidden_state
        got = tower.encode_image(pixels).states
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_candidate_tokens_see_question_and_own_candidate_only():
    blocks = torch.tensor([[0, 0, 0, 0, 0, 0]])
    cands = torch.tensor([[-1, -1, 0, 0, 1, 1]])
    m = joint_mask(2, blocks, cands)[0, 0][2:, 2:]  # text-to-text part
    assert m[0].all() and m[1].all()  # question tokens see the whole block
    assert m[2, :4].all() and not m[2, 4:].any()  # candidate 0 sees question + itself only
    assert m[4, :2].all() and not m[4, 2:4].any() and m[4, 4:].all()
