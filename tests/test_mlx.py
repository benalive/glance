import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from glance.mlx.tower import GlanceTowerMLX, PointerHeadMLX, from_torch  # noqa: E402
from glance.model.heads import PointerHead  # noqa: E402
from glance.model.towers import GlanceTower, TowerConfig  # noqa: E402

CFG = TowerConfig(width=64, depth=3, heads=4, mlp_dim=128, patch=32, image_size=64, vocab_size=100, max_text_pos=32)


def _inputs():
    g = torch.Generator().manual_seed(0)
    pixels = torch.randn(2, 3, 64, 64, generator=g)
    ids = torch.randint(0, 100, (2, 11), generator=g)
    pos = torch.tensor([[0, 1, 2, 3, 4, 3, 4, 0, 1, 2, 3]] * 2)
    blocks = torch.tensor([[0] * 7 + [1] * 4, [0] * 7 + [-1] * 4])
    cands = torch.tensor([[-1, -1, -1, 0, 0, 1, 1, -1, -1, -1, -1]] * 2)
    return pixels, ids, pos, blocks, cands


def _pair():
    torch.manual_seed(0)
    tt = GlanceTower(CFG).eval()
    th = PointerHead(64, 16).eval()
    mt, mh = GlanceTowerMLX(CFG), PointerHeadMLX(64, 16)
    mt.load_weights(from_torch(tt.state_dict()))
    mh.load_weights(from_torch(th.state_dict()))
    return tt, th, mt, mh


def _mx(t):
    return mx.array(t.numpy())


def test_mlx_tower_matches_torch():
    tt, th, mt, mh = _pair()
    pixels, ids, pos, blocks, cands = _inputs()
    with torch.no_grad():
        ti, tx = tt(pixels, ids, pos, blocks, cands)
    mi, mxt = mt(_mx(pixels), _mx(ids), _mx(pos), _mx(blocks), _mx(cands))
    np.testing.assert_allclose(np.array(mi), ti.numpy(), atol=1e-4, rtol=1e-4)
    # padded positions (block -1) are not compared
    valid = (blocks >= 0).numpy()
    np.testing.assert_allclose(np.array(mxt)[valid], tx.numpy()[valid], atol=1e-4, rtol=1e-4)


def test_mlx_cached_path_matches_joint():
    _, _, mt, _ = _pair()
    pixels, ids, pos, blocks, cands = _inputs()
    _, joint = mt(_mx(pixels), _mx(ids), _mx(pos), _mx(blocks), _mx(cands))
    kv, _ = mt.encode_image(_mx(pixels))
    cached = mt.encode_text(_mx(ids), _mx(pos), _mx(blocks), kv, _mx(cands))
    valid = (blocks >= 0).numpy()
    np.testing.assert_allclose(np.array(cached)[valid], np.array(joint)[valid], atol=1e-4, rtol=1e-4)


def test_mlx_head_matches_torch():
    tt, th, mt, mh = _pair()
    pixels, ids, pos, blocks, cands = _inputs()
    q_pos = torch.tensor([[0, 0], [0, 7], [1, 0]])
    slot_pos = torch.tensor([[0, 4], [0, 6], [0, 9], [0, 10], [1, 4], [1, 6]])
    slot_q = torch.tensor([0, 0, 1, 1, 2, 2])
    tok_pos = torch.tensor([[0, 3], [0, 4], [0, 5], [0, 6], [0, 9], [0, 10], [1, 3], [1, 4], [1, 5], [1, 6]])
    tok_slot = torch.tensor([0, 0, 1, 1, 2, 3, 4, 4, 5, 5])
    with torch.no_grad():
        _, tx = tt(pixels, ids, pos, blocks, cands)
        tl = th(tx, q_pos, slot_pos, slot_q, tok_pos, tok_slot)
    _, mxt = mt(_mx(pixels), _mx(ids), _mx(pos), _mx(blocks), _mx(cands))
    ml = mh(mxt, _mx(q_pos), _mx(slot_pos), _mx(slot_q), _mx(tok_pos), _mx(tok_slot))
    np.testing.assert_allclose(np.array(ml), tl.numpy(), atol=1e-4, rtol=1e-4)


def _hf_cfg(repo):
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(repo, local_files_only=True)
    except Exception:
        pytest.skip(f"{repo} config not cached")


def test_mlx_modernbert_matches_hf_with_packed_masks():
    from transformers import AutoModel

    from glance.mlx.modernbert import ModernBertMLX, modernbert_masks
    from glance.mlx.tower import text_block_mask
    from glance.model.masks import modernbert_masks as torch_masks
    from glance.model.masks import text_block_mask as torch_block_mask

    cfg = _hf_cfg("jhu-clsp/ettin-encoder-32m")
    torch.manual_seed(0)
    hf = AutoModel.from_config(cfg).eval()
    ids = torch.randint(5, 5000, (1, 150))
    blocks = torch.cat([torch.zeros(90), torch.ones(60)]).long()[None]
    pos = torch.cat([torch.arange(90), torch.arange(60)])[None]
    cands = torch.full((1, 150), -1)
    cands[0, 20:30], cands[0, 30:40] = 0, 1
    with torch.no_grad():
        m = torch_block_mask(blocks, cands)[:, None]
        want = hf(input_ids=ids, position_ids=pos, attention_mask=torch_masks(m, cfg.local_attention, pos)).last_hidden_state
    mb = ModernBertMLX(cfg)
    mb.load_weights([(k, mx.array(v.float().numpy())) for k, v in hf.state_dict().items()])
    bm = text_block_mask(_mx(blocks), _mx(cands))
    got = mb(modernbert_masks(bm, cfg.local_attention, _mx(pos)), _mx(pos), input_ids=_mx(ids))
    np.testing.assert_allclose(np.array(got), want.numpy(), atol=2e-4, rtol=2e-4)
