import pytest
import torch

from glance.data.packing import collate, encode_question, group_by_image
from glance.data.schema import Decision
from glance.model.heads import PointerHead, decision_loss, pad_by_question
from glance.model.towers import GlanceTower, TowerConfig


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m", local_files_only=True)
    except Exception:
        pytest.skip("Ettin tokenizer not cached")


def _d(uid, img, cands, label, kind="choice"):
    t = [0.0] * len(cands)
    t[label] = 1.0
    return Decision(uid, "t", kind, "what is on the table?", cands, t, b"", img)


def test_candidates_share_start_and_end_in_sep(tok):
    cands = ["a red apple", "cat", "two blue cups"]
    e = encode_question(tok, _d("a", "i0", cands, 1))
    assert e.ids[0] == tok.cls_token_id and e.pos[0] == 0
    assert all(e.ids[s] == tok.sep_token_id for s in e.slot_offsets)
    starts = [e.slot_offsets[0] - len(tok(cands[0], add_special_tokens=False)["input_ids"])]
    starts += [s + 1 for s in e.slot_offsets[:-1]]
    assert len({e.pos[s] for s in starts}) == 1, "every candidate starts at the same position"
    assert e.pos[starts[0]] == starts[0]  # that position is right after the question


def test_group_by_image_packs_same_image_only(tok):
    ds = [_d(f"q{i}", f"img{i % 2}", ["yes", "no"], 0, "noul") for i in range(6)]
    seqs = group_by_image(ds, tok, max_len=320, max_q_per_seq=2)
    assert sum(len(qs) for _, qs in seqs) == 6 and all(len(qs) <= 2 for _, qs in seqs)


def test_end_to_end_candidate_permutation_permutes_probabilities(tok):
    torch.manual_seed(0)
    cfg = TowerConfig(width=64, depth=2, heads=4, mlp_dim=128, patch=32, image_size=64, vocab_size=len(tok))
    tower, head = GlanceTower(cfg).eval(), PointerHead(64, 16).eval()
    cands = ["a red apple", "cat", "two blue cups", "nothing"]
    perm = [2, 0, 3, 1]
    a = encode_question(tok, _d("a", "i", cands, 0))
    b = encode_question(tok, _d("b", "i", [cands[i] for i in perm], perm.index(0)))
    other = encode_question(tok, _d("c", "i", ["yes", "no"], 0, "noul"))
    pixels = torch.randn(1, 3, 64, 64)
    probs = []
    for e in (a, b):
        batch = collate([[e, other]], tok.pad_token_id)
        with torch.no_grad():
            _, txt = tower(pixels, batch["input_ids"], batch["position_ids"], batch["block_ids"], batch["cand_ids"])
            logits = head(txt, batch["q_pos"], batch["slot_pos"], batch["slot_q"], batch["cand_tok_pos"],
                          batch["cand_tok_slot"])
            padded = pad_by_question(logits, batch["slot_q"], batch["slot_k"], 2, batch["k_max"])
            _, p = decision_loss(padded, batch["targets"], batch["is_score"])
        probs.append(p)
    torch.testing.assert_close(probs[1][0, :4], probs[0][0, perm], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(probs[1][1], probs[0][1], atol=1e-5, rtol=1e-5)


def test_same_length_candidates_get_distinct_slots(tok):
    """Regression for the R1 collapse: without candidate isolation, the [SEP] slots of 'yes' and
    'no' saw identical context at identical positions and every yes/no came out exactly 0.5."""
    torch.manual_seed(0)
    cfg = TowerConfig(width=64, depth=2, heads=4, mlp_dim=128, patch=32, image_size=64, vocab_size=len(tok))
    tower, head = GlanceTower(cfg).eval(), PointerHead(64, 16).eval()
    e = encode_question(tok, _d("y", "i", ["yes", "no"], 0, "noul"))
    batch = collate([[e]], tok.pad_token_id)
    with torch.no_grad():
        _, txt = tower(torch.randn(1, 3, 64, 64), batch["input_ids"], batch["position_ids"], batch["block_ids"],
                       batch["cand_ids"])
        logits = head(txt, batch["q_pos"], batch["slot_pos"], batch["slot_q"])
    assert abs(float(logits[0] - logits[1])) > 1e-4


def test_mean_pooled_candidates_start_apart(tok):
    """Regression for the readout saddle: with the slot token alone, 'left'/'right' started at
    cos 0.999996 and gradients cancelled (~1e-6). Mean-pooling each candidate's tokens separates them."""
    torch.manual_seed(0)
    cfg = TowerConfig(width=64, depth=2, heads=4, mlp_dim=128, patch=32, image_size=64, vocab_size=len(tok))
    tower = GlanceTower(cfg).eval()
    e = encode_question(tok, _d("m", "i", ["left", "right"], 0))
    b = collate([[e]], tok.pad_token_id)
    with torch.no_grad():
        _, txt = tower(torch.randn(1, 3, 64, 64), b["input_ids"], b["position_ids"], b["block_ids"], b["cand_ids"])
    slot = [txt[0, s] for s in e.slot_offsets]
    pooled = [txt[0, e.cand_starts[k]:e.slot_offsets[k] + 1].mean(0) for k in range(2)]
    assert torch.cosine_similarity(slot[0], slot[1], dim=0) > 0.99
    assert torch.cosine_similarity(pooled[0], pooled[1], dim=0) < 0.9
