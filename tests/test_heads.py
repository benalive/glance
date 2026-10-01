import torch

from glance.model.heads import PointerHead, decision_loss, pad_by_question


def test_padding_and_loss_ignore_missing_candidates():
    # q0 has 3 candidates, q1 has 2.
    logits = torch.tensor([2.0, 0.0, -1.0, 0.5, 0.5])
    slot_q = torch.tensor([0, 0, 0, 1, 1])
    slot_k = torch.tensor([0, 1, 2, 0, 1])
    padded = pad_by_question(logits, slot_q, slot_k, n_q=2, k_max=3)
    assert torch.isinf(padded[1, 2])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    loss, p = decision_loss(padded, targets, torch.tensor([False, False]))
    assert torch.allclose(p.sum(-1), torch.ones(2)) and p[1, 2] == 0
    # q1 predicts exactly its soft target: brier 0, CE = entropy ln 2.
    assert torch.isfinite(loss)
    manual_q1 = 0.3 * torch.log(torch.tensor(2.0))
    q0_p = torch.softmax(torch.tensor([2.0, 0.0, -1.0]), 0)
    manual_q0 = ((q0_p - targets[0]) ** 2).sum() - 0.3 * torch.log(q0_p[0])
    assert torch.allclose(loss, (manual_q0 + manual_q1) / 2, atol=1e-6)


def test_rps_penalises_distance_for_score_items():
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    near = torch.tensor([[-9.0, -9.0, -9.0, 9.0, -9.0]])  # predicts level 4 when truth is 5
    far = torch.tensor([[9.0, -9.0, -9.0, -9.0, -9.0]])  # predicts level 1
    score = torch.tensor([True])
    assert decision_loss(near, target, score, ce_weight=0)[0] < decision_loss(far, target, score, ce_weight=0)[0]
    # Brier would treat both errors the same
    brier_near = decision_loss(near, target, torch.tensor([False]), ce_weight=0)[0]
    brier_far = decision_loss(far, target, torch.tensor([False]), ce_weight=0)[0]
    assert torch.allclose(brier_near, brier_far, atol=1e-6)


def test_pointer_head_shapes():
    head = PointerHead(16, 8)
    states = torch.randn(2, 10, 16)
    q_pos = torch.tensor([[0, 0], [1, 3]])
    slot_pos = torch.tensor([[0, 4], [0, 6], [1, 7], [1, 9]])
    out = head(states, q_pos, slot_pos, torch.tensor([0, 0, 1, 1]))
    assert out.shape == (4,)
