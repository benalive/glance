import pytest
import torch

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


def test_onnx_runtime_matches_torch_for_every_request_shape(tmp_path):
    """Export the C5 image and question paths, then compare ONNX Runtime with PyTorch on requests of
    different sizes and kinds (dynamic batch, length, question and slot counts)."""
    from transformers import AutoTokenizer

    from glance.data.packing import collate, group_by_image
    from glance.data.schema import Decision
    from glance.model.candidates import build_candidate
    from glance.onnx_export import OnnxRuntime, export

    torch.manual_seed(0)  # random inputs and untrained fusion layers: fix them so the tolerance check is stable
    cand = build_candidate("C5").eval()
    tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m", local_files_only=True)
    export(cand, tok, tmp_path)
    rt = OnnxRuntime(tmp_path, 2)
    px = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        feats = cand.image_features(px)  # rounded to fp16, as training and the server read features
        assert torch.allclose(rt.image_features(px), cand.frozen_image(px), atol=1e-3)  # ONNX gives the unrounded fp32
        kinds = [("noul", ["yes", "no"]), ("choice", ["a red car", "a blue bus", "nothing"]),
                 ("score", ["bad", "poor", "fair", "good", "excellent"])]
        for n in (1, 3, 8):
            ds = [Decision(f"q{i}", "t", kinds[i % 3][0], f"question number {i} about the picture?" + " very" * i,
                           kinds[i % 3][1], [1 / len(kinds[i % 3][1])] * len(kinds[i % 3][1]), b"", "img") for i in range(n)]
            seqs = group_by_image(ds, tok, 320, 4)
            b = collate([es for _, es in seqs], tok.pad_token_id)
            want = cand(feats.expand(len(seqs), *feats.shape[1:]), b)
            assert torch.allclose(rt.logits(feats, b), want, atol=1e-3), n
