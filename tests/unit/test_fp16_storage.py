"""FP16-storage export (NEXT_PLAN v6 F4, for H11): half the size, FP32 compute, batch-independent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")

from vifeedback.inference import onnx_export as OX  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(0)
    cfg = transformers.RobertaConfig(
        vocab_size=5000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=40,
        num_labels=3,
        pad_token_id=1,
    )
    model = transformers.RobertaForSequenceClassification(cfg).eval()
    ids = torch.randint(3, 5000, (4, 12))
    mask = torch.ones_like(ids)
    mask[1, 8:] = 0  # a padded row
    ids[1, 8:] = 1
    path = OX.export_fp16_storage(model, (ids, mask), tmp_path_factory.mktemp("fp16") / "m.onnx")
    return model, ids, mask, path


def _run(path, ids, mask) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]


def test_the_file_is_about_half_the_fp32_weights(tiny):
    model, _, _, path = tiny
    fp32_bytes = 4 * sum(p.numel() for p in model.parameters())
    assert path.stat().st_size < 0.6 * fp32_bytes


def test_logits_equal_the_fp16_rounded_pytorch_model(tiny):
    model, ids, mask, path = tiny
    with torch.no_grad():
        ref = OX.fp16_rounded(model)(input_ids=ids, attention_mask=mask).logits.numpy()
    got = _run(path, ids, mask)
    assert np.abs(got - ref).max() < 1e-4
    assert (got.argmax(1) == ref.argmax(1)).all()


def test_a_row_scores_the_same_alone_and_in_a_batch(tiny):
    _, ids, mask, path = tiny
    batch = _run(path, ids, mask)
    alone = np.concatenate([_run(path, ids[i : i + 1], mask[i : i + 1]) for i in range(len(ids))])
    assert np.abs(batch - alone).max() < 1e-4


def test_rounding_leaves_the_original_model_untouched(tiny):
    model, _, _, _ = tiny
    before = next(model.parameters()).detach().clone()
    OX.fp16_rounded(model)
    assert torch.equal(next(model.parameters()), before)
