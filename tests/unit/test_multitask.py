"""Unit tests for the shared-encoder model (Cycle 1 H4).

A tiny randomly initialized encoder is injected, so nothing is downloaded. Pinned: each task gets its
own correctly sized head, the heads train at the single-task runs' 5x learning rate (otherwise the
comparison with them is not controlled), and joint exact match needs both predictions right.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vifeedback.training import multitask as MT  # noqa: E402
from vifeedback.training.trainer import TrainConfig  # noqa: E402


def _tiny() -> MT.MultiTaskModel:
    cfg = transformers.RobertaConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=40,
    )
    m = MT.MultiTaskModel.__new__(MT.MultiTaskModel)
    torch.nn.Module.__init__(m)
    m.encoder = transformers.RobertaModel(cfg, add_pooling_layer=False)
    m.heads = torch.nn.ModuleDict(
        {"sentiment": MT._Head(16, 3, 0.1), "topic": MT._Head(16, 4, 0.1)}
    )
    return m


def test_each_task_gets_its_own_correctly_sized_head() -> None:
    m = _tiny().eval()
    ids = torch.randint(5, 60, (4, 7))
    out = m(ids, torch.ones_like(ids))
    assert out["sentiment"].shape == (4, 3)
    assert out["topic"].shape == (4, 4)


def test_heads_train_at_the_single_task_head_learning_rate() -> None:
    """trainer.build_optimizer gives the classifier 5x the encoder lr; so must this."""
    cfg = TrainConfig(lr=2e-5)
    m = _tiny()
    opt = MT._optimizer(m, cfg)
    by_name = dict(zip([n for n, p in m.named_parameters()], opt.param_groups, strict=True))
    assert by_name["heads.topic.out_proj.weight"]["lr"] == pytest.approx(1e-4)
    assert by_name["encoder.embeddings.word_embeddings.weight"]["lr"] == pytest.approx(2e-5)
    assert by_name["heads.sentiment.dense.bias"]["weight_decay"] == 0.0


def test_joint_exact_match_needs_both_tasks_right() -> None:
    y = {"sentiment": np.array([0, 1, 2, 0]), "topic": np.array([0, 1, 2, 3])}
    logits = {
        "sentiment": np.eye(3)[[0, 1, 2, 1]] * 5,  # last one wrong
        "topic": np.eye(4)[[0, 1, 0, 3]] * 5,  # third one wrong
    }
    r = MT.evaluate_joint(y, logits)
    assert r["joint_exact_match"] == pytest.approx(0.5)
    assert set(r["per_task"]) == {"sentiment", "topic"}
