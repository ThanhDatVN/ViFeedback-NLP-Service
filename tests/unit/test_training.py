"""Unit tests for the training components.

Training *convergence* is deliberately not tested — it is slow and non-deterministic, and the
registry plus the gate reviews cover it (docs/EVALUATION_PROTOCOL.md § Software testing). What is
tested here is everything that can silently corrupt a run without failing loudly: loss definitions,
the layer-wise learning-rate assignment, the LR schedule, and seeding.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vifeedback.training import losses as L  # noqa: E402
from vifeedback.training.seeding import seed_everything  # noqa: E402
from vifeedback.training.trainer import (  # noqa: E402
    TrainConfig,
    build_optimizer,
    linear_warmup_schedule,
    softmax,
)

# UIT-VSFC train sentiment counts, measured at Gate G0.
VSFC_COUNTS = np.array([5325, 458, 5643])


class TestClassWeights:
    @pytest.mark.parametrize("scheme", ["balanced", "sqrt", "effective"])
    def test_minority_class_gets_the_largest_weight(self, scheme: str) -> None:
        w = L.class_weights_from_counts(VSFC_COUNTS, scheme).numpy()
        assert w.argmax() == 1, f"{scheme}: neutral must be up-weighted"
        assert np.all(w > 0)

    def test_balanced_matches_sklearn(self) -> None:
        # Guarded rather than module-level: sklearn is needed by this one test, and a host policy
        # blocking its native extension (ADR-017) should not skip the other twenty-six.
        compute_class_weight = pytest.importorskip(
            "sklearn.utils.class_weight",
            reason="scikit-learn unavailable (may be blocked by host policy)",
        ).compute_class_weight

        y = np.concatenate([np.full(c, i) for i, c in enumerate(VSFC_COUNTS)])
        sk = compute_class_weight("balanced", classes=np.arange(3), y=y)
        ours = L.class_weights_from_counts(VSFC_COUNTS, "balanced").numpy()
        # We normalize to mean 1, so compare up to a positive scale factor.
        assert np.allclose(ours / ours.sum(), sk / sk.sum())

    def test_schemes_are_ordered_by_aggressiveness(self) -> None:
        """balanced > sqrt on a 4% class; `effective` saturates and must not be the most extreme."""
        b = L.class_weights_from_counts(VSFC_COUNTS, "balanced").numpy()
        s = L.class_weights_from_counts(VSFC_COUNTS, "sqrt").numpy()
        assert b[1] > s[1] > 1.0

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown scheme"):
            L.class_weights_from_counts(VSFC_COUNTS, "nonsense")


class TestLosses:
    def test_focal_with_gamma_zero_equals_cross_entropy(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(64, 3)
        target = torch.randint(0, 3, (64,))
        focal = L.FocalLoss(gamma=0.0)(logits, target)
        ce = torch.nn.CrossEntropyLoss()(logits, target)
        assert torch.allclose(focal, ce, atol=1e-6)

    def test_focal_down_weights_easy_examples(self) -> None:
        """The defining property: confident-correct examples contribute less than under CE."""
        easy = torch.tensor([[10.0, 0.0, 0.0]])
        target = torch.tensor([0])
        ce = torch.nn.CrossEntropyLoss()(easy, target)
        focal = L.FocalLoss(gamma=2.0)(easy, target)
        assert focal < ce

    def test_logit_adjustment_shifts_toward_the_rare_class(self) -> None:
        """Adding -tau*log(prior) makes the rare class cheaper to predict at training time."""
        loss = L.LogitAdjustedLoss(VSFC_COUNTS, tau=1.0)
        adj = loss.adjustment.numpy()
        assert adj[1] < adj[0] and adj[1] < adj[2], (
            "neutral must get the largest (most negative) shift"
        )

    def test_rdrop_kl_is_zero_for_identical_logits_and_positive_otherwise(self) -> None:
        a = torch.randn(16, 3)
        assert L.rdrop_kl(a, a).item() == pytest.approx(0.0, abs=1e-6)
        assert L.rdrop_kl(a, torch.randn(16, 3)).item() > 0

    def test_rdrop_kl_is_symmetric(self) -> None:
        a, b = torch.randn(16, 3), torch.randn(16, 3)
        assert L.rdrop_kl(a, b).item() == pytest.approx(L.rdrop_kl(b, a).item(), abs=1e-6)

    @pytest.mark.parametrize(
        "name", ["ce", "classweight", "focal", "focal-weighted", "logit-adjust"]
    )
    def test_factory_builds_a_usable_loss(self, name: str) -> None:
        loss = L.build_loss(name, VSFC_COUNTS)
        value = loss(torch.randn(8, 3), torch.randint(0, 3, (8,)))
        assert torch.isfinite(value) and value.item() >= 0

    def test_factory_rejects_unknown_names(self) -> None:
        with pytest.raises(ValueError, match="unknown loss"):
            L.build_loss("nope", VSFC_COUNTS)


class _TinyEncoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(4, 4)


class _TinyModel(torch.nn.Module):
    """Minimal stand-in with the RoBERTa parameter-name layout, so LLRD can be tested without
    downloading a 135M-parameter checkpoint."""

    base_model_prefix = "roberta"

    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        self.roberta = torch.nn.Module()
        self.roberta.embeddings = torch.nn.Embedding(10, 4)
        self.roberta.encoder = torch.nn.Module()
        self.roberta.encoder.layer = torch.nn.ModuleList(
            [_TinyEncoderLayer() for _ in range(n_layers)]
        )
        self.classifier = torch.nn.Linear(4, 3)


class TestOptimizer:
    def test_head_gets_a_higher_lr_than_the_encoder(self) -> None:
        cfg = TrainConfig(lr=2e-5)
        opt = build_optimizer(_TinyModel(), cfg)
        lrs = [g["lr"] for g in opt.param_groups]
        assert max(lrs) > cfg.lr

    def test_llrd_decays_monotonically_from_top_to_bottom(self) -> None:
        cfg = TrainConfig(lr=2e-5, llrd=0.9)
        model = _TinyModel(n_layers=4)
        opt = build_optimizer(model, cfg)
        by_name = dict(zip([n for n, p in model.named_parameters()], opt.param_groups, strict=True))

        layer_lrs = [by_name[f"roberta.encoder.layer.{i}.dense.weight"]["lr"] for i in range(4)]
        assert layer_lrs == sorted(layer_lrs), "lower layers must get smaller lrs"
        assert by_name["roberta.embeddings.weight"]["lr"] < layer_lrs[0]
        assert by_name["classifier.weight"]["lr"] > layer_lrs[-1]

    def test_no_weight_decay_on_bias_and_layernorm(self) -> None:
        cfg = TrainConfig(weight_decay=0.01)
        model = _TinyModel()
        opt = build_optimizer(model, cfg)
        for (name, _), group in zip(model.named_parameters(), opt.param_groups, strict=True):
            expected = 0.0 if "bias" in name else 0.01
            assert group["weight_decay"] == expected, name


class TestSchedule:
    def test_warms_up_then_decays_to_zero(self) -> None:
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)
        sched = linear_warmup_schedule(opt, num_training_steps=100, warmup_ratio=0.1)

        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()

        assert lrs[0] == pytest.approx(0.0, abs=1e-9)
        assert lrs[10] == pytest.approx(1.0, abs=1e-6), "peak at the end of warmup"
        assert lrs[10] > lrs[50] > lrs[99]
        assert lrs[99] < 0.05


class TestSeeding:
    def test_same_seed_gives_identical_draws(self) -> None:
        seed_everything(42)
        a = (np.random.rand(5), torch.randn(5))
        seed_everything(42)
        b = (np.random.rand(5), torch.randn(5))
        assert np.allclose(a[0], b[0])
        assert torch.allclose(a[1], b[1])

    def test_different_seeds_diverge(self) -> None:
        seed_everything(42)
        a = np.random.rand(5)
        seed_everything(1337)
        assert not np.allclose(a, np.random.rand(5))


class TestSoftmax:
    def test_rows_sum_to_one_and_match_scipy(self) -> None:
        from scipy.special import softmax as sp_softmax

        x = np.random.default_rng(0).normal(size=(20, 3)) * 10
        out = softmax(x)
        assert np.allclose(out.sum(axis=1), 1.0)
        assert np.allclose(out, sp_softmax(x, axis=1))

    def test_is_numerically_stable_on_large_logits(self) -> None:
        out = softmax(np.array([[1000.0, 999.0, 998.0]]))
        assert np.all(np.isfinite(out)) and out.sum() == pytest.approx(1.0)


class TestTrainConfig:
    def test_run_id_is_wellformed(self) -> None:
        """Format: p<phase>-<task>-<model>-<preproc>-<recipe>-s<seed>-<confighash>-<split>.

        The config hash was added after review (R11): without it, changing the learning rate,
        epoch count or max length left the id unchanged, and `save_run()` writes to
        `results/runs/<run_id>/` — so a different configuration silently overwrote the artifact a
        published number pointed to.
        """
        cfg = TrainConfig(task="sentiment", model_key="phobert-base", recipe="focal", seed=1337)
        rid = cfg.run_id("validation")

        assert rid.startswith("p2-sent-phobert-base-raw-focal-s1337-")
        assert rid.endswith("-val")
        assert rid == rid.lower() and " " not in rid

        parts = rid.split("-")
        assert parts[-2] == cfg.config_hash()
        assert len(cfg.config_hash()) == 8

    def test_max_length_default_matches_the_gate_g0_decision(self) -> None:
        """Gate G0 measured PhoBERT subword p99.9 = 87 and chose 96. If the default drifts, the
        latency numbers in Phase 6 stop corresponding to the documented decision."""
        assert TrainConfig().max_length == 96
