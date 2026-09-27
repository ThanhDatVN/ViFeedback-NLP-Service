"""Unit tests for the Cycle 1 interventions: augmentation, cRT, and their effect on run identity.

Pinned: augmentation never changes labels or dataset size and is reproducible per seed; cRT really
re-initializes and retrains only the head, with balanced sampling that lifts a suppressed minority
class; and adding the new config fields did not change the identity of any existing configuration.
"""

from __future__ import annotations

import numpy as np
import pytest

from vifeedback.training.augment import RECIPES, augment

TEXTS = [
    "giảng viên không nhiệt tình",
    "thầy dạy rất hay",
    "phòng học được trang bị tốt",
    "sinh viên mong được học thêm",
] * 50


class TestAugment:
    def test_size_is_preserved_and_exposure_tracks_p(self) -> None:
        out, changed = augment(TEXTS, "diac-teen", 0.3, seed=42)
        assert len(out) == len(TEXTS)
        assert 0.15 < changed.mean() <= 0.35  # some selected sentences have nothing to change

    def test_p_zero_changes_nothing(self) -> None:
        out, changed = augment(TEXTS, "diac-teen", 0.0, seed=42)
        assert out == TEXTS and not changed.any()

    def test_reproducible_per_seed_and_different_across_seeds(self) -> None:
        a = augment(TEXTS, "diac-teen", 0.3, seed=1)[0]
        b = augment(TEXTS, "diac-teen", 0.3, seed=1)[0]
        c = augment(TEXTS, "diac-teen", 0.3, seed=2)[0]
        assert a == b and a != c

    def test_charnoise_is_never_used_for_training(self) -> None:
        """charnoise stays an out-of-family robustness check."""
        for suites in RECIPES.values():
            assert not any(s.startswith("charnoise") for s in suites)

    @pytest.mark.parametrize(("recipe", "p"), [("nope", 0.3), ("diac-teen", 1.5)])
    def test_bad_arguments_raise(self, recipe: str, p: float) -> None:
        with pytest.raises(ValueError):
            augment(TEXTS, recipe, p, seed=0)


class TestCRT:
    def _model(self):
        transformers = pytest.importorskip("transformers")
        cfg = transformers.RobertaConfig(
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=40,
            num_labels=3,
        )
        return transformers.RobertaForSequenceClassification(cfg)

    def test_balanced_head_retraining_recovers_a_minority_class(self) -> None:
        torch = pytest.importorskip("torch")
        from vifeedback.training.crt import retrain_head

        rng = np.random.default_rng(0)
        y = rng.choice(3, size=3000, p=[0.48, 0.04, 0.48])
        centers = rng.normal(scale=3.0, size=(3, 16))
        feats = torch.tensor(centers[y] + rng.normal(size=(3000, 16)), dtype=torch.float32)
        yd = rng.choice(3, size=600, p=[0.48, 0.04, 0.48])
        fd = torch.tensor(centers[yd] + rng.normal(size=(600, 16)), dtype=torch.float32)

        model = self._model()
        old_head = model.classifier
        r = retrain_head(model, feats, y, fd, yd, task="sentiment", epochs=5, seed=0)

        assert r["model"].classifier is not old_head, "the head must be re-initialized"
        with torch.no_grad():
            pred = r["model"].classifier(fd[:, None, :]).argmax(1).numpy()
        assert (pred[yd == 1] == 1).mean() > 0.8
        assert r["best_dev_macro_f1"] > 0.9

    def test_encoder_is_untouched(self) -> None:
        torch = pytest.importorskip("torch")
        from vifeedback.training.crt import retrain_head

        model = self._model()
        before = {k: v.clone() for k, v in model.roberta.state_dict().items()}
        f = torch.randn(90, 16)
        y = np.array([0, 1, 2] * 30)
        retrain_head(model, f, y, f, y, task="sentiment", epochs=1, seed=0)
        for k, v in model.roberta.state_dict().items():
            assert torch.equal(v, before[k]), k


class TestIdentity:
    def test_new_fields_do_not_change_existing_config_hashes(self) -> None:
        """Pinned value: the hash of this config before the Cycle 1 fields existed."""
        pytest.importorskip("torch")
        from vifeedback.training.trainer import TrainConfig

        assert TrainConfig(task="sentiment", seed=42, lr=2e-5).config_hash() == "0f7bd4c4"

    def test_setting_a_new_field_changes_the_hash(self) -> None:
        pytest.importorskip("torch")
        from vifeedback.training.trainer import TrainConfig

        base = TrainConfig(task="sentiment", seed=42)
        assert (
            base.config_hash()
            != TrainConfig(task="sentiment", seed=42, crt_epochs=10).config_hash()
        )
        assert (
            base.config_hash()
            != TrainConfig(
                task="sentiment", seed=42, augment="diac-teen", augment_p=0.3
            ).config_hash()
        )

    def test_report_defaults_mirror_the_trainer(self) -> None:
        pytest.importorskip("torch")
        from vifeedback.evaluation.report import _OPTIONAL_DEFAULTS
        from vifeedback.training.trainer import TrainConfig

        assert dict(TrainConfig._OPTIONAL_IDENTITY_FIELDS) == _OPTIONAL_DEFAULTS
