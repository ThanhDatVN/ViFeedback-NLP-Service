"""End-to-end checks of the training loop on CPU with a tiny slice of the real corpus.

These exist because `--grad-accum` and `--freeze-embeddings` were added to support the Kaggle
models and shipped without ever being executed. `--grad-accum` in particular is load-bearing for a
*correctness* reason rather than a convenience one: `phobert-large` needs batch 16 to fit, and
comparing it against `phobert-base` at batch 32 without accumulation would not be a controlled
comparison.

Marked `slow` and `needs_data`; excluded from the fast PR loop.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from vifeedback.data.loader import MANIFEST, load  # noqa: E402
from vifeedback.training.trainer import TrainConfig, build_optimizer, train  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.needs_data]

TINY = 48


@pytest.fixture(scope="module")
def tiny():
    if not MANIFEST.exists():
        pytest.skip("data/raw not populated — run `vifeedback data fetch`")
    df = load("train")
    # Stratify by hand so every class appears; otherwise a 48-row head is all majority classes
    # and a 3-way classifier cannot even be constructed meaningfully.
    parts = [df[df.sentiment == c].head(TINY // 3) for c in (0, 1, 2)]
    import pandas as pd

    small = pd.concat(parts, ignore_index=True)
    return small["sentence"].tolist(), small["sentiment"].to_numpy()


def _cfg(**kw) -> TrainConfig:
    base = dict(
        task="sentiment",
        model_key="phobert-base",
        epochs=1,
        batch_size=8,
        eval_batch_size=8,
        max_length=32,
        fp16=False,
        device="cpu",
        num_workers=0,
        seed=42,
    )
    base.update(kw)
    return TrainConfig(**base)


class TestGradientAccumulation:
    def test_runs_and_produces_a_finite_loss(self, tiny):
        x, y = tiny
        out = train(_cfg(grad_accum=2), x, y, x, y, verbose=False)
        assert len(out["history"]) == 1
        assert np.isfinite(out["history"][0]["train_loss"])
        assert 0.0 <= out["history"][0]["dev_macro_f1"] <= 1.0

    def test_accumulation_changes_the_weights_the_same_way_batching_would(self, tiny):
        """batch 4 x accum 2 must land closer to batch 8 x accum 1 than batch 4 x accum 1 does.

        This is the property the Kaggle comparison depends on. If accumulation were a no-op, or
        double-counted the loss, this ordering would break.
        """
        x, y = tiny

        def head(cfg):
            out = train(cfg, x, y, x, y, verbose=False)
            return out["model"].classifier.out_proj.weight.detach().flatten().numpy().copy()

        big = head(_cfg(batch_size=8, grad_accum=1))
        accum = head(_cfg(batch_size=4, grad_accum=2))
        small = head(_cfg(batch_size=4, grad_accum=1))

        assert np.linalg.norm(accum - big) < np.linalg.norm(small - big)


class TestFreezeEmbeddings:
    def test_embeddings_stop_requiring_grad(self, tiny):
        x, y = tiny
        out = train(_cfg(freeze_embeddings=True, epochs=1), x, y, x, y, verbose=False)
        model = out["model"]
        emb = [p for n, p in model.named_parameters() if "embeddings" in n]
        assert emb, "no embedding parameters found — the name filter is wrong"
        assert all(not p.requires_grad for p in emb)
        assert any(p.requires_grad for n, p in model.named_parameters() if "embeddings" not in n)

    def test_frozen_embeddings_are_unchanged_after_training(self, tiny):
        """The point of freezing, stated as a test: the 192M-row table must not move."""
        from transformers import AutoModelForSequenceClassification

        from vifeedback.constants import MODEL_IDS

        x, y = tiny
        before = (
            AutoModelForSequenceClassification.from_pretrained(
                MODEL_IDS["phobert-base"], num_labels=3
            )
            .roberta.embeddings.word_embeddings.weight.detach()
            .clone()
        )
        out = train(_cfg(freeze_embeddings=True), x, y, x, y, verbose=False)
        after = out["model"].roberta.embeddings.word_embeddings.weight.detach()
        assert torch.allclose(before, after)

    def test_optimizer_excludes_frozen_parameters(self):
        """A frozen parameter left in the optimizer still costs its Adam state - which is the
        entire 3 GB that freezing is supposed to save on xlm-roberta-base."""
        from transformers import AutoModelForSequenceClassification

        from vifeedback.constants import MODEL_IDS

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_IDS["phobert-base"], num_labels=3
        )
        for n, p in model.named_parameters():
            if "embeddings" in n:
                p.requires_grad = False

        opt = build_optimizer(model, _cfg())
        in_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert in_opt == trainable
        assert in_opt < sum(p.numel() for p in model.parameters())


class TestCheckpointing:
    def test_save_checkpoint_writes_a_loadable_model(self, tiny, tmp_path):
        from transformers import AutoModelForSequenceClassification

        x, y = tiny
        out = train(_cfg(), x, y, x, y, verbose=False)
        out["model"].save_pretrained(tmp_path)
        out["tokenizer"].save_pretrained(tmp_path)

        reloaded = AutoModelForSequenceClassification.from_pretrained(tmp_path)
        assert reloaded.config.num_labels == 3
        assert reloaded.config.id2label[0] == "negative"
        assert reloaded.config.id2label[1] == "neutral"
