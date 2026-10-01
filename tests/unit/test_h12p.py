"""H12''s held-out split (cycle5.yaml v6) on a synthetic frame: no corpus text needed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vifeedback.training import h12p as H


def _frame(n: int = 9000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "topic": rng.choice(["Academic", "Spam", "Service"], n, p=[0.6, 0.1, 0.3]),
            "sentiment": rng.choice(["neutral", "negative", "positive"], n, p=[0.68, 0.19, 0.13]),
        }
    )


def test_holdout_is_stratified_in_scope_and_disjoint():
    frame = _frame()
    held, train = H.holdout_split(frame, {"Spam"})
    in_scope = np.flatnonzero(frame.topic.to_numpy() != "Spam")
    assert len(np.intersect1d(held, train)) == 0
    assert np.array_equal(np.sort(np.concatenate([held, train])), in_scope)
    assert abs(len(held) - H.N_HOLDOUT) <= 2  # per-label rounding
    share = frame.sentiment.iloc[in_scope].value_counts(normalize=True)
    held_share = frame.sentiment.iloc[held].value_counts(normalize=True)
    assert (held_share - share).abs().max() < 0.002


def test_holdout_is_deterministic():
    frame = _frame()
    assert np.array_equal(H.holdout_split(frame, {"Spam"})[0], H.holdout_split(frame, {"Spam"})[0])
