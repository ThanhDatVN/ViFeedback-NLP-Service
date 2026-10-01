"""The H12' held-out split (cycle5.yaml v6) on a synthetic frame: no corpus text needed."""

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


def test_rule_needs_every_condition():
    held = {"observed_diff": 0.02, "ci_low": 0.004, "ci_high": 0.03}
    ok = {
        "2_uit_validation_macro_f1": -0.004,
        "3_uit_validation_neutral_f1": -0.01,
        "4_uit_validation_stripped_macro_f1": 0.0,
    }
    assert H.apply_rule(held, ok, -0.01, 0.05)["passed"]
    assert not H.apply_rule({**held, "ci_low": -0.001}, ok, -0.01, 0.05)["passed"]
    worse = {**ok, "4_uit_validation_stripped_macro_f1": -0.0107}  # H8's failure
    assert not H.apply_rule(held, worse, -0.01, 0.05)["passed"]
    assert not H.apply_rule(held, ok, -0.03, 0.05)["passed"]  # loses H10b's real-typing gain
    assert not H.apply_rule(held, ok, 0.0, 0.11)["passed"]


def test_holdout_is_deterministic():
    frame = _frame()
    assert np.array_equal(H.holdout_split(frame, {"Spam"})[0], H.holdout_split(frame, {"Spam"})[0])
