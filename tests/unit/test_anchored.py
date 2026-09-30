"""Cycle 5 H10b (training/anchored.py): the split, the metrics a constant prediction cannot game, the rule."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vifeedback.training import anchored as A
from vifeedback.training import consistency as C


def test_confirmation_split_is_the_declared_permutation():
    confirm, train = A.confirm_split()
    assert (len(confirm), len(train)) == (1500, 6035)
    assert not set(confirm) & set(train)
    assert np.array_equal(confirm, np.random.default_rng(43).permutation(7535)[:1500])
    with pytest.raises(ValueError):
        A.confirm_split(7000)


def test_a_constant_prediction_cannot_game_the_metrics():
    """H10's failure (ADR-038): one label for every informal comment gives zero flips."""
    rng = np.random.default_rng(0)
    teacher = rng.choice(3, size=1000, p=[0.55, 0.35, 0.10])
    constant = np.zeros(1000, dtype=int)
    m = A.pair_metrics(constant, constant, teacher)
    assert m["flip_rate"] == 0.0  # the old metric is fooled
    assert m["agreement"] < 0.6 and m["label_tv"] > 0.4  # the new ones are not
    good = A.pair_metrics(teacher, teacher, teacher)
    assert good["agreement"] == 1.0 and good["label_tv"] == 0.0


def test_anchored_losses_pull_towards_the_frozen_target():
    torch = pytest.importorskip("torch")
    q = torch.tensor([[0.7, 0.2, 0.1]])
    o = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
    n = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    one = C.consistency_loss(o, None, "anchored_orig", q)
    want = (q * (q.log() - torch.log_softmax(o.detach(), -1))).sum()
    assert torch.allclose(one, want, atol=1e-6)
    both = C.consistency_loss(o, n, "anchored_both", q)
    both.backward()
    assert o.grad is not None and n.grad is not None
    assert float(C.consistency_loss(q.log(), None, "anchored_orig", q)) == pytest.approx(
        0, abs=1e-6
    )


def test_anchored_recipes_have_their_own_run_ids():
    a = C.config("anchored_orig", 42)
    assert a.run_id().startswith("p15-sent-phobert-base-seg_pyvi-h10b-anchored_orig-s42-")
    assert C.config("anchored_both", 42).config_hash() != a.config_hash()


GOOD = {
    "4_uit_validation_macro_f1": -0.005,
    "5_uit_validation_neutral_f1": -0.02,
    "6_uit_validation_stripped_macro_f1": -0.01,
    "7_neu_validation_macro_f1": -0.01,
}
UP = {"observed_diff": 0.05, "ci_low": 0.01, "ci_high": 0.08}
DOWN = {"observed_diff": -0.05, "ci_low": -0.08, "ci_high": -0.01}


def test_rule_passes_at_its_limits_and_fails_past_each():
    assert A.apply_rule(UP, DOWN, 0.10, GOOD)["passed"]
    assert not A.apply_rule({**UP, "ci_low": 0.0}, DOWN, 0.10, GOOD)["passed"]
    assert not A.apply_rule(UP, {**DOWN, "ci_high": 0.0}, 0.10, GOOD)["passed"]
    assert not A.apply_rule(UP, DOWN, 0.1001, GOOD)["passed"]
    for k in GOOD:
        assert not A.apply_rule(UP, DOWN, 0.05, {**GOOD, k: GOOD[k] - 1e-4})["passed"]


def _summary(tmp, name, uit, agree, tv):
    d = tmp / name
    d.mkdir(parents=True)
    body = {
        "sets": {"uit_validation": {"macro_f1": uit}},
        "vilexnorm_dev": {"agreement": agree, "flip_rate": 0.1, "label_tv": tv},
    }
    (d / "summary.json").write_text(json.dumps(body), encoding="utf-8")


def test_selection_guards_the_label_distribution(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "OUT", tmp_path)
    _summary(tmp_path, "control-s42", 0.864, 0.70, 0.05)
    _summary(tmp_path, "anchored_orig-s42", 0.862, 0.80, 0.04)
    _summary(tmp_path, "anchored_both-s42", 0.866, 0.90, 0.20)  # best agreement, but collapsing
    out = A.select()
    assert out["recipes"]["anchored_both"]["eligible"] is False
    assert out["chosen"] == "anchored_orig" and out["outcome"] == "confirm"


def test_selection_stops_without_an_agreement_gain(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "OUT", tmp_path)
    _summary(tmp_path, "control-s42", 0.864, 0.80, 0.05)
    _summary(tmp_path, "anchored_orig-s42", 0.862, 0.79, 0.04)
    _summary(tmp_path, "anchored_both-s42", 0.866, 0.78, 0.05)
    assert A.select()["chosen"] is None
