"""Cycle 5 H10 logic (training/consistency.py) that needs no model: the split, the loss, the rule."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vifeedback.training import consistency as C


def test_dev_split_is_the_declared_permutation():
    dev, train = C.split_indices()
    assert (len(dev), len(train)) == (837, 7535)
    assert not set(dev) & set(train) and len(set(dev) | set(train)) == 8372
    perm = np.random.default_rng(42).permutation(8372)
    assert np.array_equal(dev, perm[:837]) and np.array_equal(train, perm[837:])
    assert np.array_equal(C.split_indices()[0], dev)  # the same every time
    with pytest.raises(ValueError, match="declared 8372"):
        C.split_indices(8000)


def test_recipes_change_only_the_run_identity():
    one, sym = C.config("onesided", 42), C.config("symmetric", 7)
    assert (one.epochs, one.lr, one.augment, one.augment_p) == (4, 2e-5, "diac-teen", 0.3)
    assert one.run_id().startswith("p13-sent-phobert-base-seg_pyvi-h10-onesided-s42-")
    assert sym.seed == 7 and C.config("symmetric", 42).config_hash() != one.config_hash()
    with pytest.raises(ValueError):
        C.config("both", 42)


def test_onesided_loss_is_kl_to_a_fixed_target():
    torch = pytest.importorskip("torch")
    o = torch.tensor([[2.0, 0.0, -1.0], [0.1, 0.2, 0.3]], requires_grad=True)
    n = torch.tensor([[0.0, 1.0, 0.0], [0.1, 0.2, 0.3]], requires_grad=True)
    loss = C.consistency_loss(o, n, "onesided")
    p_n = torch.softmax(n.detach(), -1)
    expected = (p_n * (p_n.log() - torch.log_softmax(o.detach(), -1))).sum(-1).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert o.grad is not None and n.grad is None  # no gradient through the normalized form
    same = C.consistency_loss(n.detach(), n.detach(), "onesided")
    assert abs(float(same)) < 1e-7


def test_symmetric_loss_is_r_drop_and_moves_both_forms():
    torch = pytest.importorskip("torch")
    from vifeedback.training.losses import rdrop_kl

    o = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    n = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)
    loss = C.consistency_loss(o, n, "symmetric")
    assert torch.allclose(loss, rdrop_kl(o, n)) and float(loss) > 0
    loss.backward()
    assert o.grad is not None and n.grad is not None
    with pytest.raises(ValueError):
        C.consistency_loss(o, n, "other")


def test_flip_rate_and_its_paired_bootstrap():
    assert C.flip_rate(np.array([0, 1, 2, 2]), np.array([0, 2, 2, 1])) == 0.5
    rng = np.random.default_rng(0)
    ctrl = [rng.random(1000) < 0.2 for _ in range(5)]
    same = C.flip_bootstrap(ctrl, ctrl, n_resamples=500)
    assert same["observed_diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    never = [np.zeros(1000, bool)] * 5
    down = C.flip_bootstrap(never, ctrl, n_resamples=500)
    assert down["observed_diff"] < -0.15 and down["ci_high"] < 0 and down["seeds"] == 5


def _flip(ci_high: float) -> dict:
    return {"observed_diff": -0.02, "ci_low": -0.03, "ci_high": ci_high}


GOOD = {
    "2_uit_validation_macro_f1": -0.005,
    "3_uit_validation_neutral_f1": -0.02,
    "4_uit_validation_stripped_macro_f1": -0.01,
    "5_neu_validation_macro_f1": -0.01,
}


def test_rule_passes_at_its_limits():
    out = C.apply_rule(_flip(-0.001), GOOD)
    assert out["passed"] and all(r["passed"] for r in out["rules"].values())


@pytest.mark.parametrize("name", list(GOOD))
def test_rule_fails_just_past_each_guard(name):
    diffs = {**GOOD, name: GOOD[name] - 0.0001}
    out = C.apply_rule(_flip(-0.001), diffs)
    assert not out["passed"] and not out["rules"][name]["passed"]


def test_rule_needs_the_whole_interval_below_zero():
    assert not C.apply_rule(_flip(0.0), GOOD)["passed"]


def _summary(tmp, name, uit, flip):
    d = tmp / name
    d.mkdir(parents=True)
    body = {"sets": {"uit_validation": {"macro_f1": uit}}, "vilexnorm_dev": {"flip_rate": flip}}
    (d / "summary.json").write_text(json.dumps(body), encoding="utf-8")


def test_selection_takes_the_eligible_recipe_with_fewest_flips(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT", tmp_path)
    _summary(tmp_path, "control-s42", 0.864, 0.17)
    _summary(tmp_path, "onesided-s42", 0.860, 0.12)
    _summary(tmp_path, "symmetric-s42", 0.850, 0.08)  # fewest flips, but 0.014 below on UIT-VSFC
    out = C.select()
    assert out["recipes"]["symmetric"]["eligible"] is False
    assert out["chosen"] == "onesided" and out["outcome"] == "confirm"
    assert (tmp_path / "selection.json").exists()


def test_selection_stops_when_no_recipe_lowers_the_flip_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT", tmp_path)
    _summary(tmp_path, "control-s42", 0.864, 0.17)
    _summary(tmp_path, "onesided-s42", 0.862, 0.17)
    _summary(tmp_path, "symmetric-s42", 0.861, 0.18)
    out = C.select()
    assert out["chosen"] is None and "lowers the dev flip rate" in out["outcome"]


def test_selection_needs_every_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT", tmp_path)
    _summary(tmp_path, "control-s42", 0.864, 0.17)
    _summary(tmp_path, "onesided-s42", 0.862, 0.12)
    with pytest.raises(FileNotFoundError, match="symmetric-s42"):
        C.select()
