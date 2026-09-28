"""Cycle 4 H8 logic (training/domain.py) that needs no model: configs, the rule's statistics, selection."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vifeedback.training import domain as D


def test_recipes_vary_only_what_cycle4_declares():
    mixed, seq = D.config("mixed", 42), D.config("sequential", 7)
    assert (mixed.epochs, mixed.lr, mixed.augment, mixed.augment_p) == (4, 2e-5, "diac-teen", 0.3)
    assert (seq.epochs, seq.lr, seq.seed) == (2, 1e-5, 7)
    assert mixed.run_id().startswith("p12-sent-phobert-base-seg_pyvi-h8-mixed-s42-")
    assert D.config("two-heads", 42).config_hash() != mixed.config_hash()


def test_seed_paired_bootstrap_sign_and_null():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    noisy = [np.where(rng.random(600) < 0.4, rng.integers(0, 3, 600), y) for _ in range(3)]
    same = D.seed_paired_bootstrap(y, noisy, noisy, n_resamples=200)
    assert same["observed_diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = D.seed_paired_bootstrap(y, [y, y, y], noisy, n_resamples=200)
    assert better["observed_diff"] > 0 and better["ci_low"] > 0 and better["seeds"] == 3


def _summary(tmp, name, uit, neu):
    d = tmp / name
    d.mkdir(parents=True)
    sets = {"uit_validation": {"macro_f1": uit}, "neu_validation": {"macro_f1": neu}}
    (d / "summary.json").write_text(json.dumps({"sets": sets}), encoding="utf-8")


def test_selection_takes_the_best_eligible_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "OUT", tmp_path)
    _summary(tmp_path, "control-s42", 0.86, 0.43)
    _summary(tmp_path, "mixed-s42", 0.845, 0.70)  # best on NEU-ESC but 0.015 below on UIT-VSFC
    _summary(tmp_path, "two-heads-s42", 0.858, 0.52)
    _summary(tmp_path, "sequential-s42", 0.851, 0.60)
    out = D.select()
    assert out["recipes"]["mixed"]["eligible"] is False
    assert out["chosen"] == "sequential" and out["outcome"] == "confirm"
    assert (tmp_path / "selection.json").exists()


def test_selection_stops_when_nothing_is_eligible(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "OUT", tmp_path)
    _summary(tmp_path, "control-s42", 0.86, 0.43)
    for r in D.RECIPES:
        _summary(tmp_path, f"{r}-s42", 0.80, 0.70)
    out = D.select()
    assert out["chosen"] is None and out["outcome"].startswith("not supported")


def test_selection_needs_every_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "OUT", tmp_path)
    _summary(tmp_path, "control-s42", 0.86, 0.43)
    with pytest.raises(FileNotFoundError):
        D.select()
