"""The H12' split, rule, inputs and confirmation (cycle5.yaml v6), with no corpus text or model."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from vifeedback import paths
from vifeedback.constants import SEEDS
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


def test_teacher_table_covers_the_declared_pairs_in_order():
    """The committed teacher table (Kaggle trains on it) has a row for every pair H10b used."""
    for split, n in (("train", 6035), ("confirm", 1500), ("dev", 837)):
        q = H.teacher_q_rows(split)
        assert q.shape == (n, 3) and np.allclose(q.sum(1), 1, atol=1e-5)
    committed = pd.read_csv(paths.RESULTS / "studies" / "cycle5" / "h10b_confirm_index.csv").row
    assert set(H.pair_rows("confirm")) == set(committed)
    assert not set(H.pair_rows("train")) & set(H.pair_rows("dev"))


def _fake_study(tmp_path, monkeypatch, stripped_drop: float = 0.0):
    """Ten models' summaries and prediction files for `confirm`, on a synthetic forum split."""
    from vifeedback.evaluation import external as X
    from vifeedback.training.domain import LABELS, OFF_TOPIC

    rng = np.random.default_rng(1)
    n = 400
    frame = pd.DataFrame(
        {
            "text": [f"post {i}" for i in range(n)],
            "topic": rng.choice(["Academic", OFF_TOPIC[0]], n, p=[0.8, 0.2]),
            "sentiment": rng.choice(LABELS, n, p=[0.2, 0.65, 0.15]),
        }
    )
    monkeypatch.setattr(X, "load_neu_esc", lambda split: frame)
    monkeypatch.setattr(H, "N_HOLDOUT", 60)
    out, index = tmp_path / "h12p", tmp_path / "index.csv"
    monkeypatch.setattr(H, "OUT", out)
    monkeypatch.setattr(H, "INDEX", index)
    monkeypatch.setattr(paths, "MODELS", tmp_path / "models")  # no released scope detector
    held, _ = H.holdout_split(frame, OFF_TOPIC)
    pd.DataFrame({"row": held, "sentiment": frame.sentiment.to_numpy()[held]}).to_csv(
        index, index=False
    )
    y = frame.sentiment.to_numpy()[held]
    teacher = [LABELS[i] for i in H.teacher_q_rows("confirm").argmax(1)]

    def sets(drop: float) -> dict:
        def one(f1: float) -> dict:
            return {"macro_f1": f1, "per_class": {"neutral": {"f1": f1 - 0.2}}}

        return {
            "uit_validation": one(0.86),
            "uit_validation_stripped": one(0.84 - drop),
            "neu_validation": one(0.45),
            "neu_validation_all": one(0.44),
        }

    for s in SEEDS:
        for name, pred, drop in (
            (f"{H.RECIPE}-s{s}", y, stripped_drop),  # the candidate gets every post right
            (f"control-s{s}", np.full(len(y), "neutral"), 0.0),
        ):
            d = out / name
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({"sets": sets(drop)}), encoding="utf-8")
            pd.DataFrame({"pred": pred}).to_csv(d / "predictions_holdout.csv")
            pd.DataFrame({"orig": teacher, "norm": teacher}).to_csv(d / "predictions_confirm.csv")
    (out / "eligibility.json").write_text(json.dumps({"eligible": True}), encoding="utf-8")


def test_confirm_applies_the_rule_from_prediction_files(tmp_path, monkeypatch):
    _fake_study(tmp_path, monkeypatch)
    out = H.confirm()
    assert out["passed"] and abs(out["holdout_posts"] - 60) <= 2  # per-label rounding
    assert out["rules"]["1_holdout_macro_f1_up"]["ci_low"] > 0
    assert (tmp_path / "h12p" / "confirm" / "decision.json").exists()


def test_confirm_fails_on_a_guard(tmp_path, monkeypatch):
    _fake_study(tmp_path, monkeypatch, stripped_drop=0.02)
    out = H.confirm()
    assert not out["passed"]
    assert not out["rules"]["4_uit_validation_stripped_macro_f1"]["passed"]


def test_confirm_refuses_missing_files(tmp_path, monkeypatch):
    _fake_study(tmp_path, monkeypatch)
    (tmp_path / "h12p" / "control-s7" / "predictions_holdout.csv").unlink()
    with pytest.raises(FileNotFoundError, match="control-s7"):
        H.confirm()


def test_import_accepts_results_and_refuses_anything_else(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    reg = tmp_path / "registry.csv"
    pd.DataFrame({"run_id": ["old"], "macro_f1": [0.8]}).to_csv(reg, index=False)
    monkeypatch.setattr(paths, "REGISTRY", reg)
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("results/studies/cycle5/h12p/two_heads_anchored-s7/summary.json", "{}")
        z.writestr("registry_rows.csv", "run_id,macro_f1\nold,0.8\nnew,0.9\n")
    written = H.import_results(good)
    assert "registry.csv (+1 rows)" in written
    assert list(pd.read_csv(reg).run_id) == ["old", "new"]
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("src/vifeedback/evil.py", "x")
    with pytest.raises(ValueError, match="unexpected"):
        H.import_results(bad)
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as z:
        z.writestr("results/runs/p16-x/../../../../evil.txt", "x")
    with pytest.raises(ValueError, match="unexpected"):
        H.import_results(traversal)
