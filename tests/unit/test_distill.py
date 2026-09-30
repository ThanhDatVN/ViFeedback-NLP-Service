"""Cycle 5 H11 building blocks (training/distill.py) on tiny models: students, loss, soft labels."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vifeedback.training import distill as K  # noqa: E402


def _tiny_teacher(layers: int = 12):
    torch.manual_seed(0)
    cfg = transformers.RobertaConfig(
        vocab_size=300,
        hidden_size=32,
        num_hidden_layers=layers,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=40,
        num_labels=3,
        pad_token_id=1,
    )
    return transformers.RobertaForSequenceClassification(cfg).eval()


def test_teacher_alternate_student_copies_layers_1_3_5_7_9_11_and_the_head():
    t = _tiny_teacher()
    s = K.student_from_teacher(t)
    assert s.config.num_hidden_layers == 6 and t.config.num_hidden_layers == 12
    for j, i in enumerate(K.ALTERNATE):
        a = s.roberta.encoder.layer[j].state_dict()
        b = t.roberta.encoder.layer[i].state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert torch.equal(s.classifier.out_proj.weight, t.classifier.out_proj.weight)
    assert torch.equal(
        s.roberta.embeddings.word_embeddings.weight, t.roberta.embeddings.word_embeddings.weight
    )


def test_pretrained_first6_student_keeps_the_first_six_layers(tmp_path):
    t = _tiny_teacher()
    t.save_pretrained(tmp_path)
    s = K.student_from_pretrained(str(tmp_path))
    assert s.config.num_hidden_layers == 6 and s.config.num_labels == 3
    for i in range(6):
        a, b = s.roberta.encoder.layer[i].state_dict(), t.roberta.encoder.layer[i].state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_loss_is_zero_kl_when_the_student_matches_the_teacher():
    logits = torch.tensor([[1.0, 0.2, -0.5], [0.3, 0.3, 2.0]])
    q = torch.softmax(logits / K.TEMPERATURE, -1)
    unlabeled = torch.tensor([-1, -1])
    assert abs(float(K.distill_loss(logits, q, unlabeled))) < 1e-6


def test_loss_mixes_ce_only_on_labelled_rows():
    import torch.nn.functional as F

    logits = torch.tensor([[1.0, 0.2, -0.5], [0.3, 0.3, 2.0]])
    q = torch.softmax(logits / K.TEMPERATURE, -1)  # KL term 0 on both rows
    labels = torch.tensor([0, -1])
    got = float(K.distill_loss(logits, q, labels))
    ce = float(F.cross_entropy(logits[:1], labels[:1]))
    assert got == pytest.approx((1 - K.ALPHA) * ce / 2, rel=1e-5)  # mean over both rows


def test_kl_is_scaled_by_t_squared():
    logits = torch.zeros(1, 3)
    q = torch.tensor([[0.8, 0.1, 0.1]])
    kl = float((q * (q.log() - torch.log_softmax(logits / 2.0, -1))).sum())
    got = float(K.distill_loss(logits, q, torch.tensor([-1]), temperature=2.0))
    assert got == pytest.approx(4 * kl, rel=1e-5)


def test_tempered_probs_flatten_with_temperature():
    z = np.array([[3.0, 0.0, 0.0]])
    assert K.tempered_probs(z, 2.0)[0, 0] < K.tempered_probs(z, 1.0)[0, 0]
    assert np.allclose(K.tempered_probs(z, 2.0).sum(1), 1)


def test_soft_label_cache_round_trips_and_deduplicates(tmp_path):
    c = K.SoftLabelCache(tmp_path / "c.npz")
    assert c.missing(["a", "b", "a"]) == ["a", "b"]
    c.add(["a", "b"], np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]))
    c.save()
    d = K.SoftLabelCache(tmp_path / "c.npz")
    assert d.missing(["a", "b", "c"]) == ["c"]
    assert np.allclose(d.get(["b", "a"]), [[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])


def test_ensemble_averages_the_teachers_and_scores_each_text_once(tmp_path, monkeypatch):
    calls = []
    fixed = {"t1": np.array([[2.0, 0.0, 0.0]]), "t2": np.array([[0.0, 2.0, 0.0]])}

    def fake_logits(model, tok, texts):
        calls.append((model, list(texts)))
        return np.repeat(fixed[model], len(texts), axis=0)

    monkeypatch.setattr(K, "teacher_logits", fake_logits)
    cache = K.SoftLabelCache(tmp_path / "c.npz")

    def load(ck):
        return ck, None

    out = K.ensemble_soft_labels(["t1", "t2"], ["x", "x", "y"], cache, verbose=False, load=load)
    want = (
        K.tempered_probs(fixed["t1"], K.TEMPERATURE) + K.tempered_probs(fixed["t2"], K.TEMPERATURE)
    ) / 2
    assert np.allclose(out, np.repeat(want, 3, axis=0))
    assert [c[1] for c in calls] == [["x", "y"], ["x", "y"]]
    K.ensemble_soft_labels(["t1", "t2"], ["y", "x"], cache, verbose=False, load=load)
    assert len(calls) == 2  # cached: no teacher runs again


def test_config_carries_the_init_and_teacher_in_the_run_id():
    a = K.config("teacher-alternate", 42, "served")
    b = K.config("pretrained-first6", 42, "served")
    assert a.run_id().startswith(
        "p14-sent-phobert-base-seg_pyvi-h11-teacher-alternate-from-served-s42-"
    )
    assert a.config_hash() != b.config_hash()
    with pytest.raises(ValueError):
        K.config("random", 42, "served")


# --- orchestration (cycle5.yaml v2 H11) -----------------------------------------------------------


def test_drop_bound_is_zero_for_identical_predictions_and_positive_for_a_worse_student():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 800)
    teach = [np.where(rng.random(800) < 0.2, rng.integers(0, 3, 800), y) for _ in range(3)]
    same = K.seed_averaged_drop_bound(y, teach, teach, n_resamples=400)
    assert same["observed_drop"] == 0 and same["upper_95_one_sided"] == 0
    worse = [np.where(rng.random(800) < 0.3, rng.integers(0, 3, 800), t) for t in teach]
    got = K.seed_averaged_drop_bound(y, teach, worse, n_resamples=400)
    assert got["observed_drop"] > 0.05 and got["upper_95_one_sided"] > got["observed_drop"]


PARITY = {"max_abs_logit_diff": 1e-4, "label_agreement": 1.0, "batch_vs_single_max_diff": 1e-4}
DROPS = {"uit": 0.01, "neutral": 0.03, "stripped": 0.015}
POOLED = {"observed_drop": 0.004, "upper_95_one_sided": 0.01}


def test_h11_rule_passes_at_its_limits():
    out = K.apply_rule(200 * 10**6, POOLED, DROPS, PARITY)
    assert out["passed"]


@pytest.mark.parametrize(
    "change",
    [
        {"size": 200 * 10**6 + 1},
        {"pooled": {**POOLED, "upper_95_one_sided": 0.0101}},
        {"drops": {**DROPS, "uit": 0.0101}},
        {"drops": {**DROPS, "neutral": 0.0301}},
        {"drops": {**DROPS, "stripped": 0.0151}},
        {"parity": {**PARITY, "label_agreement": 0.9994}},
        {"parity": {**PARITY, "batch_vs_single_max_diff": 2e-4}},
    ],
)
def test_h11_rule_fails_past_any_limit(change):
    out = K.apply_rule(
        change.get("size", 185 * 10**6),
        change.get("pooled", POOLED),
        change.get("drops", DROPS),
        change.get("parity", PARITY),
    )
    assert not out["passed"]


def _h10(tmp, monkeypatch, selection=None, confirm=None):
    import json

    from vifeedback.training import consistency as C

    monkeypatch.setattr(C, "OUT", tmp)
    if selection is not None:
        (tmp / "selection.json").write_text(json.dumps({"outcome": selection}), encoding="utf-8")
    if confirm is not None:
        d = tmp / "confirm-onesided"
        d.mkdir(parents=True)
        (d / "decision.json").write_text(json.dumps({"passed": confirm}), encoding="utf-8")


def test_teacher_waits_for_h10(tmp_path, monkeypatch):
    _h10(tmp_path, monkeypatch, selection="confirm")
    with pytest.raises(RuntimeError, match="H10 has not decided"):
        K.resolve_teacher()


def test_teacher_is_h10_recipe_only_if_h10_passed(tmp_path, monkeypatch):
    _h10(tmp_path, monkeypatch, selection="confirm", confirm=True)
    name, ckpts, _ = K.resolve_teacher()
    assert name == "h10-onesided" and sorted(ckpts) == sorted([42, 1337, 2024, 7, 31337])
    assert ckpts[42].name.startswith("p13-sent-phobert-base-seg_pyvi-h10-onesided-s42-")


def test_teacher_is_the_served_recipe_when_h10_fails(tmp_path, monkeypatch):
    from vifeedback.training import domain as D

    monkeypatch.setattr(D, "control_checkpoint", lambda s: tmp_path / f"served-{s}")
    _h10(tmp_path, monkeypatch, selection="confirm", confirm=False)
    assert K.resolve_teacher()[0] == "served"
    other = tmp_path / "stopped"
    other.mkdir()
    _h10(other, monkeypatch, selection="not supported: no recipe is eligible")
    assert K.resolve_teacher()[0] == "served"


def test_selection_takes_the_higher_mean(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(K, "OUT", tmp_path)
    for init, uit, neu in (("teacher-alternate", 0.86, 0.45), ("pretrained-first6", 0.84, 0.49)):
        d = tmp_path / f"{init}-s42"
        d.mkdir()
        sets = {"uit_validation": {"macro_f1": uit}, "neu_validation_all": {"macro_f1": neu}}
        (d / "summary.json").write_text(json.dumps({"sets": sets}), encoding="utf-8")
    assert K.select()["chosen"] == "pretrained-first6"  # 0.665 > 0.655
