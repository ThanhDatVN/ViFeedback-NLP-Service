"""Unit tests for the Study A error-analysis tooling.

What is pinned here is what would silently bias an audit: folds that drop the minority class, a
"random" stratum that overlaps the targeted ones, uncertainty scores with the wrong orientation, and
an annotation sheet that shows the model's answer before the annotator's own column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vifeedback.evaluation import error_analysis as EA

# UIT-VSFC train sentiment counts, measured at Gate G0.
VSFC_COUNTS = (5325, 458, 5643)


def _labels() -> np.ndarray:
    return np.concatenate([np.full(c, i) for i, c in enumerate(VSFC_COUNTS)])


def _table(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.45, 0.1, 0.45])
    logits = rng.normal(size=(n, 3)) + 2.5 * np.eye(3)[y]
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    texts = [f"giảng viên {'không ' if i % 4 == 0 else ''}nhiệt tình {i}" for i in range(n)]
    return EA.prediction_table(texts, texts, y, p, "sentiment", split="validation", source="test")


class TestStratifiedFolds:
    def test_folds_partition_the_data(self) -> None:
        y = _labels()
        folds = EA.stratified_folds(y, k=5, seed=42)
        held = np.concatenate([ho for _, ho in folds])
        assert len(held) == len(y) and len(np.unique(held)) == len(y)
        for tr, ho in folds:
            assert not set(tr) & set(ho)
            assert len(tr) + len(ho) == len(y)

    def test_every_fold_keeps_the_minority_class_proportion(self) -> None:
        """458 neutral examples: each held-out fold must carry ~92 of them, not a lucky few."""
        y = _labels()
        for _, ho in EA.stratified_folds(y, k=5, seed=42):
            n_neutral = int((y[ho] == 1).sum())
            assert abs(n_neutral - 458 / 5) <= 1

    def test_is_deterministic_in_the_seed(self) -> None:
        y = _labels()
        a = EA.stratified_folds(y, seed=7)
        b = EA.stratified_folds(y, seed=7)
        c = EA.stratified_folds(y, seed=8)
        assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b, strict=True))
        assert not all(np.array_equal(x[1], z[1]) for x, z in zip(a, c, strict=True))


class TestUncertainty:
    def test_all_scores_rank_a_uniform_row_above_a_confident_row(self) -> None:
        p = np.array([[0.98, 0.01, 0.01], [1 / 3, 1 / 3, 1 / 3]])
        for name, score in EA.uncertainty(p).items():
            assert score[1] > score[0], f"{name} is oriented the wrong way"

    def test_entropy_is_normalized_to_unit_interval(self) -> None:
        p = np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
        h = EA.uncertainty(p)["entropy"]
        assert h[0] == pytest.approx(0.0, abs=1e-6)
        assert h[1] == pytest.approx(1.0, abs=1e-6)


class TestFlags:
    def test_whole_token_matching(self) -> None:
        """'không' inside another word must not fire; as its own token it must."""
        f = EA.linguistic_flags(["thầy không dạy", "khôngkhí tốt", "rất hay nhưng khó"])
        assert f.negation.tolist() == [True, False, False]
        assert f.contrast.tolist() == [False, False, True]
        assert f.intensifier.tolist() == [False, False, True]


class TestAuditSample:
    def test_strata_are_disjoint_and_labelled(self) -> None:
        s = EA.stratified_audit_sample(_table(), per_error_cell=4, per_correct_class=3, n_random=10)
        assert s.example_index.is_unique, "an example may appear in only one stratum"
        kinds = {x.split(":")[0] for x in s.stratum}
        assert kinds == {"error", "correct_uncertain", "random"}

    def test_random_component_is_present_and_sized(self) -> None:
        s = EA.stratified_audit_sample(_table(), n_random=15)
        assert (s.stratum == "random").sum() == 15

    def test_error_cells_take_the_most_confident_errors_first(self) -> None:
        t = _table(n=600)
        s = EA.stratified_audit_sample(t, per_error_cell=2, per_correct_class=0, n_random=0)
        for stratum, grp in s.groupby("stratum"):
            gold, pred = stratum.split(":")[1].split("->")
            cell = t[(t.gold == gold) & (t.pred == pred)]
            assert grp.p_pred.min() >= cell.p_pred.nlargest(2).min() - 1e-12

    def test_is_reproducible(self) -> None:
        a = EA.stratified_audit_sample(_table(), seed=3)
        b = EA.stratified_audit_sample(_table(), seed=3)
        assert a.example_index.tolist() == b.example_index.tolist()


class TestAnnotationSheet:
    def test_annotator_columns_precede_model_columns(self, tmp_path) -> None:
        """Annotators label before seeing the model: its columns must come last."""
        s = EA.stratified_audit_sample(_table())
        path = EA.export_annotation_sheet(s, tmp_path / "sheet.csv")
        cols = pd.read_csv(path, encoding="utf-8-sig").columns.tolist()
        assert cols.index("annotator_label") < cols.index("pred")
        assert cols.index("gold_assessment") < cols.index("p_pred")
        for c in EA.ANNOTATION_COLUMNS:
            assert c in cols

    def test_vietnamese_text_round_trips(self, tmp_path) -> None:
        s = EA.stratified_audit_sample(_table())
        path = EA.export_annotation_sheet(s, tmp_path / "sheet.csv")
        back = pd.read_csv(path, encoding="utf-8-sig")
        assert back.text.str.contains("giảng viên").all()


class TestSummaries:
    def test_suspected_issues_are_errors_sorted_by_score(self) -> None:
        t = _table(n=600)
        issues = EA.suspected_label_issues(t, top_k=20)
        assert (~issues.correct).all()
        assert issues.issue_score.is_monotonic_decreasing

    def test_error_rate_by_flag_reports_support(self) -> None:
        r = EA.error_rate_by_flag(_table())
        neg = r[(r.flag == "negation") & r.present]
        assert len(neg) == 1 and neg.n.iloc[0] == 75  # every 4th synthetic sentence

    def test_fold_disagreement(self) -> None:
        agree = np.tile(np.array([[[0.9, 0.05, 0.05]]]), (5, 1, 1))
        split = np.array([[[0.9, 0.05, 0.05]]] * 3 + [[[0.05, 0.9, 0.05]]] * 2)
        assert EA.fold_disagreement(agree)["vote_agreement"][0] == 1.0
        assert EA.fold_disagreement(split)["vote_agreement"][0] == pytest.approx(0.6)


class TestNeutralDiagnosis:
    def test_counts_and_directions(self) -> None:
        t = _table(n=600)
        d = EA.neutral_diagnosis(t)
        ng = t[t.gold == "neutral"]
        assert d["neutral_support"] == len(ng)
        assert sum(d["neutral_errors_to"].values()) == int((~ng.correct).sum())
        assert "neutral" not in d["neutral_errors_to"]


class TestClassBias:
    def test_recovers_a_suppressed_minority_class(self) -> None:
        """Neutral logits pushed down by a constant: a positive bias must undo it."""
        rng = np.random.default_rng(0)
        y = rng.choice(3, size=3000, p=[0.45, 0.1, 0.45])
        logits = rng.normal(size=(3000, 3)) + 3.0 * np.eye(3)[y]
        logits[:, 1] -= 2.0
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        r = EA.tune_class_bias(p, y, cls=1, k=3)
        assert r["bias"] > 1.0
        assert r["macro_f1"] > r["at_zero"]

    def test_zero_bias_is_identity(self) -> None:
        p = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
        assert EA.apply_class_bias(p, 1, 0.0).tolist() == [0, 2]
