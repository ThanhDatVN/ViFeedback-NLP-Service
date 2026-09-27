"""Neutral audit analysis: validation, Wilson intervals, κ and the frozen decision tree."""

from __future__ import annotations

import pandas as pd
import pytest

from vifeedback.evaluation import audit as A


def _sheet(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    """rows: (stratum, gold, annotator_label, gold_assessment, neutral_subtype)."""
    df = pd.DataFrame(
        rows, columns=["stratum", "gold", "annotator_label", "gold_assessment", "neutral_subtype"]
    )
    df.insert(0, "split", "validation")
    df.insert(1, "example_index", range(len(df)))
    df["notes"] = ""
    return df


def _scope(
    n_incorrect: int, n_ambiguous: int, n_agree: int
) -> list[tuple[str, str, str, str, str]]:
    rows = [("error:neutral->negative", "neutral", "negative", "incorrect", "n/a")] * n_incorrect
    rows += [
        ("error:neutral->positive", "neutral", "ambiguous", "ambiguous", "mixed")
    ] * n_ambiguous
    rows += [("error:positive->neutral", "positive", "positive", "agree", "n/a")] * n_agree
    return rows


class TestWilson:
    def test_known_values(self):
        lo, hi = A.wilson(5, 20)
        assert lo == pytest.approx(0.1119, abs=1e-3)
        assert hi == pytest.approx(0.4687, abs=1e-3)

    def test_zero_successes_has_a_positive_upper_bound(self):
        lo, hi = A.wilson(0, 40)
        assert lo == 0.0 and 0.05 < hi < 0.1


class TestKappa:
    def test_perfect_and_chance(self):
        a = pd.Series(["neutral", "positive", "negative", "positive"])
        assert A.cohen_kappa(a, a) == pytest.approx(1.0)
        b = pd.Series(["x", "x", "y", "y"])
        c = pd.Series(["x", "y", "x", "y"])
        assert A.cohen_kappa(b, c) == pytest.approx(0.0)


class TestDecisionTree:
    def test_incorrect_branch_first(self):
        d = A.decide(A._clean(_sheet(_scope(3, 5, 2))))
        assert d["share_incorrect"] == pytest.approx(0.3)
        assert d["branch"] == "incorrect_gold"

    def test_ambiguous_branch(self):
        d = A.decide(A._clean(_sheet(_scope(2, 4, 4))))
        assert d["branch"] == "policy_ceiling"

    def test_representation_otherwise(self):
        d = A.decide(A._clean(_sheet(_scope(1, 3, 6))))
        assert d["branch"] == "representation"

    def test_rows_outside_scope_do_not_count(self):
        rows = _scope(0, 0, 10) + [("random", "neutral", "negative", "incorrect", "n/a")] * 20
        assert A.decide(A._clean(_sheet(rows)))["branch"] == "representation"


class TestReport:
    def test_incomplete_sheet_is_refused(self):
        s = _sheet(_scope(1, 1, 1))
        s.loc[0, "gold_assessment"] = ""
        with pytest.raises(ValueError, match="not complete"):
            A.report(s)

    def test_neutral_rows_need_a_subtype(self):
        s = _sheet([("random", "neutral", "neutral", "agree", "")])
        assert any("need a subtype" in p for p in A.validate(s))

    def test_random_stratum_and_agreement(self):
        rows = [("random", "positive", "positive", "agree", "n/a")] * 38
        rows += [("random", "neutral", "negative", "incorrect", "no_opinion")] * 2
        first = _sheet(rows + _scope(1, 1, 8))
        second = first.copy()
        second.loc[0, "annotator_label"] = "negative"  # one disagreement
        r = A.report(first, second, kind="intra")
        assert r["random_stratum"]["n"] == 40
        assert r["random_stratum"]["incorrect"]["k"] == 2
        assert r["agreement"]["n"] == 50 and r["agreement"]["kind"] == "intra"
        assert r["agreement"]["raw_agreement"] == pytest.approx(49 / 50)
        assert r["agreement"]["warnings"] == []
