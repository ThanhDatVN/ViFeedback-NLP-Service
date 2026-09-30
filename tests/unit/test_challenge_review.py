"""U2 sensitivity view (evaluation/challenge_review.py): decisions are validated and applied per row."""

from __future__ import annotations

import pandas as pd
import pytest

from vifeedback.evaluation import challenge_review as CR


def _review(tmp_path, decisions):
    rows = [{"id": i, "owner_decision": d, "owner_note": ""} for i, d in decisions.items()]
    p = tmp_path / "review.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_blank_decisions_mean_not_reviewed(tmp_path):
    assert CR.read_decisions(_review(tmp_path, {"c1": "", "c2": " "})) == {}


def test_decisions_are_normalized_and_validated(tmp_path):
    got = CR.read_decisions(_review(tmp_path, {"c1": " Keep", "c2": "neutral", "c3": ""}))
    assert got == {"c1": "keep", "c2": "neutral"}
    with pytest.raises(ValueError, match="owner_decision must be one of"):
        CR.read_decisions(_review(tmp_path, {"c1": "trung tính"}))


def test_reviewed_labels_relabel_keep_and_drop_ambiguous():
    gold = pd.Series({"c1": "positive", "c2": "negative", "c3": "neutral"})
    rev, keep = CR.reviewed_labels(gold, {"c1": "keep", "c2": "neutral", "c3": "ambiguous"})
    assert rev.to_dict() == {"c1": "positive", "c2": "neutral", "c3": "neutral"}
    assert keep.to_dict() == {"c1": True, "c2": True, "c3": False}
    assert gold["c2"] == "negative"  # the frozen labels are not modified


def test_sensitivity_on_the_committed_files_is_consistent():
    """With every flagged row kept, the reviewed view equals the frozen one for every system."""
    review = pd.read_csv(CR.REVIEW)
    out = CR.sensitivity({str(i): "keep" for i in review["id"]})
    assert out["decisions"]["keep"] == len(review) and out["relabelled_rows"] == {}
    assert {"ce", "augmented (served)"} <= set(out["systems"])
    for v in out["systems"].values():
        assert all(abs(d) < 1e-12 for d in v["delta"].values())
