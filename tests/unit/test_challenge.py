"""Challenge set v1: the frozen file, and the scoring used by the H6 rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vifeedback.evaluation import challenge as CH

NEG, NEU, POS = 0, 1, 2


def _df(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """rows: (category, pair_id, sentiment)."""
    df = pd.DataFrame(rows, columns=["category", "pair_id", "sentiment"])
    df["y"] = df.sentiment.map({"negative": NEG, "neutral": NEU, "positive": POS})
    df["scored"] = df.y.notna()
    return df


class TestFrozenFile:
    def test_committed_file_matches_declared_hash(self):
        df = CH.load()
        assert len(df) == 305
        assert set(df.sentiment) == {"negative", "neutral", "positive", "none"}
        assert (df.scored == (df.category != CH.OUT_OF_SCOPE)).all()

    def test_crlf_checkout_hashes_the_same(self, tmp_path):
        crlf = CH.CHALLENGE_V1.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        p = tmp_path / "c.csv"
        p.write_bytes(crlf)
        assert len(CH.load(p)) == 305

    def test_edited_file_is_refused(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_bytes(CH.CHALLENGE_V1.read_bytes().replace(b"positive", b"negative", 1))
        with pytest.raises(ValueError, match="does not match"):
            CH.load(p)

    def test_empty_pair_id_is_not_missing(self):
        df = CH.load()
        assert (df.pair_id != "").sum() == 30
        assert df.pair_id.isna().sum() == 0


class TestNegationPairs:
    def test_counts_both_correct_and_flips(self):
        df = _df(
            [
                ("negation_pair", "n01", "positive"),
                ("negation_pair", "n01", "negative"),
                ("negation_pair", "n02", "positive"),
                ("negation_pair", "n02", "negative"),
                ("objective_neutral", "", "neutral"),
            ]
        )
        pred = np.array([POS, NEG, POS, POS, NEU])
        r = CH.negation_pairs(df, pred)
        assert r == {"pairs": 2, "both_correct": 0.5, "prediction_flips": 0.5}


class TestH6Rule:
    def _set(self, n_typed=60, n_other=60):
        rows = [("unaccented_typed", "", "positive")] * n_typed + [
            ("objective_neutral", "", "neutral")
        ] * n_other
        rows += [(CH.OUT_OF_SCOPE, "", "none")] * 5
        return _df(rows)

    def test_switch_when_typed_gain_is_clear_and_other_rows_hold(self):
        df = self._set()
        ce = np.array([NEG] * 60 + [NEU] * 60 + [NEU] * 5)  # typed all wrong
        aug = np.array([POS] * 60 + [NEU] * 60 + [POS] * 5)  # typed all right; OOS ignored
        r = CH.h6(df, ce, aug)
        assert r["typed_rows"] == 60 and r["other_rows"] == 60
        assert r["typed_paired_aug_minus_ce"]["ci_low"] > 0
        assert r["switch"] is True

    def test_no_switch_when_other_rows_lose_more_than_the_margin(self):
        df = self._set()
        ce = np.array([NEG] * 60 + [NEU] * 60 + [NEU] * 5)
        aug = np.array([POS] * 60 + [NEU] * 57 + [NEG] * 3 + [NEU] * 5)  # other: -0.05
        r = CH.h6(df, ce, aug)
        assert r["other_diff"] == pytest.approx(-0.05)
        assert r["switch"] is False

    def test_no_switch_without_a_typed_gain(self):
        df = self._set()
        same = np.array([POS] * 60 + [NEU] * 65)
        assert CH.h6(df, same, same.copy())["switch"] is False


def test_category_report_scores_only_labelled_rows():
    df = _df(
        [
            ("a", "", "positive"),
            ("a", "", "negative"),
            ("b", "", "neutral"),
            (CH.OUT_OF_SCOPE, "", "none"),
        ]
    )
    r = CH.category_report(df, np.array([POS, POS, NEU, NEG]))
    assert r["scored_rows"] == 3
    assert r["by_category"]["a"] == {"n": 2, "accuracy": 0.5}
    assert CH.OUT_OF_SCOPE not in r["by_category"]


class TestSeedPairedDrops:
    """Cycle 3 V1: a drop is confirmed only in >= 4 of 5 seeds and with pooled McNemar p < 0.05."""

    def _df(self, n=40):
        return _df([("teencode_typed", "", "positive")] * n)

    def test_consistent_drop_is_confirmed(self):
        df = self._df()
        right = np.full(40, POS)
        aug_pred = right.copy()
        aug_pred[:3] = NEG  # 3 more errors per seed, every seed
        ce = {s: right for s in ("1", "2", "3", "4", "5")}
        aug = {s: aug_pred for s in ce}
        r = CH.seed_paired_drops(df, ce, aug)["categories"]["teencode_typed"]
        assert r["seeds_augmented_lower"] == 5 and r["ce_only_right"] == 15
        assert r["confirmed_drop"] is True

    def test_one_seed_drop_is_not_confirmed(self):
        df = self._df()
        right = np.full(40, POS)
        worse = right.copy()
        worse[:4] = NEG
        ce = {s: right for s in ("1", "2", "3", "4", "5")}
        aug = {**ce, "1": worse}
        r = CH.seed_paired_drops(df, ce, aug)["categories"]["teencode_typed"]
        assert r["seeds_augmented_lower"] == 1
        assert r["confirmed_drop"] is False
