"""Unit tests for the robustness suites.

Pinned: determinism independent of order (so an example can be regenerated alone), labels never
touched, the changed-mask being honest, no transformation that silently does nothing, and a grouped
bootstrap that really keeps an example's variants together.
"""

from __future__ import annotations

import numpy as np
import pytest

from vifeedback.evaluation import robustness as R

TEXTS = [
    "thầy dạy không hay như thế nào vậy",
    "giảng viên nhiệt tình được",
    "phòng học rất nóng nhưng thầy vui tính",
    "ok",
    "sinh viên mong được học thêm",
]


class TestDeterminism:
    @pytest.mark.parametrize("suite", sorted(R.SUITES))
    def test_same_seed_same_output(self, suite: str) -> None:
        assert R.perturb(TEXTS, suite, seed=1)[0] == R.perturb(TEXTS, suite, seed=1)[0]

    @pytest.mark.parametrize("suite", ["teencode-30", "charnoise-10", "nodiacritic-50"])
    def test_output_does_not_depend_on_position_in_the_batch(self, suite: str) -> None:
        """Example i's rng is keyed on its index, so running it alone reproduces it."""
        full = R.perturb(TEXTS, suite, seed=7)[0]
        # Re-run only example 2 at index 2 by padding the batch with different sentences.
        other = ["a", "b", TEXTS[2]]
        assert R.perturb(other, suite, seed=7)[0][2] == full[2]

    def test_unknown_suite_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown suite"):
            R.perturb(TEXTS, "nope")


class TestTransformations:
    def test_nodiacritic_strips_everything_including_d_stroke(self) -> None:
        out, changed = R.perturb(["đi học được"], "nodiacritic")
        assert out == ["di hoc duoc"] and changed.all()

    def test_teencode_100_replaces_every_dictionary_word_and_multiword_first(self) -> None:
        out = R.perturb(["giảng viên không biết như thế nào"], "teencode-100")[0][0]
        assert "gv" in out.split() and "ntn" in out.split()
        assert "không" not in out.split() and "giảng" not in out.split()

    def test_no_teencode_variant_equals_its_source(self) -> None:
        """A variant identical to the source would count as 'unchanged' and dilute the suite."""
        for src, variants in R.TEENCODE.items():
            assert src not in variants, src

    def test_charnoise_rate_zero_is_identity(self) -> None:
        rng = np.random.default_rng(0)
        assert R.charnoise(TEXTS[0], rng, rate=0.0) == TEXTS[0]

    def test_changed_mask_matches_the_text(self) -> None:
        out, changed = R.perturb(TEXTS, "teencode-30", seed=3)
        assert changed.tolist() == [a != b for a, b in zip(TEXTS, out, strict=True)]

    def test_every_suite_changes_something_on_realistic_text(self) -> None:
        for suite in R.SUITES:
            _, changed = R.perturb(TEXTS * 4, suite, seed=0)
            assert changed.any(), f"{suite} changed nothing"


class TestSlices:
    def test_whole_token_matching(self) -> None:
        m = R.slice_masks(["thầy không dạy", "khôngkhí tốt", "colonsmile hay", "wzjwz12 dạy"])
        assert m["negation"].tolist() == [True, False, False, False]
        assert m["emoticon_token"].tolist() == [False, False, True, False]
        assert m["anonymized_name"].tolist() == [False, False, False, True]
        assert m["short_lt5"].all()

    def test_slice_report_flags_low_support(self) -> None:
        y = np.array([0, 1, 2, 0])
        rows = R.slice_report(y, y, {"all": np.ones(4, bool)}, k=3, min_support=30)
        assert rows[0]["n"] == 4 and rows[0]["reliable"] is False
        assert rows[0]["macro_f1"] == pytest.approx(1.0)


class TestPairedDelta:
    def test_identical_predictions_give_zero_delta_and_zero_width(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 300)
        p = np.where(rng.random(300) < 0.8, y, (y + 1) % 3)
        r = R.paired_delta(y, p, p, 3, n_boot=200)
        assert r["delta"] == 0 and r["delta_ci"] == [0.0, 0.0] and r["flip_rate"] == 0

    def test_grouping_widens_the_interval_for_duplicated_variants(self) -> None:
        """Ten copies of each example are not ten times the evidence."""
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, 60)
        a = np.where(rng.random(60) < 0.85, y, (y + 1) % 3)
        b = np.where(rng.random(60) < 0.70, y, (y + 2) % 3)
        yy, aa, bb = (np.repeat(v, 10) for v in (y, a, b))
        naive = R.paired_delta(yy, aa, bb, 3, n_boot=500)
        grouped = R.paired_delta(yy, aa, bb, 3, groups=np.repeat(np.arange(60), 10), n_boot=500)
        width = lambda r: r["delta_ci"][1] - r["delta_ci"][0]  # noqa: E731
        assert grouped["n_groups"] == 60
        assert width(grouped) > 2 * width(naive)


class TestNegationProbe:
    def _pairs(self):
        import pandas as pd

        from vifeedback import paths

        return pd.read_csv(paths.DATA / "probes" / "negation_v1.csv")

    def test_probe_file_is_wellformed(self) -> None:
        d = self._pairs()
        assert d.pair_id.is_unique
        assert set(d.set) == {"pos_to_neg", "neg_to_pos"}
        pn = d[d.set == "pos_to_neg"]
        assert (pn.base_label == "positive").all() and (pn.negated_label == "negative").all()
        for _, r in d.iterrows():
            assert r.cue in r.negated.split(), f"pair {r.pair_id}: cue missing from negated text"

    def test_a_constant_classifier_gets_zero_pairs(self) -> None:
        """Always predicting negative is half right on every pair and right on none."""
        r = R.negation_probe(self._pairs(), lambda xs: np.zeros(len(xs), dtype=int))
        assert r["pos_to_neg"]["pair_accuracy"] == 0.0
        assert r["pos_to_neg"]["flip_rate"] == 0.0

    def test_an_oracle_gets_every_pair(self) -> None:
        d = self._pairs()
        lookup = {
            **dict(zip(d.base, d.base_label, strict=True)),
            **dict(zip(d.negated, d.negated_label, strict=True)),
        }
        ids = {"negative": 0, "neutral": 1, "positive": 2}
        r = R.negation_probe(d, lambda xs: np.array([ids[lookup[x]] for x in xs]))
        assert all(v["pair_accuracy"] == 1.0 for v in r.values())


def test_macro_f1_is_unreliable_when_a_class_is_nearly_absent() -> None:
    """Large n, one neutral example: n passes, macro-F1 must not."""
    y = np.array([0] * 200 + [1] + [2] * 40)
    rows = R.slice_report(y, y, {"s": np.ones(len(y), bool)}, k=3)
    assert rows[0]["reliable"] is True
    assert rows[0]["macro_f1_reliable"] is False
