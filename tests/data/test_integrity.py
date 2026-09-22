"""Phase 0 dataset integrity suite — gate G0.

These tests fail the build if the upstream data ever changes under us, which is the point: every
number in this project is computed against the split sizes and label maps asserted here.

Marked `needs_data`; run `vifeedback data fetch` first.
"""

from __future__ import annotations

import pytest

from vifeedback.constants import EXPECTED_SPLIT_SIZES, EXPECTED_TOTAL, LABELS, SPLITS, TASKS
from vifeedback.data import integrity as I
from vifeedback.data.loader import MANIFEST, load_all

pytestmark = pytest.mark.needs_data


@pytest.fixture(scope="module")
def dfs():
    if not MANIFEST.exists():
        pytest.skip("data/raw not populated — run `vifeedback data fetch`")
    return load_all()


class TestSplits:
    def test_sizes_match_the_official_release(self, dfs) -> None:
        r = I.check_split_sizes(dfs)
        assert r["actual"] == EXPECTED_SPLIT_SIZES, r
        assert r["total"] == EXPECTED_TOTAL

    def test_schema(self, dfs) -> None:
        for split in SPLITS:
            assert list(dfs[split].columns) == ["sentence", "sentiment", "topic"]


class TestLabels:
    def test_ids_in_range_no_nulls_all_classes_present(self, dfs) -> None:
        r = I.check_labels(dfs)
        assert r["passed"], r

    @pytest.mark.parametrize("task", TASKS)
    def test_label_map_is_contiguous_from_zero(self, task: str) -> None:
        """A non-contiguous or re-ordered map silently transposes every confusion matrix."""
        assert sorted(LABELS[task]) == list(range(len(LABELS[task])))


class TestTextQuality:
    def test_no_empty_or_null_sentences(self, dfs) -> None:
        r = I.check_text_quality(dfs)
        assert r["passed"], r

    def test_text_is_nfc_composed_on_ingest(self, dfs) -> None:
        """The loader composes once; nothing downstream may compare NFC against NFD."""
        for split in SPLITS:
            forms = I.check_text_quality(dfs)["splits"][split]["unicode_forms"]
            assert "NFD" not in forms and "both" not in forms, (split, forms)


class TestLeakage:
    """The result is published whatever it is (docs/DATA_CARD.md § 6, L5).

    Thresholds are regression guards pinned to the measured values, not aspirations: they exist so
    that a future change to the dedup key or the data source shows up as a failing test.
    """

    def test_no_exact_duplicates_across_splits(self, dfs) -> None:
        head = I.leakage_summary(dfs)["headline"]
        assert head["test_rows_seen_in_train_exact"] == 0

    def test_near_duplicate_leakage_stays_immaterial(self, dfs) -> None:
        head = I.leakage_summary(dfs)["headline"]
        assert head["share_of_test_normalized"] < 0.05, (
            f"near-duplicate leakage rose to {head['share_of_test_normalized']:.2%}; "
            "re-read docs/DATA_CARD.md § 6 before trusting any headline metric"
        )


class TestDistribution:
    def test_neutral_is_the_minority_class_everywhere(self, dfs) -> None:
        """The premise of the whole macro-F1 framing (ADR-001)."""
        for split in SPLITS:
            counts = dfs[split]["sentiment"].value_counts()
            assert counts[1] == counts.min(), split
            assert counts[1] / counts.sum() < 0.10, split

    def test_facility_is_the_minority_topic_everywhere(self, dfs) -> None:
        for split in SPLITS:
            counts = dfs[split]["topic"].value_counts()
            assert counts[2] == counts.min(), split

    def test_sentiment_and_topic_are_not_independent(self, dfs) -> None:
        """Evidence for the Phase 4 multi-task experiment, asserted so it cannot silently vanish."""
        v = I.cramers_v(I.joint_distribution(dfs))
        assert v > 0.2, f"Cramér's V fell to {v:.3f}; the multi-task rationale no longer holds"


class TestSurfaceProperties:
    """Facts that decide which preprocessing conditions are meaningful (ADR-006)."""

    def test_corpus_is_already_lowercased(self, dfs) -> None:
        from vifeedback.data.profile import surface_properties

        assert surface_properties(dfs)["sentences_with_uppercase"] == 0

    def test_corpus_is_not_word_segmented(self, dfs) -> None:
        """If this fails, the P0/P1 segmentation ablation is measuring nothing."""
        from vifeedback.data.profile import surface_properties

        assert surface_properties(dfs)["sentences_with_underscore"] <= 5
