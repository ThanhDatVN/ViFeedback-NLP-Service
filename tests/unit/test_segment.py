"""Unit tests for the word-segmentation backends (Phase 3 / hypothesis H2).

The central risk these guard against is a **silent no-op**: a backend that returns its input
unchanged would show up in the ablation as "segmentation makes no difference", which is a completely
different claim from the one the experiment is meant to test.
"""

from __future__ import annotations

import pytest

from vifeedback.preprocess import segment as S

# Multi-syllable Vietnamese words that any competent segmenter must join.
JOINABLE = [
    ("sinh viên", "sinh_viên"),
    ("giảng viên", "giảng_viên"),
    ("thời gian", "thời_gian"),
]


class TestAvailability:
    def test_reports_every_backend(self) -> None:
        avail = S.available()
        assert set(avail) == set(S.BACKENDS)
        for name, info in avail.items():
            assert isinstance(info["available"], bool), name
            assert info["reason"], f"{name} must explain its status"

    def test_none_is_always_available(self) -> None:
        assert S.available()["none"]["available"]

    def test_vncorenlp_status_follows_jvm_presence(self) -> None:
        """Risk R1 made testable: no JVM means RDRSegmenter is not measurable on this machine, and
        the ablation must record that rather than crash."""
        info = S.available()["vncorenlp"]
        if not S.java_available():
            assert not info["available"]
            assert "JVM" in info["reason"]


class TestNoSegmenter:
    def test_is_the_identity(self) -> None:
        texts = ["giảng viên nhiệt tình .", "phòng học quá nóng ."]
        assert S.NoSegmenter()(texts) == texts

    def test_returns_a_new_list(self) -> None:
        texts = ["a", "b"]
        assert S.NoSegmenter()(texts) is not texts


@pytest.mark.skipif(
    not S.available()["underthesea"]["available"], reason="underthesea not installed"
)
class TestUndertheseaSegmenter:
    @pytest.fixture(scope="class")
    def seg(self):
        return S.UndertheseaSegmenter()

    @pytest.mark.parametrize(("raw", "expected_substring"), JOINABLE)
    def test_joins_multisyllable_words(self, seg, raw: str, expected_substring: str) -> None:
        assert expected_substring in seg([raw])[0]

    def test_actually_changes_the_input(self, seg) -> None:
        """The no-op guard. If this fails, every Phase 3 conclusion is void."""
        raw = ["giảng viên nhiệt tình với sinh viên ."]
        out = seg(raw)
        assert out[0] != raw[0]
        assert "_" in out[0]

    def test_preserves_sentence_count(self, seg) -> None:
        raw = ["sinh viên .", "giảng viên .", "phòng học ."]
        assert len(seg(raw)) == len(raw)

    def test_is_deterministic(self, seg) -> None:
        raw = ["chương trình đào tạo rất tốt ."]
        assert seg(raw) == seg(raw)

    def test_reduces_whitespace_token_count(self, seg) -> None:
        """Joining syllables must shrink the whitespace-token count — the mechanical signature of
        segmentation actually having happened."""
        raw = ["giảng viên nhiệt tình với sinh viên ."]
        out = seg(raw)
        assert len(out[0].split()) < len(raw[0].split())


class TestSegmentationStats:
    def test_detects_a_real_change(self) -> None:
        raw = ["sinh viên tốt", "giảng viên hay"]
        seg = ["sinh_viên tốt", "giảng_viên hay"]
        st = S.segmentation_stats(raw, seg)
        assert st["sentences_changed"] == 2
        assert st["changed_share"] == 1.0
        assert st["underscores_added"] == 2
        assert st["token_reduction"] > 0

    def test_detects_a_silent_no_op(self) -> None:
        raw = ["sinh viên tốt", "giảng viên hay"]
        st = S.segmentation_stats(raw, list(raw))
        assert st["sentences_changed"] == 0
        assert st["underscores_added"] == 0
        assert st["token_reduction"] == 0.0

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            S.segmentation_stats(["a", "b"], ["a"])


class TestFactory:
    def test_rejects_unknown_backends(self) -> None:
        with pytest.raises(ValueError, match="unknown segmenter"):
            S.get_segmenter("nonsense")

    def test_builds_the_identity_backend(self) -> None:
        assert isinstance(S.get_segmenter("none"), S.NoSegmenter)

    @pytest.mark.skipif(S.java_available(), reason="a JVM is present here")
    def test_vncorenlp_fails_with_an_actionable_message_without_java(self) -> None:
        with pytest.raises(RuntimeError, match="JVM"):
            S.get_segmenter("vncorenlp")
