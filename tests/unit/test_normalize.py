"""Unit tests for Vietnamese normalization primitives."""

from __future__ import annotations

import unicodedata

import pytest

from vifeedback.preprocess import normalize as N


class TestStripDiacritics:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("không tốt", "khong tot"),
            ("Thầy dạy hay", "Thay day hay"),
            ("đường", "duong"),
            ("Đại học", "Dai hoc"),
            ("giảng viên nhiệt tình", "giang vien nhiet tinh"),
            ("ăn ắn ằn ẳn ẵn ặn", "an an an an an an"),
            ("", ""),
            ("abc 123 !", "abc 123 !"),
        ],
    )
    def test_strips_all_marks(self, text: str, expected: str) -> None:
        assert N.strip_diacritics(text) == expected

    def test_d_stroke_is_not_a_combining_mark(self) -> None:
        """`đ` does not decompose under NFD, so it needs an explicit mapping.

        Without it, diacritic stripping silently leaves `đ` behind and the perturbation suite
        tests a weaker shift than it claims to.
        """
        assert unicodedata.normalize("NFD", "đ") == "đ"
        assert N.strip_diacritics("đ") == "d"

    def test_output_is_ascii_for_vietnamese_input(self) -> None:
        assert N.strip_diacritics("Cộng hòa xã hội chủ nghĩa Việt Nam").isascii()

    def test_is_idempotent(self) -> None:
        once = N.strip_diacritics("không tốt lắm")
        assert N.strip_diacritics(once) == once

    def test_is_many_to_one(self) -> None:
        """The ambiguity the model must resolve, made explicit."""
        assert {N.strip_diacritics(w) for w in ("mà", "má", "mã", "mả", "mạ", "ma")} == {"ma"}


class TestUnicodeForm:
    def test_nfc_and_nfd_differ_bytewise_but_not_visually(self) -> None:
        nfc = unicodedata.normalize("NFC", "tốt")
        nfd = unicodedata.normalize("NFD", "tốt")
        assert nfc != nfd
        assert N.to_nfc(nfd) == nfc

    @pytest.mark.parametrize(
        ("text", "form"),
        [
            (unicodedata.normalize("NFC", "tốt"), "NFC"),
            (unicodedata.normalize("NFD", "tốt"), "NFD"),
            ("abc", "ascii"),
        ],
    )
    def test_detects_form(self, text: str, form: str) -> None:
        assert N.detect_unicode_form(text) == form


class TestWhitespaceAndElongation:
    def test_collapses_whitespace(self) -> None:
        assert N.collapse_whitespace("  a \t b \n c  ") == "a b c"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("hayyyy", "hayy"), ("quáaaaa", "quáaa"), ("tot", "tot"), ("aa", "aa")],
    )
    def test_caps_elongation_without_erasing_it(self, text: str, expected: str) -> None:
        assert N.collapse_elongation(text) == expected


class TestDedupKey:
    def test_matches_across_case_diacritics_and_punctuation(self) -> None:
        assert N.dedup_key("Thầy dạy hay!") == N.dedup_key("thay day hay .")

    def test_catches_the_d_stroke_typo_class(self) -> None:
        """`truyền đạt` vs `truyền dạt` is a real typo pattern in this corpus."""
        assert N.dedup_key("truyền đạt tốt .") == N.dedup_key("truyền dạt tốt .")

    def test_distinguishes_genuinely_different_sentences(self) -> None:
        assert N.dedup_key("thầy dạy hay") != N.dedup_key("thầy dạy dở")


class TestSyllableCount:
    @pytest.mark.parametrize(
        ("text", "n"),
        [("sinh viên", 2), ("giảng viên nhiệt tình", 4), ("", 0), ("  a  b  ", 2)],
    )
    def test_counts_whitespace_delimited_syllables(self, text: str, n: int) -> None:
        assert N.syllable_count(text) == n
