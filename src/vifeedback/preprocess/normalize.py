"""Vietnamese text normalization primitives.

These functions serve two opposite purposes and are deliberately kept in one place:

* **defensively** — building the preprocessing variants of docs/DATA_CARD.md § 8, and the normalized
  key used for cross-split near-duplicate detection;
* **offensively** — generating the `nodiacritic` / `charnoise` perturbation suites that quantify
  robustness in Phase 5 (docs/EVALUATION_PROTOCOL.md § 7).

Every function is pure, deterministic and unit-tested.
"""

from __future__ import annotations

import re
import unicodedata

# `đ` and `Đ` are distinct Vietnamese letters, not `d` + a combining mark, so NFD decomposition
# leaves them untouched. They must be mapped explicitly or diacritic stripping silently misses them.
_D_STROKE = str.maketrans({"đ": "d", "Đ": "D"})

_WS_RE = re.compile(r"\s+")
# Vietnamese letters (precomposed ranges) plus ASCII, digits and sentence punctuation.
_ELONGATION_RE = re.compile(r"(.)\1{2,}", re.UNICODE)


def to_nfc(text: str) -> str:
    """Canonical composition.

    Vietnamese diacritics round-trip through NFC and NFD to *visually identical* but bytewise
    different strings. Mixing the two silently breaks dictionary lookups, duplicate detection and
    vocabulary matching, so every text entering the pipeline is composed exactly once, here.
    """
    return unicodedata.normalize("NFC", text)


def detect_unicode_form(text: str) -> str:
    """Return 'NFC', 'NFD', 'both' or 'ascii' — reported in the data card."""
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)
    if nfc == nfd:
        return "ascii"
    if text == nfc:
        return "NFC"
    if text == nfd:
        return "NFD"
    return "both"


def strip_diacritics(text: str) -> str:
    """Remove all Vietnamese tone and vowel marks: `không tốt` -> `khong tot`.

    Powers the `nodiacritic` perturbation suite and the normalized near-duplicate key. Note the
    ambiguity this creates is precisely the point: `ma` maps back onto {mà, má, mã, mả, mạ}.
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", without_marks).translate(_D_STROKE)


def collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def collapse_elongation(text: str, keep: int = 2) -> str:
    """`hayyyy` -> `hayy`. Character repetition carries *intensity* in Vietnamese user text, so it
    is capped rather than removed — collapsing to a single character discards that signal."""
    return _ELONGATION_RE.sub(lambda m: m.group(1) * keep, text)


def basic_clean(text: str) -> str:
    """The `norm` variant: NFC + whitespace collapse. Non-destructive; applied everywhere."""
    return collapse_whitespace(to_nfc(text))


def dedup_key(text: str) -> str:
    """Aggressive normalization used *only* for duplicate detection, never for training input.

    Lowercase + diacritics stripped + punctuation removed + whitespace collapsed. Two sentences
    sharing a key are near-duplicates for leakage purposes (docs/DATA_CARD.md § 7).
    """
    t = strip_diacritics(to_nfc(text)).lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return collapse_whitespace(t)


def syllable_count(text: str) -> int:
    """Vietnamese is written with whitespace-delimited syllables, so a whitespace token count is a
    syllable count, not a word count — the distinction that makes word segmentation a task at all."""
    return len(basic_clean(text).split())
