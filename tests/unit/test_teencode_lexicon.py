"""S2a: a teencode lexicon learned from (original, normalized) pairs, applied at empirical rates."""

from __future__ import annotations

import numpy as np

from vifeedback.preprocess import teencode_lexicon as T

PAIRS = [
    ("ko biết", "không biết"),
    ("ko hiểu j", "không hiểu gì"),
    ("mn ơi ko", "mọi người ơi không"),
    ("không sao", "không sao"),
    ("mn đi", "mọi người đi"),
    ("ntn z", "như thế nào vậy"),
] * 3


def test_learns_variants_rates_and_multi_word_forms():
    lex = T.build(PAIRS, min_count=3)
    assert set(lex["không"]["variants"]) == {"ko"}
    # 9 of 12 occurrences of "không" are written "ko"
    assert lex["không"]["rate"] == 9 / 12
    assert lex["mọi người"]["variants"] == {"mn": 6}
    assert "sao" not in lex  # never written differently


def test_rare_variants_are_dropped():
    lex = T.build(PAIRS[:6], min_count=4)  # "ko" is seen 3 times in these six pairs
    assert lex == {}


def test_perturb_is_seeded_and_uses_longest_match():
    lex = {
        "mọi người": {"rate": 1.0, "variants": {"mn": 1}},
        "người": {"rate": 1.0, "variants": {"ng": 1}},
    }
    assert T.perturb("mọi người đi học", np.random.default_rng(0), lex) == "mn đi học"
    lex2 = {"không": {"rate": 0.5, "variants": {"ko": 3, "k": 1}}}
    a = T.perturb("không không không không", np.random.default_rng(7), lex2)
    b = T.perturb("không không không không", np.random.default_rng(7), lex2)
    assert a == b


def test_augment_recipe_uses_the_lexicon(tmp_path, monkeypatch):
    import json

    from vifeedback.training import augment as A

    f = tmp_path / "lex.json"
    f.write_text(json.dumps({"không": {"rate": 1.0, "variants": {"ko": 1}}}), encoding="utf-8")
    monkeypatch.setattr(T, "LEXICON", f)
    monkeypatch.setattr(T, "_CACHE", {})
    monkeypatch.setattr(T.load, "__defaults__", (f,))
    texts = ["thầy không đến lớp"] * 200
    out, changed = A.augment(texts, "diac-teen-vln", p=1.0, seed=1)
    assert changed.mean() > 0.85  # nodiacritic-50 can leave a 4-syllable sentence unchanged (1/16)
    assert any("ko" in t.split() for t in out)  # the lexicon half
    assert (
        "diac-teen-vln" in A.RECIPES and "teencode-vln" not in A.SUITES
    )  # never an evaluation suite
