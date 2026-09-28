"""PhobertBPE against transformers' PhobertTokenizer built from the same small files (NEXT_PLAN v5 F1).

The full-corpus check is scripts/check_phobert_tokenizer.py (every UIT-VSFC and NEU-ESC text); this
one needs no model download, so it runs in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from vifeedback.inference.phobert_tokenizer import PhobertBPE

VOCAB = ["giảng_viên", "nhiệt_tình", "th@@", "ầy", "d@@", "ạy", "hay", "ng", "a@@", "b", "c", "rất"]
MERGES = [
    ("t", "h"),
    ("th", "ầ"),
    ("thầ", "y</w>"),
    ("n", "g</w>"),
    ("r", "ấ"),
    ("rấ", "t</w>"),
    ("h", "a"),
    ("ha", "y</w>"),
]
TEXTS = [
    "giảng_viên nhiệt_tình",
    "thầy dạy rất hay",
    "",
    "a <mask> b </s>c",
    "không_có trong từ_điển",
    "dòng\nhai  khoảng   trắng",
    "rất " * 70,
]


@pytest.fixture
def artifact(tmp_path):
    (tmp_path / "vocab.txt").write_text("".join(f"{w} 1\n" for w in VOCAB), encoding="utf-8")
    (tmp_path / "bpe.codes").write_text(
        "".join(f"{a} {b} 5\n" for a, b in MERGES), encoding="utf-8"
    )
    (tmp_path / "added_tokens.json").write_text(f'{{"<mask>": {len(VOCAB) + 4}}}', encoding="utf-8")
    return tmp_path


def test_matches_phobert_tokenizer_id_for_id(artifact):
    transformers = pytest.importorskip("transformers")
    hf = transformers.PhobertTokenizer(str(artifact / "vocab.txt"), str(artifact / "bpe.codes"))
    ours = PhobertBPE(artifact)
    for max_length in (96, 8):
        a = hf(TEXTS, return_tensors="np", padding=True, truncation=True, max_length=max_length)
        b = ours(TEXTS, return_tensors="np", padding=True, truncation=True, max_length=max_length)
        assert np.array_equal(a["input_ids"], b["input_ids"])
        assert np.array_equal(a["attention_mask"], b["attention_mask"])


def test_special_ids_padding_and_truncation(artifact):
    enc = PhobertBPE(artifact)(["rất hay", "rất " * 20], max_length=6)
    assert enc["input_ids"].shape == (2, 6)
    assert (
        list(enc["input_ids"][0][:2]) == [0, VOCAB.index("rất") + 4]
        and enc["input_ids"][0][-1] == 1
    )
    assert (
        enc["input_ids"][1][0] == 0 and enc["input_ids"][1][-1] == 2
    )  # cut to 4 pieces + <s> </s>
    assert list(enc["attention_mask"][0]) == [1, 1, 1, 1, 0, 0]


def test_is_phobert_dir(artifact, tmp_path_factory):
    assert PhobertBPE.is_phobert_dir(artifact)
    assert not PhobertBPE.is_phobert_dir(tmp_path_factory.mktemp("empty"))
