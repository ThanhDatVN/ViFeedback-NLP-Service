"""External evaluation corpora: hash checks, label mapping and the invariance statistics."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from vifeedback.evaluation import external as X


@pytest.fixture
def fake_ext(tmp_path, monkeypatch):
    """A reference file and a data folder under tmp_path, with one NEU-ESC-like test file."""
    ext, ref = tmp_path / "external", tmp_path / "ref.json"
    (ext / "neu_esc").mkdir(parents=True)
    csv = (
        b"Text,Sentiment,Classification\n"
        b"a,Neutral,Academic\nb,Positive,Service\nc,Negative,Other\nd,Toxic,Spam\n"
    )
    (ext / "neu_esc" / "test_set.csv").write_bytes(csv)
    spec = {
        "neu_esc": {
            "source": "huggingface",
            "repo": "x/y",
            "revision": "r",
            "files": {"test": {"path": "test_set.csv", "sha256": hashlib.sha256(csv).hexdigest()}},
        }
    }
    ref.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(X, "EXT", ext)
    monkeypatch.setattr(X, "REF", ref)
    return ext


def test_neu_esc_labels_map_onto_uit_vsfc(fake_ext):
    df = X.load_neu_esc("test")
    assert df.sentiment.tolist() == ["neutral", "positive", "negative", "negative"]
    assert df.source_label.tolist()[-1] == "Toxic"  # kept, so results can exclude it
    assert set(X.NEU_ESC_COURSE_TOPICS) <= set(df.topic)


def test_a_changed_file_is_refused(fake_ext):
    (fake_ext / "neu_esc" / "test_set.csv").write_text(
        "Text,Sentiment,Classification\nz,Neutral,Academic\n"
    )
    with pytest.raises(ValueError, match="does not match"):
        X.load_neu_esc("test")


def test_committed_reference_pins_revisions_and_hashes():
    ref = X.reference()
    assert len(ref["vilexnorm"]["revision"]) == 40
    assert all(len(f["sha256"]) == 64 for f in ref["vilexnorm"]["files"].values())
    assert len(ref["neu_esc"]["revision"]) == 40


def test_flip_rate_and_paired_test():
    a, b = np.array([0, 1, 2, 2]), np.array([0, 2, 2, 1])
    r = X.flip_rate(a, b)
    assert (r["flips"], r["n"], r["rate"]) == (2, 4, 0.5)
    lo, hi = r["wilson_95"]
    assert lo < 0.5 < hi
    t = X.paired_flip_test(np.array([1, 1, 1, 0], bool), np.array([0, 0, 0, 0], bool))
    assert (t["only_first_flips"], t["only_second_flips"]) == (3, 0)
    assert t["exact_mcnemar_p"] == pytest.approx(0.25)


def test_sentence_case_capitalizes_only_the_first_letter():
    v = X.case_variants(["thầy dạy hay", ""])
    assert v["sentence_case"] == ["Thầy dạy hay", ""]
    assert v["as_is"] == ["thầy dạy hay", ""]


def test_neu_esc_numeric_codes_follow_the_dataset_card(tmp_path, monkeypatch):
    ext, ref = tmp_path / "external", tmp_path / "ref.json"
    (ext / "neu_esc").mkdir(parents=True)
    csv = b"text,sentiment,classification\na,0.0,2.0\nb,1.0,4.0\nc,2.0,0.0\nd,3.0,3.0\n"
    (ext / "neu_esc" / "test_set.csv").write_bytes(csv)
    ref.write_text(
        json.dumps(
            {
                "neu_esc": {
                    "files": {
                        "test": {"path": "test_set.csv", "sha256": hashlib.sha256(csv).hexdigest()}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(X, "EXT", ext)
    monkeypatch.setattr(X, "REF", ref)
    df = X.load_neu_esc("test")
    assert df.source_label.tolist() == ["Neutral", "Positive", "Negative", "Toxic"]
    assert df.topic.tolist() == ["Academic", "Service", "Spam", "Other"]
    assert df.sentiment.tolist() == ["neutral", "positive", "negative", "negative"]
