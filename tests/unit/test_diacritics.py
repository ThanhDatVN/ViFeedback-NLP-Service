"""S2b: restore diacritics from train-only counts, and leave accented input alone."""

from __future__ import annotations

from vifeedback.preprocess.diacritics import Restorer, accented_share

TRAIN = ["thầy dạy rất dễ hiểu"] * 5 + ["cô dạy rất hay"] * 5 + ["thay đổi lịch học"] * 2


def test_restores_the_likely_sequence():
    r = Restorer.fit(TRAIN, max_share_restored=0.0)
    assert r.restore("thay day rat de hieu") == "thầy dạy rất dễ hiểu"
    assert r.restore("co day rat hay") == "cô dạy rất hay"


def test_accented_tokens_and_unknown_keys_are_kept():
    r = Restorer.fit(TRAIN, max_share_restored=0.0)
    assert r.restore("thầy dạy xyz") == "thầy dạy xyz"


def test_only_unaccented_input_is_rewritten():
    r = Restorer.fit(TRAIN, max_share_restored=0.0)
    assert accented_share("thầy dạy rất dễ hiểu") > r.threshold
    assert r("thầy dạy rất dễ hiểu") == "thầy dạy rất dễ hiểu"
    assert r("thay day rat de hieu") != "thay day rat de hieu"


def test_save_and_load_round_trip(tmp_path):
    r = Restorer.fit(TRAIN, max_share_restored=0.0)
    r.save(tmp_path / "r.json")
    assert Restorer.load(tmp_path / "r.json").restore("co day rat hay") == "cô dạy rất hay"
