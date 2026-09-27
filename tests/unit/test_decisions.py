"""The Cycle 1 decision rules, checked on synthetic runs with known answers.

The rules in configs/experiments/cycle1.yaml were declared before any result existed. These tests
pin that the code applies exactly those rules: pairing by seed against the same-session CE control,
the advance threshold, and the augmentation criterion's two conditions.
"""

from __future__ import annotations

import json

import pytest
import yaml

from vifeedback.evaluation import decisions as D


def _pc(f1: float, recall: float = 0.55, precision: float = 0.8) -> dict:
    c = {"precision": 0.96, "recall": 0.96, "f1": 0.96}
    return {
        "negative": c,
        "neutral": {"precision": precision, "recall": recall, "f1": f1},
        "positive": c,
    }


def _rob(clean: float, deg: float) -> dict:
    suites = ("nodiacritic", "nodiacritic-50", "teencode-100", "charnoise-5")
    return {s: {"macro_f1": clean - deg} for s in suites}


def _write(root, recipe: str, seed: int, metrics: dict) -> None:
    d = root / f"p8-sent-phobert-base-seg_pyvi-{recipe}-s{seed}-deadbeef-val"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(
        yaml.safe_dump({"recipe": recipe, "seed": seed}), encoding="utf-8"
    )
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


@pytest.fixture
def runs(tmp_path):
    ce = {42: 0.8634, 1337: 0.8709, 2024: 0.8494}
    for s, v in ce.items():
        _write(
            tmp_path,
            "crt",
            s,
            {
                "macro_f1": v + 0.010,  # cRT helps by +0.010 on every seed
                "per_class": _pc(0.70),
                "stage1_ce": {"macro_f1": v, "per_class": _pc(0.66), "robustness": _rob(v, 0.20)},
            },
        )
        _write(
            tmp_path,
            "logit-adjust",
            s,
            {"macro_f1": v + (0.004 if s == 42 else -0.002), "per_class": _pc(0.67)},
        )
        _write(
            tmp_path,
            "aug-diac-teen",
            s,
            {
                "macro_f1": v - 0.002,  # small clean cost
                "per_class": _pc(0.66),
                "robustness": _rob(v - 0.002, 0.10),  # halves the degradation
                "augmentation": {"changed_share": 0.27},
            },
        )
    return D.load_runs(8, tmp_path), ce


class TestH1:
    def test_pairs_against_the_same_session_control(self, runs) -> None:
        r, ce = runs
        out = D.h1(r, registry_ce=ce)
        crt = out["methods"]["crt"]
        assert crt["n"] == 3 and crt["wins"] == 3
        assert crt["mean_delta"] == pytest.approx(0.010)
        assert crt["advance_to_5_seeds"] is True

    def test_a_small_mixed_effect_does_not_advance(self, runs) -> None:
        r, ce = runs
        la = D.h1(r, registry_ce=ce)["methods"]["logit-adjust"]
        assert la["wins"] == 1
        assert la["advance_to_5_seeds"] is False

    def test_stage1_reproduction_is_reported(self, runs) -> None:
        r, ce = runs
        rep = D.h1(r, registry_ce=ce)["reproduces_registry"]
        assert rep[42] == {"stage1": 0.8634, "registry": 0.8634}


class TestH2:
    def test_halved_degradation_with_small_clean_cost_is_supported(self, runs) -> None:
        r, _ = runs
        out = D.h2(r)
        assert out["suites"]["nodiacritic-50"]["relative_reduction"] == pytest.approx(0.5)
        assert out["clean"]["mean_delta"] == pytest.approx(-0.002)
        assert out["supported_at_3_seeds"] is True

    def test_clean_cost_over_budget_is_not_supported(self, tmp_path) -> None:
        _write(
            tmp_path,
            "crt",
            42,
            {
                "macro_f1": 0.87,
                "per_class": _pc(0.7),
                "stage1_ce": {
                    "macro_f1": 0.86,
                    "per_class": _pc(0.66),
                    "robustness": _rob(0.86, 0.2),
                },
            },
        )
        _write(
            tmp_path,
            "aug-diac-teen",
            42,
            {"macro_f1": 0.85, "per_class": _pc(0.6), "robustness": _rob(0.85, 0.05)},
        )
        out = D.h2(D.load_runs(8, tmp_path))
        assert out["suites"]["nodiacritic-50"]["relative_reduction"] > 0.2
        assert out["supported_at_3_seeds"] is False  # loses 0.010 clean > 0.005 budget

    def test_charnoise_is_marked_out_of_family(self, runs) -> None:
        r, _ = runs
        suites = D.h2(r)["suites"]
        assert suites["charnoise-5"]["in_family"] is False
        assert suites["nodiacritic-50"]["in_family"] is True
