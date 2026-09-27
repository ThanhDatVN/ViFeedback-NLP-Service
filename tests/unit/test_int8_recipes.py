"""Cycle 3 S5: which nodes each INT8 recipe keeps in FP32, how one is chosen, and the quality rule."""

from __future__ import annotations

import numpy as np

from vifeedback.inference import int8_recipes as Q

NAMES = [
    "/roberta/encoder/layer.0/attention/self/query/MatMul",
    "/roberta/encoder/layer.0/attention/output/dense/MatMul",
    "/roberta/encoder/layer.0/output/dense/MatMul",
    "/roberta/encoder/layer.11/intermediate/dense/MatMul",
    "/roberta/encoder/layer.11/output/dense/MatMul",
    "/classifier/dense/Gemm",
    "/classifier/out_proj/Gemm",
]


def test_recipes_exclude_the_intended_nodes():
    assert Q.RECIPES["pc"](NAMES) == []
    assert Q.RECIPES["pc-head"](NAMES) == NAMES[-2:]
    assert set(Q.RECIPES["pc-head-last2"](NAMES)) == set(NAMES[3:])
    ffn = Q.RECIPES["pc-head-ffnout"](NAMES)
    assert (
        NAMES[2] in ffn and NAMES[4] in ffn and NAMES[1] not in ffn
    )  # attention output stays INT8


def test_fidelity_counts_neutral_agreement():
    fp32 = np.array([[0, 5, 0], [0, 5, 0], [5, 0, 0], [0, 0, 5]], float)
    int8 = np.array([[0, 5, 0], [5, 0, 0], [5, 0, 0], [0, 0, 5]], float)
    f = Q.fidelity(fp32, int8)
    assert (
        f["label_agreement"] == 0.75
        and f["neutral_agreement"] == 0.5
        and f["fp32_neutral_rows"] == 2
    )


def test_select_prefers_neutral_fidelity_within_the_size_limit():
    c = {
        "big": {
            "size_mb": 300,
            "neutral_agreement": 1.0,
            "label_agreement": 1.0,
            "mean_abs_logit_diff": 0,
        },
        "a": {
            "size_mb": 150,
            "neutral_agreement": 0.90,
            "label_agreement": 0.99,
            "mean_abs_logit_diff": 0.1,
        },
        "b": {
            "size_mb": 150,
            "neutral_agreement": 0.95,
            "label_agreement": 0.98,
            "mean_abs_logit_diff": 0.2,
        },
    }
    assert Q.select(c) == "b"
    assert Q.select({"big": c["big"]}) is None


def test_acceptance_fails_on_a_neutral_loss():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    fp32 = y.copy()
    int8 = y.copy()
    int8[np.flatnonzero(y == 1)[:40]] = 0  # lose neutral recall
    r = Q.acceptance(y, fp32, int8)
    assert r["neutral_f1_loss"] > 0.02 and r["quality_passed"] is False
    assert Q.acceptance(y, fp32, fp32.copy())["quality_passed"] is True
