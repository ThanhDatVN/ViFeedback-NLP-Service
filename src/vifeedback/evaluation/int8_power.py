"""E1 (NEXT_PLAN v5): can a larger acceptance set demonstrate INT8 non-inferiority at 0.005?

The careful INT8 recipe (pc-head-last2, 178.5 MB) lost only 0.0004 macro-F1 on UIT-VSFC validation,
but the one-sided 95% upper bound was 0.0095: 1,583 sentences, 73 of them neutral, cannot show a
0.005 margin. Before any new rule is declared, this estimates the bound a bigger labelled set
(UIT-VSFC validation + NEU-ESC validation) would give, without looking at INT8 on that set:

1. INT8's disagreements with FP32 are measured on held-out data that is NOT in the acceptance set
   (the INT8 study's stratified train subset, and a sample of NEU-ESC train), as the empirical
   distribution P(INT8 label | source, gold label, FP32 label).
2. On the acceptance set, where FP32's predictions are known, INT8 predictions are simulated from
   that distribution, and the paired-bootstrap one-sided 95% upper bound of the macro-F1 drop is
   computed; repeated over many simulations.

The share of simulations whose bound is below 0.005 is the power of a declared S5' test.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vifeedback import paths

OUT = paths.RESULTS / "studies" / "int8_power"
INT8 = paths.MODELS / "int8_candidates" / "pc-head-last2" / "model.int8.onnx"
FP32 = paths.MODELS / "int8_candidates" / "fp32_plain" / "model.onnx"
K = 3


def _predict(model_file, texts: list[str]) -> np.ndarray:
    from vifeedback.inference.onnx_export import OnnxClassifier

    # The FP32 graph's directory has no tokenizer; the served release's is the same (p9).
    clf = OnnxClassifier(model_file.parent, model_file=model_file.name)
    return np.concatenate([clf.predict(texts[i : i + 64])[0] for i in range(0, len(texts), 64)])


def _macro_f1_batch(y: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Macro-F1 for each row of bootstrap indices `idx` (draws x n), vectorized."""
    yy, pp = y[idx], pred[idx]
    f1s = []
    for c in range(K):
        tp = ((yy == c) & (pp == c)).sum(axis=1)
        fp = ((yy != c) & (pp == c)).sum(axis=1)
        fn = ((yy == c) & (pp != c)).sum(axis=1)
        denom = 2 * tp + fp + fn
        f1s.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(f1s, axis=0)


def upper_bound(
    y: np.ndarray, fp32: np.ndarray, int8: np.ndarray, draws: int, rng
) -> dict[str, float]:
    """Paired bootstrap of macro-F1(FP32) - macro-F1(INT8); one-sided 95% upper bound."""
    n = len(y)
    diffs = []
    for s in range(0, draws, 200):
        idx = rng.integers(0, n, (min(200, draws - s), n))
        diffs.append(_macro_f1_batch(y, fp32, idx) - _macro_f1_batch(y, int8, idx))
    d = np.concatenate(diffs)
    full = np.arange(n)[None, :]
    observed = float(_macro_f1_batch(y, fp32, full)[0] - _macro_f1_batch(y, int8, full)[0])
    return {"observed_drop": observed, "upper_95_one_sided": float(np.quantile(d, 0.95))}


def conditional(src: np.ndarray, y: np.ndarray, fp32: np.ndarray, int8: np.ndarray) -> dict:
    """P(INT8 label | source, gold, FP32 label), with the counts behind it."""
    table: dict = {}
    for s in np.unique(src):
        for g in range(K):
            for p in range(K):
                m = (src == s) & (y == g) & (fp32 == p)
                if m.any():
                    counts = np.bincount(int8[m], minlength=K)
                    table[(str(s), g, p)] = counts
    return table


def simulate(src, y, fp32, table, rng) -> np.ndarray:
    """INT8 labels drawn per example from the conditional table (FP32's label where unseen)."""
    out = fp32.copy()
    for i in range(len(y)):
        counts = table.get((str(src[i]), int(y[i]), int(fp32[i])))
        if counts is not None and counts.sum() > 0:
            out[i] = rng.choice(K, p=counts / counts.sum())
    return out


def run(
    simulations: int = 40, draws: int = 2000, neu_sample: int = 3000, seed: int = 0
) -> dict[str, Any]:
    from vifeedback.evaluation import llm_reference as L
    from vifeedback.preprocess.variants import load_variant
    from vifeedback.training.domain import neu_esc, serving_transform

    rng = np.random.default_rng(seed)
    transform = serving_transform()
    study = json.loads(
        (paths.RESULTS / "studies" / "int8_recipes" / "summary.json").read_text(encoding="utf-8")
    )
    fs = study["fidelity_subset"]

    # 1. Held-out estimation data (never the acceptance set).
    tr = load_variant("seg_pyvi", "train")
    sub = L.prompt_dev_subset(
        tr.sentiment.to_numpy(), n=fs["n"], per_class_min=fs["per_class_min"], seed=fs["seed"]
    )
    uit_x, uit_y = tr.sentence.to_numpy()[sub].tolist(), tr.sentiment.to_numpy()[sub]
    neu_tr = neu_esc("train", transform, in_scope=False)
    pick = rng.choice(len(neu_tr["y"]), size=min(neu_sample, len(neu_tr["y"])), replace=False)
    neu_x, neu_y = [neu_tr["x"][i] for i in pick], neu_tr["y"][pick]
    est_x = uit_x + neu_x
    est_y = np.r_[uit_y, neu_y]
    est_src = np.array(["uit"] * len(uit_x) + ["neu"] * len(neu_x))
    est_fp32, est_int8 = _predict(FP32, est_x), _predict(INT8, est_x)
    table = conditional(est_src, est_y, est_fp32, est_int8)

    # 2. The acceptance set: only FP32 is run on it.
    uit_dv = load_variant("seg_pyvi", "validation")
    neu_dv = neu_esc("validation", transform, in_scope=False)
    acc_x = uit_dv.sentence.tolist() + neu_dv["x"]
    acc_y = np.r_[uit_dv.sentiment.to_numpy(), neu_dv["y"]]
    acc_src = np.array(["uit"] * len(uit_dv) + ["neu"] * len(neu_dv["y"]))
    acc_fp32 = _predict(FP32, acc_x)

    def bounds(mask: np.ndarray) -> dict[str, Any]:
        bound_list: list[float] = []
        for _ in range(simulations):
            sim = simulate(acc_src[mask], acc_y[mask], acc_fp32[mask], table, rng)
            bound_list.append(
                upper_bound(acc_y[mask], acc_fp32[mask], sim, draws, rng)["upper_95_one_sided"]
            )
        ub = np.array(bound_list)
        return {
            "n": int(mask.sum()),
            "upper_bound_median": float(np.median(ub)),
            "upper_bound_p10_p90": [float(np.quantile(ub, 0.1)), float(np.quantile(ub, 0.9))],
            "power_below_0_005": float((ub < 0.005).mean()),
            "bounds": ub.tolist(),
        }

    uit_mask = acc_src == "uit"
    out = {
        "question": "NEXT_PLAN v5 E1: would UIT-VSFC + NEU-ESC validation demonstrate INT8 non-inferiority at 0.005?",
        "recipe": study["chosen"],
        "estimation": {
            "uit_train_subset": len(uit_x),
            "neu_train_sample": len(neu_x),
            "disagreement_uit": float(
                (est_fp32[est_src == "uit"] != est_int8[est_src == "uit"]).mean()
            ),
            "disagreement_neu": float(
                (est_fp32[est_src == "neu"] != est_int8[est_src == "neu"]).mean()
            ),
        },
        "acceptance_sets": {
            "uit_validation_only": bounds(uit_mask),
            "uit_plus_neu_validation": bounds(np.ones(len(acc_y), dtype=bool)),
        },
        "simulations": simulations,
        "bootstrap_draws": draws,
        "reference": {
            "uit_validation_observed_upper_bound": study["validation"][
                "macro_f1_drop_upper_95_one_sided"
            ]
        },
    }
    # Calibration check: the simulation must put the bound actually observed on UIT-VSFC validation
    # (INT8 really run there) inside the bulk of its simulated bounds for that same set; otherwise
    # the simulated power is not evidence of anything.
    sims = np.array(out["acceptance_sets"]["uit_validation_only"]["bounds"])
    observed = out["reference"]["uit_validation_observed_upper_bound"]
    q = float((sims <= observed).mean())
    out["calibration"] = {
        "observed_uit_bound_quantile_among_simulations": q,
        "calibrated": 0.1 <= q <= 0.9,
        "reading": (
            "calibrated: the simulated power can be used"
            if 0.1 <= q <= 0.9
            else "NOT calibrated: the simulated bounds are off, so the power is not evidence"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    from vifeedback.evaluation.report import yaml_safe

    (OUT / "summary.json").write_text(json.dumps(yaml_safe(out), indent=2), encoding="utf-8")
    return out


def accept() -> dict[str, Any]:
    """S5' (cycle4.yaml v4): the declared rule on UIT-VSFC validation + NEU-ESC validation."""
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.inference import int8_recipes as Q
    from vifeedback.preprocess.variants import load_variant
    from vifeedback.training.domain import neu_esc, serving_transform

    uit = load_variant("seg_pyvi", "validation")
    neu = neu_esc("validation", serving_transform(), in_scope=False)
    x = uit.sentence.tolist() + neu["x"]
    y = np.r_[uit.sentiment.to_numpy(), neu["y"]]
    is_uit = np.r_[np.ones(len(uit), dtype=bool), np.zeros(len(neu["y"]), dtype=bool)]
    fp32, int8 = _predict(FP32, x), _predict(INT8, x)

    pooled = Q.acceptance(y, fp32, int8)
    uit_part = Q.acceptance(y[is_uit], fp32[is_uit], int8[is_uit])
    neu_drop = M.macro_f1(y[~is_uit], fp32[~is_uit], K) - M.macro_f1(y[~is_uit], int8[~is_uit], K)
    size_mb = INT8.stat().st_size / 1e6
    rules = {
        "1_pooled_upper_bound": {
            "value": pooled["macro_f1_drop_upper_95_one_sided"],
            "limit": "<= 0.005",
        },
        "2_neu_esc_drop": {"value": float(neu_drop), "limit": "<= 0.005"},
        "3_uit_neutral_f1_loss": {"value": uit_part["neutral_f1_loss"], "limit": "<= 0.02"},
        "4_size_mb": {"value": size_mb, "limit": "<= 200"},
    }
    rules["1_pooled_upper_bound"]["passed"] = rules["1_pooled_upper_bound"]["value"] <= 0.005
    rules["2_neu_esc_drop"]["passed"] = rules["2_neu_esc_drop"]["value"] <= 0.005
    rules["3_uit_neutral_f1_loss"]["passed"] = rules["3_uit_neutral_f1_loss"]["value"] <= 0.02
    rules["4_size_mb"]["passed"] = rules["4_size_mb"]["value"] <= 200

    def part(mask: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(mask.sum()),
            "label_agreement": float((fp32[mask] == int8[mask]).mean()),
            "macro_f1_fp32": M.macro_f1(y[mask], fp32[mask], K),
            "macro_f1_int8": M.macro_f1(y[mask], int8[mask], K),
        }

    out = {
        "declared_in": "configs/experiments/cycle4.yaml v4 S5prime_int8 (ADR-035)",
        "graph": str(INT8.relative_to(paths.ROOT)),
        "rules": rules,
        "passed": all(bool(r["passed"]) for r in rules.values()),
        "pooled": pooled,
        "uit_validation": {**part(is_uit), "neutral_f1_loss": uit_part["neutral_f1_loss"]},
        "neu_esc_validation": part(~is_uit),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s5prime.json").write_text(json.dumps(yaml_safe(out), indent=2), encoding="utf-8")
    return out
