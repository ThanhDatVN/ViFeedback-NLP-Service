"""Cycle 5, H10b: consistency anchored on a frozen teacher (configs/experiments/cycle5.yaml v4).

H10's own-prediction consistency collapsed informal text onto one class (ADR-038). Here each real
comment is trained towards what a frozen teacher, the 5-seed ensemble of the served recipe, says
about its human normalization. The target is fixed and diverse, so no constant prediction meets it.

* ``anchored_orig``: KL(q(normalized) || p(original)).
* ``anchored_both``: the mean of that and KL(q(normalized) || p(normalized)).

Metrics on pairs: agreement (the model's label on the original equals the teacher's label on the
normalized form), the flip rate (as H10), and label_tv (total-variation distance between the model's
label distribution on the originals and the teacher's on the normalized forms).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vifeedback import paths
from vifeedback.constants import SEEDS, label_names
from vifeedback.training import consistency as C
from vifeedback.training import domain as D

OUT = paths.RESULTS / "studies" / "cycle5" / "h10b"
CONFIRM_INDEX = paths.RESULTS / "studies" / "cycle5" / "h10b_confirm_index.csv"
N_TRAIN_PAIRS = 7535
N_CONFIRM = 1500
LABELS = label_names("sentiment")


# --- data ---------------------------------------------------------------------------------------


def confirm_split(n: int = N_TRAIN_PAIRS) -> tuple[np.ndarray, np.ndarray]:
    """cycle5.yaml v4: positions within H10's 7,535 training pairs, by default_rng(43); the first
    1,500 confirm, the other 6,035 train."""
    if n != N_TRAIN_PAIRS:
        raise ValueError(f"expected {N_TRAIN_PAIRS} training pairs, found {n}")
    perm = np.random.default_rng(43).permutation(n)
    return perm[:N_CONFIRM], perm[N_CONFIRM:]


def write_confirm_index() -> Path:
    """The confirmation pairs as row indices into ViLexNorm train (no text)."""
    import pandas as pd

    confirm, _ = confirm_split()
    rows = C.split_indices()[1][confirm]
    CONFIRM_INDEX.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": np.sort(rows)}).to_csv(CONFIRM_INDEX, index=False)
    return CONFIRM_INDEX


def pairs(split: str, transform) -> dict[str, Any]:
    """ViLexNorm pairs through the serving transform: train (6,035), confirm (1,500) or dev (837)."""
    if split == "dev":
        return C.vilexnorm_pairs("dev", transform)
    base = C.vilexnorm_pairs("train_pairs", transform)
    confirm, train = confirm_split(len(base["orig"]))
    idx = confirm if split == "confirm" else train
    return {
        "orig": [base["orig"][i] for i in idx],
        "norm": [base["norm"][i] for i in idx],
        "changed": base["changed"][idx],
    }


def teacher_q(texts: list[str]) -> np.ndarray:
    """The frozen teacher: mean softmax (T = 1) of the served recipe's five seeds, cached by the
    SHA-1 of each text (models/distill, no text stored)."""
    from vifeedback.training import distill as K

    cache = K.SoftLabelCache(paths.MODELS / "distill" / "soft_labels_served_T1.npz")
    todo = cache.missing(texts)
    if todo:
        acc = np.zeros((len(todo), len(LABELS)))
        for s in SEEDS:
            model, tok = K._load(D.control_checkpoint(s))
            z = K.teacher_logits(model, tok, todo)
            e = np.exp(z - z.max(axis=1, keepdims=True))
            acc += e / e.sum(axis=1, keepdims=True)
        cache.add(todo, acc / len(SEEDS))
        cache.save()
    return cache.get(texts)


# --- metrics ------------------------------------------------------------------------------------


def label_tv(pred: np.ndarray, target: np.ndarray, k: int = len(LABELS)) -> float:
    a = np.bincount(np.asarray(pred), minlength=k) / max(1, len(pred))
    b = np.bincount(np.asarray(target), minlength=k) / max(1, len(target))
    return float(0.5 * np.abs(a - b).sum())


def pair_metrics(po: np.ndarray, pn: np.ndarray, teacher: np.ndarray) -> dict[str, Any]:
    return {
        "pairs": len(po),
        "agreement": float(np.mean(po == teacher)),
        "flip_rate": float(np.mean(po != pn)),
        "label_tv": label_tv(po, teacher),
        "label_share_original": {c: float(np.mean(po == i)) for i, c in enumerate(LABELS)},
        "teacher_label_share_normalized": {
            c: float(np.mean(teacher == i)) for i, c in enumerate(LABELS)
        },
    }


def score_pairs(model, tok, pr: dict[str, Any], q: np.ndarray) -> dict[str, Any]:
    po = C.predict_labels(model, tok, pr["orig"])
    pn = C.predict_labels(model, tok, pr["norm"])
    return {**pair_metrics(po, pn, q.argmax(1)), "pred_orig": po, "pred_norm": pn}


# --- orchestration ------------------------------------------------------------------------------


def _write(name: str, summary: dict[str, Any], scored: dict[str, Any], dev: dict[str, Any]) -> Path:
    import pandas as pd

    from vifeedback.evaluation.report import yaml_safe

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    body = {
        **summary,
        "sets": {
            s: {k: v for k, v in r.items() if k not in ("pred", "prob")} for s, r in scored.items()
        },
        "vilexnorm_dev": {k: v for k, v in dev.items() if not k.startswith("pred_")},
    }
    (d / "summary.json").write_text(
        json.dumps(yaml_safe(body), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for s, r in scored.items():  # labels and probabilities only; no text
        table = pd.DataFrame(r["prob"].round(5), columns=[f"p_{c}" for c in LABELS])
        table.insert(0, "pred", [LABELS[i] for i in r["pred"]])
        table.to_csv(d / f"predictions_{s}.csv", index_label="row")
    return d


def run(recipe: str, seed: int, verbose: bool = True, smoke: bool = False) -> dict[str, Any]:
    """Train one H10b recipe at one seed; score it on the development sets (never a test split)."""
    import dataclasses

    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation import report as R
    from vifeedback.training.runner import _augment_train, _splits

    if recipe not in C.ANCHORED:
        raise ValueError(f"recipe must be one of {C.ANCHORED}")
    cfg = C.config(recipe, seed)
    transform = D.serving_transform()
    (x_tr, y_tr), uit_dv, _ = _splits("sentiment", "seg_pyvi")
    x_tr, augmentation = _augment_train(cfg, x_tr)
    tr, dv = pairs("train", transform), pairs("dev", transform)
    sets = C.evaluation_sets(transform)
    if smoke:
        cfg = dataclasses.replace(cfg, epochs=1)
        x_tr, y_tr = list(x_tr)[:256], np.asarray(y_tr)[:256]
        tr = {k: v[:256] for k, v in tr.items()}
        dv = {k: v[:128] for k, v in dv.items()}
        uit_dv = (list(uit_dv[0])[:128], np.asarray(uit_dv[1])[:128])
        sets = {k: (list(x)[:128], np.asarray(y)[:128]) for k, (x, y) in sets.items()}
    tr["q"] = teacher_q(tr["norm"])
    q_dev = teacher_q(dv["norm"])
    if verbose:
        print(f"[{cfg.run_id()}] H10b {recipe} seed {seed}: {len(tr['orig'])} training pairs")
    out = C.train_consistency(cfg, recipe, (list(x_tr), np.asarray(y_tr)), tr, uit_dv, dv, verbose)
    model, tok = out["model"], out["tokenizer"]
    scored = D.score_sets(model, tok, sets)
    dev = score_pairs(model, tok, dv, q_dev)
    summary = {
        "recipe": recipe,
        "seed": seed,
        "history": out["history"],
        "best_epoch": out["best_epoch"],
    }
    if smoke:
        return {"summary": {**summary, "smoke": True}, "scored": scored, "dev": dev}

    ckpt = paths.MODELS / cfg.run_id("ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    uv = scored["uit_validation"]
    metrics = M.evaluate(
        np.asarray(sets["uit_validation"][1]), uv["pred"], "sentiment", y_prob=uv["prob"]
    )
    metrics.update(
        {"history": out["history"], "best_epoch": out["best_epoch"], "augmentation": augmentation}
    )
    R.save_run(
        cfg.run_id("validation"),
        metrics,
        config={
            **out["config"],
            "phase": f"P{C.PHASE_NUM_B}",
            "model": cfg.model_key,
            "split": "validation",
            "fit_seconds": out["train_seconds"],
            "notes": cfg.notes,
            "reason": "cycle5.yaml v4 H10b",
            "determinism": out["determinism"],
            "device": out["device"],
            "best_epoch": out["best_epoch"],
            "selection": out["selection"],
        },
        y_true=np.asarray(sets["uit_validation"][1]),
        y_pred=uv["pred"],
        y_prob=uv["prob"],
    )
    summary.update(
        {
            "run_id": cfg.run_id("validation"),
            "checkpoint": str(ckpt.relative_to(paths.ROOT)),
            "train_seconds": out["train_seconds"],
            "augmentation": augmentation,
        }
    )
    _write(f"{recipe}-s{seed}", summary, scored, dev)
    return {"summary": summary, "scored": scored, "dev": dev}


def evaluate_control(seed: int) -> dict[str, Any]:
    """The served recipe's checkpoint of one seed, scored with the candidates' code."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ck = D.control_checkpoint(seed)
    transform = D.serving_transform()
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForSequenceClassification.from_pretrained(ck)
    dv = pairs("dev", transform)
    scored = D.score_sets(model, tok, C.evaluation_sets(transform))
    dev = score_pairs(model, tok, dv, teacher_q(dv["norm"]))
    summary = {"recipe": "control", "seed": seed, "checkpoint": str(ck.relative_to(paths.ROOT))}
    _write(f"control-s{seed}", summary, scored, dev)
    return {"summary": summary, "scored": scored, "dev": dev}


def _summary(name: str) -> dict[str, Any]:
    f = OUT / name / "summary.json"
    if not f.exists():
        raise FileNotFoundError(f"{name} has not run: {f}")
    return json.loads(f.read_text(encoding="utf-8"))


def select() -> dict[str, Any]:
    """cycle5.yaml v4 H10b recipe_selection at seed 42."""
    from vifeedback.evaluation.report import yaml_safe

    ctrl = _summary("control-s42")
    c_uit = ctrl["sets"]["uit_validation"]["macro_f1"]
    c_agree = ctrl["vilexnorm_dev"]["agreement"]
    rows = {}
    for r in C.ANCHORED:
        s = _summary(f"{r}-s42")
        uit, dev = s["sets"]["uit_validation"]["macro_f1"], s["vilexnorm_dev"]
        rows[r] = {
            "uit_validation": uit,
            "uit_minus_control": uit - c_uit,
            "dev_agreement": dev["agreement"],
            "dev_flip_rate": dev["flip_rate"],
            "dev_label_tv": dev["label_tv"],
            "eligible": uit >= c_uit - 0.01 and dev["label_tv"] <= 0.10,
        }
    eligible = {r: v for r, v in rows.items() if v["eligible"]}
    chosen = max(eligible, key=lambda r: eligible[r]["dev_agreement"]) if eligible else None
    if chosen is None:
        outcome = "not supported: no recipe is eligible"
    elif rows[chosen]["dev_agreement"] <= c_agree:
        outcome, chosen = "not supported: no eligible recipe raises dev agreement", None
    else:
        outcome = "confirm"
    out = {
        "declared_in": "configs/experiments/cycle5.yaml v4 H10b recipe_selection",
        "control_s42": {
            "uit_validation": c_uit,
            "dev_agreement": c_agree,
            "dev_flip_rate": ctrl["vilexnorm_dev"]["flip_rate"],
            "dev_label_tv": ctrl["vilexnorm_dev"]["label_tv"],
        },
        "recipes": rows,
        "chosen": chosen,
        "outcome": outcome,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(yaml_safe(out), indent=2), encoding="utf-8")
    return out


def apply_rule(
    agree: dict[str, Any], flips: dict[str, Any], tv: float, diffs: dict[str, float]
) -> dict[str, Any]:
    """cycle5.yaml v4 H10b rule (1) to (7)."""
    limits = {
        "4_uit_validation_macro_f1": -0.005,
        "5_uit_validation_neutral_f1": -0.02,
        "6_uit_validation_stripped_macro_f1": -0.01,
        "7_neu_validation_macro_f1": -0.01,
    }
    rules: dict[str, dict[str, Any]] = {
        "1_agreement_up": {**agree, "passed": agree["ci_low"] > 0},
        "2_flip_rate_down": {**flips, "passed": flips["ci_high"] < 0},
        "3_label_tv": {"mean": tv, "limit": 0.10, "passed": tv <= 0.10},
    }
    for name, limit in limits.items():
        rules[name] = {"mean_diff": diffs[name], "limit": limit, "passed": diffs[name] >= limit}
    return {"rules": rules, "passed": all(r["passed"] for r in rules.values())}


def confirm(recipe: str) -> dict[str, Any]:
    """cycle5.yaml v4 H10b rule on the 1,500 confirmation pairs, five seeds each side."""
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.evaluation.report import yaml_safe

    seeds = list(SEEDS)
    cand = {s: paths.MODELS / C.config(recipe, s).run_id("ckpt") for s in seeds}
    ctrl = {s: D.control_checkpoint(s) for s in seeds}
    for s in seeds:
        _summary(f"{recipe}-s{s}")
        _summary(f"control-s{s}")
    transform = D.serving_transform()
    cf = pairs("confirm", transform)
    teacher = teacher_q(cf["norm"]).argmax(1)

    def preds(ck: Path) -> tuple[np.ndarray, np.ndarray]:
        tok = AutoTokenizer.from_pretrained(ck)
        model = AutoModelForSequenceClassification.from_pretrained(ck)
        return C.predict_labels(model, tok, cf["orig"]), C.predict_labels(model, tok, cf["norm"])

    pc = {s: preds(cand[s]) for s in seeds}
    pb = {s: preds(ctrl[s]) for s in seeds}
    # The paired bootstraps of H10's flip code work on any per-pair indicator.
    agree = C.flip_bootstrap(
        [pc[s][0] == teacher for s in seeds], [pb[s][0] == teacher for s in seeds]
    )
    agree["p_one_sided"] = None
    flips = C.flip_bootstrap(
        [pc[s][0] != pc[s][1] for s in seeds], [pb[s][0] != pb[s][1] for s in seeds]
    )
    tv = float(np.mean([label_tv(pc[s][0], teacher) for s in seeds]))

    def val(name: str, set_name: str, key: str) -> float:
        r = _summary(name)["sets"][set_name]
        return r["macro_f1"] if key == "macro_f1" else r["per_class"]["neutral"]["f1"]

    def mean_diff(set_name: str, key: str = "macro_f1") -> float:
        return float(
            np.mean(
                [
                    val(f"{recipe}-s{s}", set_name, key) - val(f"control-s{s}", set_name, key)
                    for s in seeds
                ]
            )
        )

    decision = apply_rule(
        agree,
        flips,
        tv,
        {
            "4_uit_validation_macro_f1": mean_diff("uit_validation"),
            "5_uit_validation_neutral_f1": mean_diff("uit_validation", "neutral_f1"),
            "6_uit_validation_stripped_macro_f1": mean_diff("uit_validation_stripped"),
            "7_neu_validation_macro_f1": mean_diff("neu_validation"),
        },
    )
    out = {
        "declared_in": "configs/experiments/cycle5.yaml v4 H10b rule",
        "recipe": recipe,
        "seeds": seeds,
        **decision,
        "reported": {
            "per_seed": {
                s: {
                    "candidate": pair_metrics(pc[s][0], pc[s][1], teacher),
                    "control": pair_metrics(pb[s][0], pb[s][1], teacher),
                }
                for s in seeds
            },
            "control_label_tv_mean": float(np.mean([label_tv(pb[s][0], teacher) for s in seeds])),
            "neu_validation_all_mean_diff": mean_diff("neu_validation_all"),
        },
        "confirmation_pairs": len(teacher),
    }
    d = OUT / f"confirm-{recipe}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "decision.json").write_text(json.dumps(yaml_safe(out), indent=2), encoding="utf-8")
    cols = {"teacher_normalized": [LABELS[i] for i in teacher]}
    for s in seeds:
        for side, p in (("candidate", pc[s]), ("control", pb[s])):
            cols[f"{side}-s{s}-orig"] = [LABELS[i] for i in p[0]]
            cols[f"{side}-s{s}-norm"] = [LABELS[i] for i in p[1]]
    pd.DataFrame(cols).to_csv(d / "predictions_confirm.csv", index_label="row")  # labels only
    return out


__all__ = [
    "CONFIRM_INDEX",
    "OUT",
    "apply_rule",
    "closing_gate",
    "confirm",
    "confirm_split",
    "evaluate_control",
    "label_tv",
    "pair_metrics",
    "pairs",
    "run",
    "select",
    "teacher_q",
    "write_confirm_index",
]


def closing_gate(recipe: str = "anchored_orig") -> dict[str, Any]:
    """cycle5.yaml v5 H10b_release.closing_gate: the five models on UIT-VSFC test, once, logged."""
    from vifeedback.evaluation import closing_gate as CG
    from vifeedback.evaluation import decisions as DEC
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation.report import yaml_safe

    dst = OUT / "closing_gate" / "summary.json"
    if dst.exists():
        raise FileExistsError(f"the H10b closing gate has run already: {dst}")
    reason = "Cycle 5 H10b closing gate (cycle5.yaml v5): the models on test, once"
    gi = CG.inputs()
    five: dict[int, float] = {}
    row42: dict[str, Any] = {}
    for s in SEEDS:
        ck = paths.MODELS / C.config(recipe, s).run_id("ckpt")
        if s == 42:
            row42 = CG.row(ck, gi)
            five[s] = row42["test"]["macro_f1"]
        else:
            five[s] = M.macro_f1(gi["y_te"], EA.predict_proba(ck, gi["x_te"]).argmax(1), 3)
        CG.log_test_use(ck, f"P{C.PHASE_NUM_B}", reason)
    cycle1 = json.loads(
        (paths.RESULTS / "studies" / "closing_gate" / "summary.json").read_text(encoding="utf-8")
    )
    served5 = {int(k): float(v) for k, v in cycle1["five_seed_test"]["augmented"].items()}
    out = {
        "declared_in": "configs/experiments/cycle5.yaml v5 H10b_release.closing_gate",
        "reason": reason,
        "recipe": recipe,
        "slice_version": gi["slice_version"],
        "checkpoints": {"h10b": row42},
        "five_seed_test": {
            "h10b": five,
            "served_recipe_cycle1": served5,
            "paired_h10b_minus_served_recipe": DEC.paired(served5, five),
        },
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    return out
