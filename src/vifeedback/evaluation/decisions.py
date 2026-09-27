"""Cycle 1 decisions, computed by the rules declared in configs/experiments/cycle1.yaml.

The rules were committed before any Cycle 1 run. Encoding them here, rather than applying them by
reading a table, means the decision cannot drift toward whatever the numbers turned out to favour.

Controls come from the **same session and code**: every cRT run records its stage-1 CE model before
the head is replaced, so each seed has a CE control that differs from the treatments only in the
treatment. The registry's older CE rows are used only to check that stage 1 reproduces them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vifeedback import paths

ADVANCE_MIN_DELTA = 0.005
ADVANCE_MIN_WINS = 2
AUG_MIN_RELATIVE_REDUCTION = 0.20
AUG_MAX_CLEAN_LOSS = 0.005
AUG_PRIMARY_SUITE = "nodiacritic-50"


def load_runs(phase: int = 8, runs_dir: Path | None = None) -> dict[str, dict[int, dict[str, Any]]]:
    """{recipe: {seed: metrics}} for validation runs of one phase."""
    runs_dir = runs_dir or paths.RUNS
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for d in sorted(runs_dir.glob(f"p{phase}-*-val")):
        m, c = d / "metrics.json", d / "config.yaml"
        if not (m.exists() and c.exists()):
            continue
        import yaml

        cfg = yaml.safe_load(c.read_text(encoding="utf-8"))
        met = json.loads(m.read_text(encoding="utf-8"))
        out.setdefault(cfg["recipe"], {})[int(cfg["seed"])] = met
    return out


def _neutral(m: dict[str, Any]) -> dict[str, float]:
    n = m["per_class"]["neutral"]
    return {"precision": n["precision"], "recall": n["recall"], "f1": n["f1"]}


def paired(control: dict[int, float], treat: dict[int, float]) -> dict[str, Any]:
    seeds = sorted(set(control) & set(treat))
    d = np.array([treat[s] - control[s] for s in seeds])
    out: dict[str, Any] = {"seeds": seeds, "n": len(seeds)}
    if not len(d):
        return out
    out.update(
        {
            "per_seed_delta": [round(float(x), 4) for x in d],
            "mean_delta": float(d.mean()),
            "wins": int((d > 0).sum()),
        }
    )
    if len(d) >= 2:
        from scipy import stats

        half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
        out["ci95"] = [float(d.mean() - half), float(d.mean() + half)]
    return out


def h1(runs: dict[str, dict[int, dict[str, Any]]], registry_ce: dict[int, float]) -> dict[str, Any]:
    crt = runs.get("crt", {})
    control = {s: m["stage1_ce"]["macro_f1"] for s, m in crt.items() if "stage1_ce" in m}
    result: dict[str, Any] = {
        "control": "stage 1 (CE) of each cRT run",
        "control_macro_f1": {s: round(v, 4) for s, v in control.items()},
        "reproduces_registry": {
            s: {"stage1": round(control[s], 4), "registry": registry_ce.get(s)} for s in control
        },
        "methods": {},
    }
    treatments = {
        "crt": {s: m["macro_f1"] for s, m in crt.items()},
        "logit-adjust": {s: m["macro_f1"] for s, m in runs.get("logit-adjust", {}).items()},
    }
    for name, t in treatments.items():
        if not t:
            continue
        p = paired(control, t)
        src = runs["crt" if name == "crt" else "logit-adjust"]
        p["neutral"] = {s: _neutral(src[s]) for s in sorted(t)}
        p["neutral_control"] = {
            s: {
                k: crt[s]["stage1_ce"]["per_class"]["neutral"][k]
                for k in ("precision", "recall", "f1")
            }
            for s in sorted(control)
        }
        p["advance_to_5_seeds"] = bool(
            p.get("mean_delta", -1) >= ADVANCE_MIN_DELTA and p.get("wins", 0) >= ADVANCE_MIN_WINS
        )
        result["methods"][name] = p
    return result


def h2(runs: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    crt, aug = runs.get("crt", {}), runs.get("aug-diac-teen", {})
    seeds = sorted(
        s
        for s in set(crt) & set(aug)
        if crt[s].get("stage1_ce", {}).get("robustness") and aug[s].get("robustness")
    )
    if not seeds:
        return {"note": "no seed has both a control with robustness and an augmented run"}

    def deg(clean: float, rob: dict[str, Any], suite: str) -> float:
        return clean - rob[suite]["macro_f1"]

    ctrl_clean = {s: crt[s]["stage1_ce"]["macro_f1"] for s in seeds}
    aug_clean = {s: aug[s]["macro_f1"] for s in seeds}
    suites = list(aug[seeds[0]]["robustness"])
    per_suite = {}
    for suite in suites:
        dc = np.array([deg(ctrl_clean[s], crt[s]["stage1_ce"]["robustness"], suite) for s in seeds])
        da = np.array([deg(aug_clean[s], aug[s]["robustness"], suite) for s in seeds])
        per_suite[suite] = {
            "control_shifted": [
                round(crt[s]["stage1_ce"]["robustness"][suite]["macro_f1"], 4) for s in seeds
            ],
            "augmented_shifted": [round(aug[s]["robustness"][suite]["macro_f1"], 4) for s in seeds],
            "control_degradation_mean": float(dc.mean()),
            "augmented_degradation_mean": float(da.mean()),
            "relative_reduction": float(1 - da.mean() / dc.mean()) if dc.mean() > 0 else None,
            "in_family": not suite.startswith("charnoise"),
        }
    clean = paired(ctrl_clean, aug_clean)
    primary = per_suite[AUG_PRIMARY_SUITE]
    supported = bool(
        (primary["relative_reduction"] or 0) >= AUG_MIN_RELATIVE_REDUCTION
        and -clean["mean_delta"] <= AUG_MAX_CLEAN_LOSS
    )
    return {
        "seeds": seeds,
        "clean": clean,
        "suites": per_suite,
        "augmentation_exposure": {s: aug[s].get("augmentation") for s in seeds},
        "primary_suite": AUG_PRIMARY_SUITE,
        "criterion": f"relative reduction >= {AUG_MIN_RELATIVE_REDUCTION} and clean loss <= {AUG_MAX_CLEAN_LOSS}",
        "n_seeds": len(seeds),
        "supported": supported,
    }


MTL_MIN_GAIN = 0.005
MTL_MAX_HARM = -0.005


def load_multitask(
    phase: int = 8, runs_dir: Path | None = None
) -> dict[str, dict[str, dict[int, dict]]]:
    """{recipe: {task: {seed: metrics}}} for the shared-encoder runs (recipe mtl-l*)."""
    import yaml

    runs_dir = runs_dir or paths.RUNS
    out: dict[str, dict[str, dict[int, dict]]] = {}
    for d in sorted(runs_dir.glob(f"p{phase}-*-mtl-l*-val")):
        m, c = d / "metrics.json", d / "config.yaml"
        if not (m.exists() and c.exists()):
            continue
        cfg = yaml.safe_load(c.read_text(encoding="utf-8"))
        out.setdefault(cfg["recipe"], {}).setdefault(cfg["task"], {})[int(cfg["seed"])] = (
            json.loads(m.read_text(encoding="utf-8"))
        )
    return out


def h4(
    mtl: dict[str, dict[str, dict[int, dict]]], single: dict[str, dict[int, float]]
) -> dict[str, Any]:
    """Apply the H4 rules of cycle1.yaml v2. `single` = {task: {seed: single-task macro-F1}}."""
    out: dict[str, Any] = {}
    for recipe, by_task in sorted(mtl.items()):
        per = {
            t: paired(single.get(t, {}), {s: m["macro_f1"] for s, m in runs.items()})
            for t, runs in by_task.items()
        }
        means = {t: p.get("mean_delta", float("nan")) for t, p in per.items()}
        wins = {t: p.get("wins", 0) for t, p in per.items()}
        tasks = list(per)
        helps = any(
            means[t] >= MTL_MIN_GAIN
            and wins[t] >= ADVANCE_MIN_WINS
            and all(means[o] >= MTL_MAX_HARM for o in tasks if o != t)
            for t in tasks
        )
        negative = [t for t in tasks if means[t] <= MTL_MAX_HARM and wins[t] <= 1]
        verdict = (
            "helps"
            if helps
            else "negative_transfer"
            if negative
            else "no_material_difference"
            if all(abs(v) < MTL_MIN_GAIN for v in means.values())
            else "inconclusive"
        )
        any_task = next(iter(by_task.values()))
        out[recipe] = {
            "per_task": per,
            "verdict": verdict,
            "negative_transfer_tasks": negative,
            "joint_exact_match": {
                s: m.get("multitask", {}).get("joint_exact_match")
                for s, m in sorted(any_task.items())
            },
            "advance_to_5_seeds": verdict == "helps",
        }
    return out


H3_WITHDRAW_THRESHOLD = 0.8523  # half the gap from xlmr+pyvi (0.8403) to phobert-base+pyvi (0.8643)


def h3(registry: Any) -> dict[str, Any]:
    """Apply the H3 rule of cycle1.yaml: XLM-R raw minus XLM-R pyvi, seed-paired, same session.

    `registry` is the registry DataFrame. Only phase-8 rows are used, so both arms come from the one
    Kaggle session that ran them together.
    """
    if registry.empty or "split" not in registry.columns:
        return {"note": "H3 runs not found in the registry"}
    v = registry[registry.split == "validation"]

    def arm(prep: str) -> dict[int, float]:
        q = v[v.run_id.astype(str).str.startswith(f"p8-sent-xlmr-base-{prep}-")]
        return {int(s): float(m) for s, m in zip(q.seed, q.macro_f1, strict=True)}

    raw, pyvi = arm("raw"), arm("seg_pyvi")
    out: dict[str, Any] = {"raw": raw, "pyvi": pyvi, "paired": paired(pyvi, raw)}
    if not raw or not pyvi:
        out["note"] = "H3 runs not found in the registry"
        return out
    p = out["paired"]
    out["supported"] = bool(p.get("ci95") and p["ci95"][0] > 0)
    out["raw_mean"] = float(np.mean(list(raw.values())))
    out["withdraw_adr016"] = out["raw_mean"] >= H3_WITHDRAW_THRESHOLD
    out["threshold"] = H3_WITHDRAW_THRESHOLD
    return out
