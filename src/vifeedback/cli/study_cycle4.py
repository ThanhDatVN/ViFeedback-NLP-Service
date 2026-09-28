"""`vifeedback study`, Cycle 4: the served pipeline on NEU-ESC, H8 (student text from other
institutions), the B4' scope detector, and E1/S5' (INT8 on a larger acceptance set)."""

from __future__ import annotations

import json

import typer

from vifeedback import paths
from vifeedback.cli._apps import study_app


@study_app.command("served-neu-esc")
def study_served_neu_esc(split: str = typer.Option("validation", help="NEU-ESC split")) -> None:
    """NEXT_PLAN v5 A2 and A3, descriptive: the served pipeline on NEU-ESC, topic by topic.

    A2: how often `in_scope` flags each topic. Its threshold keeps 95% of UIT-VSFC validation;
    another institution's posts about academics or services are in scope, spam, news, jobs and
    club events are not. Rule, fixed in NEXT_PLAN v5 before this ran: Academic or Service flagged
    at more than 10% (twice validation's 5%) is a limitation for the card and the API docs, and the
    score is refitted (B4). A3: how the served labels meet NEU-ESC's, by gold label and topic.
    Labels, scores and topic names only are written.
    """
    import numpy as np
    import pandas as pd

    from vifeedback.constants import LABELS
    from vifeedback.evaluation import external as X
    from vifeedback.evaluation.audit import wilson
    from vifeedback.inference.onnx_export import OnnxClassifier
    from vifeedback.preprocess.segment import get_segmenter
    from vifeedback.preprocess.variants import VARIANTS
    from vifeedback.serving import pipeline as SP

    d = paths.MODELS / "serve" / "sentiment"
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    clf = OnnxClassifier(d, max_length=manifest.get("max_length", 96))
    if not manifest.get("ood"):
        raise typer.BadParameter("the served release has no out-of-scope score")
    restorer = SP.load_restorer(d, manifest["restorer"]) if manifest.get("restorer") else None
    ood = SP.load_ood(d, manifest["ood"], clf)
    seg = get_segmenter(VARIANTS[manifest["preprocessing"]][0])

    ne = X.load_neu_esc(split)
    texts = ne.text.tolist()
    pred_parts, scope_parts = [], []
    for i in range(0, len(texts), 64):
        ids, _, scope = SP.score(clf, SP.prepare(texts[i : i + 64], restorer, seg), ood)
        assert scope is not None  # the release has an out-of-scope score (checked above)
        pred_parts.append(ids)
        scope_parts.append(scope)
    pred = np.concatenate(pred_parts)
    scope = np.concatenate(scope_parts)
    flagged = scope < ood["threshold"]
    names = [LABELS["sentiment"][k] for k in sorted(LABELS["sentiment"])]
    y = ne.sentiment.map({c: i for i, c in enumerate(names)}).to_numpy().astype(int)
    topic = ne.topic.to_numpy()
    off = ("Spam", "News", "Jobs & Recruitment", "Club & Events")

    def describe(mask: np.ndarray) -> dict:
        n, k = int(mask.sum()), int(flagged[mask].sum())
        yy, pp = y[mask], pred[mask]
        gold_neutral = yy == names.index("neutral")
        return {
            "n": n,
            "flagged": k / n,
            "flagged_ci95": list(wilson(k, n)),
            "accuracy": float((yy == pp).mean()),
            "gold": {c: int((yy == j).sum()) for j, c in enumerate(names)},
            "predicted": {c: int((pp == j).sum()) for j, c in enumerate(names)},
            "confusion_gold_by_predicted": [
                [int(((yy == a) & (pp == b)).sum()) for b in range(3)] for a in range(3)
            ],
            "gold_neutral_predicted_polar": (
                float((pp[gold_neutral] != names.index("neutral")).mean())
                if gold_neutral.any()
                else None
            ),
        }

    by_topic = {t: {"off_topic": t in off, **describe(topic == t)} for t in sorted(set(topic))}
    in_scope_mask = ~np.isin(topic, off)
    from sklearn.metrics import roc_auc_score

    # Within one source: in-scope topics against off-topic ones (ADR-032). Higher = more in scope.
    within_auroc = float(roc_auc_score(in_scope_mask.astype(int), scope))
    watched = {t: by_topic[t]["flagged"] for t in ("Academic", "Service") if t in by_topic}
    limit = {t: v for t, v in watched.items() if v > 0.10}
    out = {
        "declared_in": "NEXT_PLAN v5 A2 (rule) and A3 (descriptive), before this ran",
        "split": split,
        "manifest_sha256": manifest["sha256"],
        "threshold": ood["threshold"],
        "in_scope_topics": describe(in_scope_mask),
        "off_topic_topics": describe(~in_scope_mask),
        "all": describe(np.ones(len(y), dtype=bool)),
        "within_source_auroc_in_scope_vs_off_topic": within_auroc,
        "by_topic": by_topic,
        "A2_rule": {
            "flagged_academic_service": watched,
            "limit_for_the_card": bool(limit),
            "refit_B4": bool(limit),
        },
    }
    dst = paths.RESULTS / "studies" / "cycle4_step0"
    dst.mkdir(parents=True, exist_ok=True)
    from vifeedback.evaluation.report import yaml_safe

    (dst / f"served_neu_esc_{split}.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "topic": topic,
            "gold": [names[i] for i in y],
            "pred": [names[i] for i in pred],
            "scope_score": scope.round(2),
            "in_scope": ~flagged,
        }
    ).to_csv(dst / f"served_neu_esc_{split}_predictions.csv", index_label="row")

    typer.echo(
        f"  NEU-ESC {split}: {len(y)} posts; threshold {ood['threshold']:.1f}; "
        f"in-scope vs off-topic AUROC {within_auroc:.3f}"
    )
    for t, r in by_topic.items():
        typer.echo(
            f"  {'off ' if r['off_topic'] else '    '}{t:20s} n {r['n']:5d}  flagged {r['flagged']:6.1%} "
            f"[{r['flagged_ci95'][0]:.1%}, {r['flagged_ci95'][1]:.1%}]  accuracy {r['accuracy']:.3f}"
        )
    typer.echo(
        f"  A2: Academic/Service flagged {', '.join(f'{t} {v:.1%}' for t, v in watched.items())}"
        f" -> {'LIMIT: card + refit (B4)' if limit else 'within 10%'}"
    )


@study_app.command("h8-run")
def study_h8_run(
    recipe: str = typer.Option(..., help="mixed | two-heads | sequential"),
    seed: int = typer.Option(42),
    smoke: bool = typer.Option(False, help="a one-minute wiring check; writes nothing"),
) -> None:
    """Cycle 4 H8 (cycle4.yaml): train one recipe at one seed; evaluate on the selection sets.

    Writes the checkpoint to models/, a registry row for UIT-VSFC validation, and
    results/studies/cycle4/h8/<recipe>-s<seed>/ (labels and probabilities only). Never on NEU-ESC test.
    """
    from vifeedback.training import domain as D

    r = D.run(recipe, seed, smoke=smoke)
    for name, v in r["scored"].items():
        typer.echo(
            f"  {name:28s} macro-F1 {v['macro_f1']:.4f}  neutral F1 {v['per_class']['neutral']['f1']:.3f}"
        )


@study_app.command("h8-control")
def study_h8_control(seed: int = typer.Option(42)) -> None:
    """Cycle 4 H8: the served recipe's checkpoint for one seed, scored with the candidates' code."""
    from vifeedback.training import domain as D

    r = D.evaluate_control(seed)
    for name, v in r["scored"].items():
        typer.echo(
            f"  {name:28s} macro-F1 {v['macro_f1']:.4f}  neutral F1 {v['per_class']['neutral']['f1']:.3f}"
        )


@study_app.command("h8-select")
def study_h8_select() -> None:
    """Cycle 4 H8 recipe selection at seed 42 (cycle4.yaml recipe_selection)."""
    from vifeedback.training import domain as D

    out = D.select()
    c = out["control_s42"]
    typer.echo(
        f"  control s42: UIT-VSFC {c['uit_validation']:.4f}  NEU-ESC {c['neu_validation']:.4f}"
    )
    for r, v in out["recipes"].items():
        typer.echo(
            f"  {r:11s} UIT-VSFC {v['uit_validation']:.4f} ({v['uit_minus_control']:+.4f})  "
            f"NEU-ESC {v['neu_validation']:.4f}  {'eligible' if v['eligible'] else 'NOT eligible'}"
        )
    typer.echo(f"  -> {out['outcome']}" + (f": {out['chosen']}" if out["chosen"] else ""))


@study_app.command("h8-confirm")
def study_h8_confirm(recipe: str = typer.Option(..., help="the recipe h8-select chose")) -> None:
    """Cycle 4 H8 confirmation (cycle4.yaml rule): 5 seeds each side, NEU-ESC test (logged)."""
    from vifeedback.training import domain as D

    out = D.confirm(recipe)
    for name, r in out["rules"].items():
        detail = {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}
        typer.echo(f"  {name:40s} {'PASS' if r['passed'] else 'FAIL'}  {detail}")
    typer.echo(f"  H8 {recipe}: {'passed' if out['passed'] else 'not passed'}")


@study_app.command("b4prime")
def study_b4prime() -> None:
    """B4' topic-aware scope detector (cycle4.yaml v3): select on NEU-ESC validation, then the rule
    on NEU-ESC test (logged). CPU; the served graph's features take about 20 minutes."""
    from vifeedback.evaluation import scope as SC

    out = SC.run()
    ch = out["selection"]["chosen"]
    typer.echo(f"  chosen: {ch['candidate']} C={ch['C']}  threshold {out['threshold']:.3f}")
    for g in out["selection"]["grid"]:
        typer.echo(
            f"    {g['candidate']:17s} C={g['C']:<5} NEU-ESC validation AUROC {g['neu_validation_auroc']:.4f}"
        )
    for name, r in out["rules"].items():
        typer.echo(
            f"  {name:28s} {r['value']:.4f} ({r['limit']})  {'PASS' if r['passed'] else 'FAIL'}"
        )
    rep = out["reported"]
    typer.echo(
        f"  served Mahalanobis within NEU-ESC test AUROC {rep['served_mahalanobis_within_neu_test_auroc']:.4f}; "
        f"U4 AUROC {rep['u4_vs_uit_validation_auroc']:.4f}"
    )
    typer.echo(f"  B4': {'passed' if out['passed'] else 'not passed'}")


@study_app.command("int8-power")
def study_int8_power(
    simulations: int = typer.Option(40),
    draws: int = typer.Option(2000),
    seed: int = typer.Option(0),
) -> None:
    """E1 (NEXT_PLAN v5): would UIT-VSFC + NEU-ESC validation demonstrate INT8 non-inferiority at
    0.005? Estimates INT8's disagreements on held-out data, simulates the acceptance set. CPU."""
    from vifeedback.evaluation import int8_power as IP

    out = IP.run(simulations=simulations, draws=draws, seed=seed)
    e = out["estimation"]
    typer.echo(
        f"  INT8 vs FP32 disagreement: UIT-VSFC train {e['disagreement_uit']:.2%}, NEU-ESC train {e['disagreement_neu']:.2%}"
    )
    for name, r in out["acceptance_sets"].items():
        typer.echo(
            f"  {name:26s} n {r['n']:5d}  upper bound median {r['upper_bound_median']:.4f} "
            f"[{r['upper_bound_p10_p90'][0]:.4f}, {r['upper_bound_p10_p90'][1]:.4f}]  "
            f"power (< 0.005) {r['power_below_0_005']:.0%}"
        )
    cal = out["calibration"]
    typer.echo(
        f"  calibration: observed UIT-VSFC bound at quantile {cal['observed_uit_bound_quantile_among_simulations']:.2f} "
        f"of the simulations -> {cal['reading']}"
    )


@study_app.command("int8-accept")
def study_int8_accept() -> None:
    """S5' (cycle4.yaml v4, ADR-035): careful INT8 against FP32 on UIT-VSFC + NEU-ESC validation. CPU."""
    from vifeedback.evaluation import int8_power as IP

    out = IP.accept()
    for name, r in out["rules"].items():
        typer.echo(
            f"  {name:24s} {r['value']:.4f} ({r['limit']})  {'PASS' if r['passed'] else 'FAIL'}"
        )
    for part in ("uit_validation", "neu_esc_validation"):
        p = out[part]
        typer.echo(
            f"  {part:20s} n {p['n']:5d}  agreement {p['label_agreement']:.2%}  "
            f"macro-F1 FP32 {p['macro_f1_fp32']:.4f} INT8 {p['macro_f1_int8']:.4f}"
        )
    typer.echo(f"  S5': {'passed' if out['passed'] else 'not passed'}")
