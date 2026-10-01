"""`vifeedback study`, Cycle 5: H10 consistency training on real typing (configs/experiments/cycle5.yaml)."""

from __future__ import annotations

import typer

from vifeedback.cli._apps import study_app


def _print_scored(scored: dict, flips: dict) -> None:
    for name, v in scored.items():
        typer.echo(
            f"  {name:28s} macro-F1 {v['macro_f1']:.4f}  neutral F1 {v['per_class']['neutral']['f1']:.3f}"
        )
    typer.echo(
        f"  {'vilexnorm_dev':28s} flip rate {flips['flip_rate']:.4f} "
        f"({flips['pairs']} pairs; {flips['flip_rate_changed_pairs']:.4f} on the "
        f"{flips['changed_pairs']} that differ after the transform)"
    )


@study_app.command("h10-split")
def study_h10_split() -> None:
    """Write the ViLexNorm development rows declared in cycle5.yaml (indices only, no text)."""
    from vifeedback.training import consistency as C

    typer.echo(f"  {C.N_DEV} development rows -> {C.write_dev_index()}")


@study_app.command("h10-run")
def study_h10_run(
    recipe: str = typer.Option(..., help="onesided | symmetric"),
    seed: int = typer.Option(42),
    smoke: bool = typer.Option(False, help="a one-minute wiring check; writes nothing"),
) -> None:
    """Cycle 5 H10 (cycle5.yaml): train one recipe at one seed; score it on the development sets.

    Writes the checkpoint to models/, a registry row for UIT-VSFC validation, and
    results/studies/cycle5/h10/<recipe>-s<seed>/ (labels and probabilities only). Never on a test split.
    """
    from vifeedback.training import consistency as C

    r = C.run(recipe, seed, smoke=smoke)
    _print_scored(r["scored"], r["flips"])


@study_app.command("h10-control")
def study_h10_control(seed: int = typer.Option(42)) -> None:
    """Cycle 5 H10: the served recipe's checkpoint for one seed, scored with the candidates' code."""
    from vifeedback.training import consistency as C

    r = C.evaluate_control(seed)
    _print_scored(r["scored"], r["flips"])


@study_app.command("h10-select")
def study_h10_select() -> None:
    """Cycle 5 H10 recipe selection at seed 42 (cycle5.yaml recipe_selection)."""
    from vifeedback.training import consistency as C

    out = C.select()
    c = out["control_s42"]
    typer.echo(
        f"  control s42: UIT-VSFC {c['uit_validation']:.4f}  "
        f"ViLexNorm dev flips {c['vilexnorm_dev_flip_rate']:.4f}"
    )
    for r, v in out["recipes"].items():
        typer.echo(
            f"  {r:10s} UIT-VSFC {v['uit_validation']:.4f} ({v['uit_minus_control']:+.4f})  "
            f"flips {v['vilexnorm_dev_flip_rate']:.4f}  "
            f"{'eligible' if v['eligible'] else 'NOT eligible'}"
        )
    typer.echo(f"  -> {out['outcome']}" + (f": {out['chosen']}" if out["chosen"] else ""))


@study_app.command("h10-confirm")
def study_h10_confirm(recipe: str = typer.Option(..., help="the recipe h10-select chose")) -> None:
    """Cycle 5 H10 confirmation (cycle5.yaml rule): 5 seeds each side, ViLexNorm test (logged)."""
    from vifeedback.training import consistency as C

    out = C.confirm(recipe)
    for name, r in out["rules"].items():
        detail = {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}
        typer.echo(f"  {name:36s} {'PASS' if r['passed'] else 'FAIL'}  {detail}")
    typer.echo(f"  H10 {recipe}: {'passed' if out['passed'] else 'not passed'}")


@study_app.command("h11-run")
def study_h11_run(
    init: str = typer.Option(..., help="teacher-alternate | pretrained-first6"),
    seed: int = typer.Option(42),
    smoke: bool = typer.Option(False, help="a wiring check; writes nothing but the teacher cache"),
) -> None:
    """Cycle 5 H11 (cycle5.yaml v2): distil one 6-layer student at one seed; score it.

    The teacher follows H10's decision (H10's recipe if it passed, else the served recipe). Writes
    the checkpoint to models/, a registry row, and results/studies/cycle5/h11/<init>-s<seed>/.
    """
    from vifeedback.training import distill as K

    r = K.run(init, seed, smoke=smoke)
    _print_scored(r["scored"], r["flips"])
    s = r["summary"]
    typer.echo(
        f"  teacher {s['teacher']}; {s['parameters'] / 1e6:.1f} M parameters, FP16 {s['fp16_weight_bytes'] / 1e6:.0f} MB"
    )


@study_app.command("h11-select")
def study_h11_select() -> None:
    """Cycle 5 H11 selection at seed 42 (cycle5.yaml v2)."""
    from vifeedback.training import distill as K

    out = K.select()
    for init, v in out["students"].items():
        typer.echo(
            f"  {init:18s} UIT-VSFC {v['uit_validation']:.4f}  NEU-ESC {v['neu_validation_all']:.4f}  "
            f"mean {v['mean']:.4f}"
        )
    typer.echo(f"  -> {out['chosen']}")


@study_app.command("h11-confirm")
def study_h11_confirm(init: str = typer.Option(..., help="the student h11-select chose")) -> None:
    """Cycle 5 H11 rule (cycle5.yaml v2): 5 seeds each side; the seed-42 FP16-storage graph checked."""
    from vifeedback.training import distill as K

    out = K.confirm(init)
    for name, r in out["rules"].items():
        detail = {
            k: round(v, 6) if isinstance(v, float) else v for k, v in r.items() if k != "passed"
        }
        typer.echo(f"  {name:40s} {'PASS' if r['passed'] else 'FAIL'}  {detail}")
    typer.echo(f"  H11 {init}: {'passed' if out['passed'] else 'not passed'}")


@study_app.command("h11-closing-gate")
def study_h11_closing_gate() -> None:
    """Cycle 5 H11 closing gate (cycle5.yaml v3): the students on UIT-VSFC test, once, logged."""
    from vifeedback.training import distill as K

    out = K.closing_gate()
    r = out["checkpoints"]["h11_student"]
    typer.echo(
        f"  seed 42: test macro-F1 {r['test']['macro_f1']:.4f} (excl. overlap "
        f"{r['test_excluding_train_overlap']['macro_f1']:.4f}), neutral F1 "
        f"{r['test']['per_class_f1']['neutral']:.3f}, no-diacritic "
        f"{r['robustness_test']['nodiacritic']['macro_f1']:.3f}"
    )
    f = out["five_seed_test"]
    p = f["paired_student_minus_teacher"]
    typer.echo(f"  5 seeds: student {f['student']}")
    typer.echo(f"  student - teacher (Cycle 1 test): {p['mean_delta']:+.4f} {p.get('ci95')}")


@study_app.command("h11-challenge")
def study_h11_challenge() -> None:
    """The seed-42 H11 student on challenge v1 (development data), for its model card."""
    from vifeedback.training import distill as K

    r = K.challenge_report()["h11_student"]
    typer.echo(
        f"  challenge v1: macro-F1 {r['macro_f1']:.4f}, accuracy {r['accuracy']:.4f}, "
        f"neutral F1 {r['per_class_f1']['neutral']:.3f}"
    )


def _print_b(scored: dict, dev: dict) -> None:
    for name, v in scored.items():
        typer.echo(
            f"  {name:28s} macro-F1 {v['macro_f1']:.4f}  neutral F1 {v['per_class']['neutral']['f1']:.3f}"
        )
    typer.echo(
        f"  {'vilexnorm_dev':28s} agreement {dev['agreement']:.4f}  flips {dev['flip_rate']:.4f}  "
        f"label_tv {dev['label_tv']:.3f}"
    )


@study_app.command("h10b-split")
def study_h10b_split() -> None:
    """Write H10b's 1,500 confirmation pairs (cycle5.yaml v4) as ViLexNorm train row indices."""
    from vifeedback.training import anchored as A

    typer.echo(f"  {A.N_CONFIRM} confirmation rows -> {A.write_confirm_index()}")


@study_app.command("h12p-split")
def study_h12p_split() -> None:
    """Write H12''s 3,000 held-out in-scope NEU-ESC train posts (cycle5.yaml v6) as row indices."""
    from vifeedback.training import h12p as H

    typer.echo(f"  {H.N_HOLDOUT} held-out rows -> {H.write_holdout_index()}")


@study_app.command("h10b-run")
def study_h10b_run(
    recipe: str = typer.Option(..., help="anchored_orig | anchored_both"),
    seed: int = typer.Option(42),
    smoke: bool = typer.Option(False, help="a wiring check; writes nothing but the teacher cache"),
) -> None:
    """Cycle 5 H10b (cycle5.yaml v4): consistency anchored on the frozen teacher, one recipe, one seed."""
    from vifeedback.training import anchored as A

    r = A.run(recipe, seed, smoke=smoke)
    _print_b(r["scored"], r["dev"])


@study_app.command("h10b-control")
def study_h10b_control(seed: int = typer.Option(42)) -> None:
    """Cycle 5 H10b: the served recipe's checkpoint for one seed, scored with the candidates' code."""
    from vifeedback.training import anchored as A

    r = A.evaluate_control(seed)
    _print_b(r["scored"], r["dev"])


@study_app.command("h10b-select")
def study_h10b_select() -> None:
    """Cycle 5 H10b recipe selection at seed 42 (cycle5.yaml v4)."""
    from vifeedback.training import anchored as A

    out = A.select()
    c = out["control_s42"]
    typer.echo(
        f"  control s42: UIT-VSFC {c['uit_validation']:.4f}  agreement {c['dev_agreement']:.4f}  "
        f"flips {c['dev_flip_rate']:.4f}  tv {c['dev_label_tv']:.3f}"
    )
    for r, v in out["recipes"].items():
        typer.echo(
            f"  {r:14s} UIT-VSFC {v['uit_validation']:.4f} ({v['uit_minus_control']:+.4f})  agreement "
            f"{v['dev_agreement']:.4f}  flips {v['dev_flip_rate']:.4f}  tv {v['dev_label_tv']:.3f}  "
            f"{'eligible' if v['eligible'] else 'NOT eligible'}"
        )
    typer.echo(f"  -> {out['outcome']}" + (f": {out['chosen']}" if out["chosen"] else ""))


@study_app.command("h10b-confirm")
def study_h10b_confirm(
    recipe: str = typer.Option(..., help="the recipe h10b-select chose"),
) -> None:
    """Cycle 5 H10b rule (cycle5.yaml v4): 5 seeds each side on the 1,500 confirmation pairs."""
    from vifeedback.training import anchored as A

    out = A.confirm(recipe)
    for name, r in out["rules"].items():
        detail = {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}
        typer.echo(f"  {name:36s} {'PASS' if r['passed'] else 'FAIL'}  {detail}")
    typer.echo(f"  H10b {recipe}: {'passed' if out['passed'] else 'not passed'}")


@study_app.command("h10b-closing-gate")
def study_h10b_closing_gate() -> None:
    """Cycle 5 H10b closing gate (cycle5.yaml v5): the five models on UIT-VSFC test, once, logged."""
    from vifeedback.training import anchored as A

    out = A.closing_gate()
    r = out["checkpoints"]["h10b"]
    typer.echo(
        f"  seed 42: test macro-F1 {r['test']['macro_f1']:.4f}, neutral F1 "
        f"{r['test']['per_class_f1']['neutral']:.3f}, no-diacritic "
        f"{r['robustness_test']['nodiacritic']['macro_f1']:.3f}"
    )
    f = out["five_seed_test"]
    p = f["paired_h10b_minus_served_recipe"]
    typer.echo(f"  5 seeds: {f['h10b']}")
    typer.echo(f"  H10b - served recipe (Cycle 1 test): {p['mean_delta']:+.4f} {p.get('ci95')}")
