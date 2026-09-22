"""Command-line entrypoint.

Every experiment in this project is a CLI call, so the laptop and Colab run identical code paths and
notebooks stay logic-free (docs/ROADMAP.md § 4).
"""

from __future__ import annotations

import json

import typer

from vifeedback import paths

app = typer.Typer(add_completion=False, help="ViFeedback — Vietnamese feedback classification")

data_app = typer.Typer(help="Phase 0: acquisition, integrity, profiling")
baseline_app = typer.Typer(help="Phase 1: classical baselines")
app.add_typer(data_app, name="data")
train_app = typer.Typer(help="Phase 2+: transformer fine-tuning")
app.add_typer(baseline_app, name="baseline")
app.add_typer(train_app, name="train")


# --- Phase 0 ------------------------------------------------------------------------------------


@data_app.command("fetch")
def data_fetch(
    force: bool = typer.Option(False, help="Re-download and overwrite data/raw"),
) -> None:
    """Download UIT-VSFC and write data/raw/*.parquet + manifest.json."""
    from vifeedback.data.loader import fetch

    manifest = fetch(force=force)
    for split, meta in manifest["splits"].items():
        typer.echo(f"  {split:11s} {meta['rows']:>6,} rows  sha256={meta['sha256'][:16]}...")
    typer.echo(f"  source: {manifest.get('source')}")


@data_app.command("report")
def data_report(
    tokenizer: bool = typer.Option(
        True, help="Include subword-length profiling (downloads a model)"
    ),
    figures: bool = typer.Option(True, help="Also regenerate EDA figures"),
) -> None:
    """Run the full Phase 0 integrity + profiling report -> results/data_report.json."""
    from vifeedback.data.profile import build_report, make_figures, write_report

    report = build_report(with_tokenizer=tokenizer)
    write_report(report)

    integ = report["integrity"]
    head = integ["leakage"]["headline"]
    typer.echo(f"  structural checks: {'PASS' if integ['structural_passed'] else 'FAIL'}")
    typer.echo(
        f"  leakage (test seen in train): exact {head['test_rows_seen_in_train_exact']} "
        f"({head['share_of_test_exact']:.2%}) | "
        f"normalized {head['test_rows_seen_in_train_normalized']} "
        f"({head['share_of_test_normalized']:.2%})"
    )
    if "max_length_decision" in report:
        d = report["max_length_decision"]
        typer.echo(
            f"  max_length: {d['max_length']} ({d['vs_model_default_256']}x shorter than 256)"
        )
    if figures:
        for name in make_figures():
            typer.echo(f"  figure: results/figures/{name}")
    typer.echo(f"  written: {paths.RESULTS / 'data_report.json'}")


@data_app.command("variants")
def data_variants(
    name: str = typer.Option("all", help="variant name, or 'all'"),
    force: bool = typer.Option(False, help="Rebuild even if already materialized"),
) -> None:
    """Materialize preprocessing variants into data/processed/ (Phase 3)."""
    from vifeedback.preprocess.variants import VARIANTS, build_all, summary_table

    names = tuple(VARIANTS) if name == "all" else (name,)
    metas = build_all(names, force=force)
    typer.echo(summary_table(metas))


@data_app.command("segmenters")
def data_segmenters() -> None:
    """Report which segmentation backends this machine can run, and why not."""
    from vifeedback.preprocess.segment import available

    for backend, info in available().items():
        mark = "OK " if info["available"] else "-- "
        typer.echo(f"  {mark} {backend:<14s} {info['reason']}")


@data_app.command("env")
def data_env() -> None:
    """Print the captured environment (CPU flags, packages, git)."""
    from vifeedback import env

    typer.echo(json.dumps(env.capture(), indent=2, ensure_ascii=False))


# --- Phase 1 ------------------------------------------------------------------------------------


@baseline_app.command("run")
def baseline_run(
    task: str = typer.Option("sentiment", help="sentiment | topic"),
    seed: int = typer.Option(42),
    include_test: bool = typer.Option(False, help="Also evaluate on test (logged; gates only)"),
    reason: str = typer.Option("", help="Why the test set is being touched"),
    quiet: bool = typer.Option(False, help="Suppress the per-model reports"),
) -> None:
    """Run the B0-B5 baseline ladder for one task."""
    from vifeedback.models.runner import compare_best, run_ladder, summary_table

    if include_test and not reason:
        raise typer.BadParameter("--reason is required when evaluating on test")

    results = run_ladder(
        task, seed=seed, include_test=include_test, reason=reason, verbose=not quiet
    )

    for split in ("validation", "test"):
        if not any(r["split"] == split for r in results):
            continue
        typer.echo("")
        typer.echo(f"=== {task.upper()} / {split.upper()} ===")
        typer.echo(summary_table(results, split))

        cmp_ = compare_best(results, task, split)
        if cmp_:
            typer.echo("")
            typer.echo(
                f"  paired bootstrap: {cmp_['best']} vs {cmp_['baseline']}"
                f"  diff {cmp_['observed_diff']:+.4f}"
                f"  [{cmp_['ci_low']:+.4f}, {cmp_['ci_high']:+.4f}]"
                f"  p={cmp_['p_value']:.4f}"
                f"  {'SIGNIFICANT' if cmp_['significant'] else 'not significant'}"
            )


@baseline_app.command("registry")
def baseline_registry(
    split: str = typer.Option("validation", help="validation | test | all"),
) -> None:
    """Print the experiment registry and the test-set evaluation count."""
    import pandas as pd

    from vifeedback.evaluation.report import load_registry, test_evaluation_count

    df = load_registry()
    if len(df) == 0:
        typer.echo("  registry is empty")
        return
    if split != "all":
        df = df[df["split"] == split]
    cols = ["run_id", "task", "model", "recipe", "macro_f1", "weighted_f1", "accuracy"]
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        typer.echo(df[cols].to_string(index=False))
    typer.echo("")
    typer.echo(f"  test-set evaluations to date: {test_evaluation_count()}")


# --- Phase 2+ ------------------------------------------------------------------------------------


@train_app.command("run")
def train_run(
    task: str = typer.Option("sentiment", help="sentiment | topic"),
    model: str = typer.Option("phobert-base", help="key from constants.MODEL_IDS"),
    recipe: str = typer.Option("base", help="base | classweight | focal | logit-adjust | ..."),
    preprocessing: str = typer.Option("raw", help="variant name; raw = condition P0"),
    seeds: str = typer.Option("42", help="comma-separated, or 'all' for the 5 canonical seeds"),
    epochs: int = typer.Option(4),
    lr: float = typer.Option(2e-5),
    batch_size: int = typer.Option(32, help="per-step batch; see --grad-accum"),
    grad_accum: int = typer.Option(
        1,
        help="gradient accumulation steps. EFFECTIVE batch = batch_size * grad_accum. "
        "A model compared against another at a different effective batch is not a "
        "controlled comparison.",
    ),
    eval_batch_size: int = typer.Option(128),
    warmup_ratio: float = typer.Option(0.1),
    weight_decay: float = typer.Option(0.01),
    weight_scheme: str = typer.Option("balanced", help="balanced | sqrt | effective"),
    focal_gamma: float = typer.Option(2.0),
    tau: float = typer.Option(1.0, help="logit-adjustment tau"),
    freeze_embeddings: bool = typer.Option(
        False,
        help="freeze the embedding matrix. Makes xlm-roberta-base fit a 4GB GPU "
        "(192M of its 277M params are embeddings).",
    ),
    save_checkpoint: bool = typer.Option(False, help="write weights to models/<run_id>/"),
    max_length: int = typer.Option(96, help="Gate G0 decision: PhoBERT subword p99.9 = 87"),
    llrd: float = typer.Option(0.0, help="layer-wise lr decay, e.g. 0.9; 0 disables"),
    rdrop: float = typer.Option(0.0, help="R-Drop alpha; 0 disables"),
    fgm: float = typer.Option(0.0, help="FGM epsilon; 0 disables"),
    label_smoothing: float = typer.Option(0.0),
    phase: int = typer.Option(2, help="phase number, used in the run id"),
    include_test: bool = typer.Option(False, help="Also evaluate on test (logged; gates only)"),
    reason: str = typer.Option("", help="Why the test set is being touched"),
) -> None:
    """Fine-tune a transformer encoder across one or more seeds."""
    from vifeedback.constants import SEEDS
    from vifeedback.training import TrainConfig, format_summary, run_seeds

    if include_test and not reason:
        raise typer.BadParameter("--reason is required when evaluating on test")

    seed_tuple = SEEDS if seeds == "all" else tuple(int(s) for s in seeds.split(","))

    loss_for_recipe = {
        "base": "ce",
        "classweight": "classweight",
        "focal": "focal",
        "focal-weighted": "focal-weighted",
        "logit-adjust": "logit-adjust",
    }

    cfg = TrainConfig(
        task=task,
        model_key=model,
        recipe=recipe,
        preprocessing=preprocessing,
        loss=loss_for_recipe.get(recipe, "ce"),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        eval_batch_size=eval_batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        weight_scheme=weight_scheme,
        focal_gamma=focal_gamma,
        logit_adjust_tau=tau,
        freeze_embeddings=freeze_embeddings,
        max_length=max_length,
        llrd=llrd or None,
        rdrop_alpha=rdrop,
        fgm_epsilon=fgm,
        label_smoothing=label_smoothing,
        extra={"phase_num": phase},
    )
    typer.echo(f"  effective batch = {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    res = run_seeds(
        cfg,
        seed_tuple,
        include_test=include_test,
        reason=reason,
        save_checkpoint=save_checkpoint,
    )
    typer.echo("")
    typer.echo(
        format_summary(
            res["summary"], f"=== {task.upper()} / {model} / {recipe} / {len(seed_tuple)} seeds ==="
        )
    )


if __name__ == "__main__":
    app()
