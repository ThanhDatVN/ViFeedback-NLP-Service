"""Command-line entrypoint.

Every experiment in this project is a CLI call, so the laptop and Colab run identical code paths and
notebooks stay logic-free (docs/ROADMAP.md § 4).
"""

from __future__ import annotations

import json

import typer

from vifeedback import paths

# Tracebacks never print local variables: a failing API call holds the key in its locals (.env).
app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="ViFeedback — Vietnamese feedback classification",
)


@app.callback()
def _startup() -> None:
    """ViFeedback — Vietnamese feedback classification."""
    from vifeedback.env import load_dotenv

    load_dotenv()  # keys from the git-ignored .env (OPENAI_API_KEY, HF_TOKEN); terminal values win


data_app = typer.Typer(
    pretty_exceptions_show_locals=False, help="Phase 0: acquisition, integrity, profiling"
)
baseline_app = typer.Typer(pretty_exceptions_show_locals=False, help="Phase 1: classical baselines")
app.add_typer(data_app, name="data")
train_app = typer.Typer(
    pretty_exceptions_show_locals=False, help="Phase 2+: transformer fine-tuning"
)
app.add_typer(baseline_app, name="baseline")
serve_app = typer.Typer(
    pretty_exceptions_show_locals=False, help="Phase 6-7: export, benchmark, serve"
)
app.add_typer(train_app, name="train")
app.add_typer(serve_app, name="serve")
study_app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="Research studies (docs/REVIEW_AND_RESEARCH_PLAN.md § 7)",
)
app.add_typer(study_app, name="study")


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
    augment: str = typer.Option(
        "", help="augmentation recipe (training/augment.py), e.g. diac-teen"
    ),
    augment_p: float = typer.Option(
        0.0, help="share of train sentences replaced by a perturbed copy"
    ),
    crt_epochs: int = typer.Option(0, help=">0: balanced classifier re-training after CE (cRT)"),
    crt_lr: float = typer.Option(1e-3, help="cRT head learning rate"),
    robustness: bool = typer.Option(False, help="evaluate validation under the Cycle 1 suites"),
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
        augment=augment,
        augment_p=augment_p,
        crt_epochs=crt_epochs,
        crt_lr=crt_lr,
        extra={"phase_num": phase},
    )
    typer.echo(f"  effective batch = {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    res = run_seeds(
        cfg,
        seed_tuple,
        include_test=include_test,
        reason=reason,
        save_checkpoint=save_checkpoint,
        robustness=robustness,
    )
    typer.echo("")
    typer.echo(
        format_summary(
            res["summary"], f"=== {task.upper()} / {model} / {recipe} / {len(seed_tuple)} seeds ==="
        )
    )


@train_app.command("multitask")
def train_multitask_cmd(
    lam: float = typer.Option(
        1.0, "--lambda", help="topic loss weight: L = L_sent + lambda * L_topic"
    ),
    model: str = typer.Option("phobert-base"),
    preprocessing: str = typer.Option("seg_pyvi"),
    seeds: str = typer.Option("42,1337,2024"),
    phase: int = typer.Option(8),
) -> None:
    """Cycle 1 H4: one shared encoder, sentiment + topic heads (training/multitask.py).

    Writes one registry row per task, so each task is compared to its single-task control with the
    same seed-paired tooling as every other run.
    """
    import dataclasses

    from vifeedback.evaluation import bootstrap as B
    from vifeedback.evaluation import report as R
    from vifeedback.preprocess.variants import load_variant
    from vifeedback.training.multitask import TASKS, evaluate_joint, predict, train_multitask
    from vifeedback.training.runner import _free_cuda
    from vifeedback.training.trainer import TrainConfig

    tr, dv = load_variant(preprocessing, "train"), load_variant(preprocessing, "validation")
    train = (tr["sentence"].tolist(), tr["sentiment"].to_numpy(), tr["topic"].to_numpy())
    dev = (dv["sentence"].tolist(), dv["sentiment"].to_numpy(), dv["topic"].to_numpy())
    recipe = f"mtl-l{lam:g}"

    for seed in (int(s) for s in seeds.split(",")):
        base = TrainConfig(
            task="sentiment",
            model_key=model,
            preprocessing=preprocessing,
            recipe=recipe,
            seed=seed,
            extra={"phase_num": phase},
        )
        typer.echo(f"\n[{recipe} seed {seed}]  lambda={lam}")
        out = train_multitask(base, lam, train, dev)
        logits = predict(out["model"], out["loader"], out["device"], base.fp16)
        y = {"sentiment": dev[1], "topic": dev[2]}
        joint = evaluate_joint(y, logits)
        for task in TASKS:
            m = joint["per_task"][task]
            k = len(m["labels"])
            ci = B.bootstrap_ci(y[task], logits[task].argmax(1), k, n_resamples=2000, seed=seed)
            m["macro_f1_ci"] = [ci["ci_low"], ci["ci_high"]]
            m["history"] = out["history"]
            m["best_epoch"] = out["best_epoch"]
            m["multitask"] = {
                "lambda_topic": lam,
                "joint_exact_match": joint["joint_exact_match"],
                "selection": "mean of sentiment and topic dev macro-F1",
            }
            cfg_t = dataclasses.replace(base, task=task)
            config = dict(out["config"]) | {
                "task": task,
                "phase": f"P{phase}",
                "model": model,
                "split": "validation",
                "fit_seconds": out["train_seconds"],
                "notes": f"shared encoder, lambda={lam}",
                "reason": "cycle 1 H4",
                "determinism": out["determinism"],
                "device": out["device"],
                "best_epoch": out["best_epoch"],
            }
            R.save_run(
                cfg_t.run_id("validation"),
                m,
                config=config,
                y_true=y[task],
                y_pred=logits[task].argmax(1),
                y_prob=softmax_np(logits[task]),
            )
            typer.echo(f"  {task:9s} macro-F1 {m['macro_f1']:.4f}")
        typer.echo(f"  joint exact match {joint['joint_exact_match']:.4f}")
        del out
        _free_cuda()


def softmax_np(z):
    import numpy as np

    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# --- Phase 6-7 -----------------------------------------------------------------------------------


@serve_app.command("export")
def serve_export(
    checkpoint: str = typer.Option(..., help="path to a saved HF checkpoint"),
    task: str = typer.Option("sentiment"),
    preprocessing: str = typer.Option(
        "", help="the checkpoint's training preprocessing; inferred from its name if omitted"
    ),
    max_length: int = typer.Option(96),
    quantize: str = typer.Option("dynamic", help="none | dynamic | static"),
    out: str = typer.Option("", help="defaults to models/serve/<task>"),
    calib_size: int = typer.Option(300, help="stratified train sentences for static INT8"),
) -> None:
    """Export, verify against the quality contract, and release (review R3).

    Built in a staging directory; the served directory is replaced only if FP32 logits match, or
    INT8 stays within 0.005 macro-F1 of PyTorch on the full validation set.
    """
    from pathlib import Path

    from vifeedback.inference.release import release, stratified_subset
    from vifeedback.preprocess.variants import VARIANTS
    from vifeedback.training.runner import _raw_and_pipeline

    if not preprocessing:
        name = Path(checkpoint).name
        found = [v for v in sorted(VARIANTS, key=len, reverse=True) if v != "raw" and v in name]
        preprocessing = found[0] if found else ""
        if not preprocessing:
            raise typer.BadParameter(
                "cannot infer the checkpoint's preprocessing from its name; pass --preprocessing"
            )
    typer.echo(f"  preprocessing {preprocessing}")

    from vifeedback.preprocess.variants import load_variant

    raw_tr, pipeline = _raw_and_pipeline(preprocessing, "train")
    y_tr = load_variant(preprocessing, "train")[task].to_numpy()
    raw_dv, _ = _raw_and_pipeline(preprocessing, "validation")
    y_dv = load_variant(preprocessing, "validation")[task].to_numpy()
    calib_idx = stratified_subset(y_tr, calib_size)

    manifest = release(
        checkpoint=checkpoint,
        task=task,
        preprocessing=preprocessing,
        quantize=quantize,
        out_dir=Path(out) if out else paths.MODELS / "serve" / task,
        pipeline=pipeline,
        calib_raw=[raw_tr[i] for i in calib_idx],
        accept_raw=raw_dv,
        accept_y=y_dv,
        max_length=max_length,
        log=typer.echo,
    )
    typer.echo(f"  manifest: {manifest['model_file']}  sha256 {manifest['sha256'][:12]}...")


@serve_app.command("publish")
def serve_publish(
    repo_id: str = typer.Option(
        ..., help="Hugging Face repository, e.g. <user>/vifeedback-sentiment"
    ),
    upload: bool = typer.Option(False, help="actually upload; without it this is a dry run"),
    private: bool = typer.Option(False, help="create the repository as private"),
    pytorch: bool = typer.Option(
        True, help="include the PyTorch checkpoint next to the ONNX graph"
    ),
) -> None:
    """Bundle the served model with a model card and checksums; upload only with --upload (R11).

    The dry run builds models/publish/<name>/ so the card and files can be reviewed first. The
    upload uses your own Hugging Face login (`hf auth login`) or HF_TOKEN; nothing is stored here.
    """
    from vifeedback.inference import publish as P

    b = P.build(repo_id, include_pytorch=pytorch)
    typer.echo(f"  bundle: {b['folder']}  ({b['bytes'] / 1e6:.0f} MB, {len(b['files'])} files)")
    for name, sha in b["files"].items():
        typer.echo(f"    {sha[:12]}  {name}")
    if not upload:
        typer.echo(f"  dry run: review {b['folder'] / 'README.md'}, then re-run with --upload")
        return
    typer.echo(f"  uploading to {repo_id} ...")
    typer.echo("  " + P.upload(repo_id, b["folder"], private=private))


@serve_app.command("bench")
def serve_bench(
    task: str = typer.Option("sentiment"),
    model_dir: str = typer.Option("", help="defaults to models/serve/<task>"),
    threads: int = typer.Option(0, help="0 = ORT default"),
    timed: int = typer.Option(1000),
) -> None:
    """Benchmark CPU latency on the reference machine. Never run this on a cloud VM."""
    import json
    from pathlib import Path

    from vifeedback import paths
    from vifeedback.data.loader import load
    from vifeedback.inference import benchmark as BM
    from vifeedback.inference.onnx_export import OnnxClassifier
    from vifeedback.preprocess.segment import get_segmenter
    from vifeedback.preprocess.variants import VARIANTS

    d = Path(model_dir) if model_dir else paths.MODELS / "serve" / task
    manifest = (
        json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if (d / "manifest.json").exists()
        else {}
    )
    clf = OnnxClassifier(d, threads or None, manifest.get("max_length", 96))
    # Time the preprocessing the artifact was verified with, not an assumed one (review R3/R10).
    backend = VARIANTS[manifest["preprocessing"]][0] if manifest.get("preprocessing") else "pyvi"
    seg = get_segmenter(backend)
    texts = load("test")["sentence"].tolist()

    rep = BM.benchmark_pipeline(
        model_fn=clf.predict,
        preprocess_fn=seg,
        texts=texts,
        label=f"{clf.path.name} threads={threads or 'default'}",
        size_mb=clf.size_mb,
        timed=timed,
    )
    out = paths.RESULTS / f"bench_{task}_{clf.path.stem}.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    m, e = rep["model_only"], rep["end_to_end"]
    typer.echo(f"  size          {rep['size_mb']} MB")
    typer.echo(f"  model p50/p95 {m['p50_ms']:.2f} / {m['p95_ms']:.2f} ms")
    typer.echo(f"  e2e   p50/p95 {e['p50_ms']:.2f} / {e['p95_ms']:.2f} ms")
    typer.echo(f"  preprocessing {rep['preprocess_share_of_p95']:.1%} of end-to-end p95")
    typer.echo(f"  throughput    {rep['throughput']}")
    if m["throttling_suspected"]:
        typer.secho(f"  WARNING: {m['warning']}", fg="red")
    typer.echo(f"  written       {out}")


@serve_app.command("run")
def serve_run(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
) -> None:
    """Run the API locally."""
    import uvicorn

    uvicorn.run("vifeedback.serving.app:app", host=host, port=port, reload=reload)


# --- Studies -------------------------------------------------------------------------------------


@study_app.command("neutral-audit")
def study_neutral_audit(
    checkpoint: str = typer.Option(
        "models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp",
        help="saved sentiment checkpoint used for the dev view",
    ),
    preprocessing: str = typer.Option(
        "seg_pyvi", help="must match the checkpoint's training input"
    ),
    oof: bool = typer.Option(True, help="also run k-fold OOF fine-tuning over train (~20 GPU-min)"),
    k: int = typer.Option(5, help="OOF folds"),
    seed: int = typer.Option(42),
) -> None:
    """Study A, steps 1 and 3: dev/OOF predictions, uncertainty, and a stratified audit sheet.

    Text-free prediction files and the summary are written to results/studies/study_a/ and are
    committed. Anything containing corpus text goes to its local/ subfolder, which is gitignored:
    the raw data is not redistributed (docs/DATA_CARD.md § 11).
    """
    import pandas as pd

    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import metrics as M
    from vifeedback.preprocess.variants import variant_dir

    out = paths.RESULTS / "studies" / "study_a"
    local = out / "local"
    local.mkdir(parents=True, exist_ok=True)
    task = "sentiment"

    def _load(split: str) -> pd.DataFrame:
        return pd.read_parquet(variant_dir(preprocessing) / f"{split}.parquet")

    tr, dv = _load("train"), _load("validation")
    summary: dict = {
        "task": task,
        "preprocessing": preprocessing,
        "checkpoint": checkpoint,
        "caveat_dev": "the checkpoint selected its epoch on validation, so the dev view is mildly "
        "optimistic; it describes errors, it does not estimate generalization",
    }

    typer.echo(f"dev predictions from {checkpoint}")
    dev_p = EA.predict_proba(checkpoint, dv["sentence"].tolist())
    dev_t = EA.prediction_table(
        dv["sentence_raw"].tolist(),
        dv["sentence"].tolist(),
        dv[task].to_numpy(),
        dev_p,
        task,
        split="validation",
        source="checkpoint",
    )
    summary["dev_metrics"] = M.evaluate(dev_t.gold_id, dev_t.pred_id, task, y_prob=dev_p)

    tables = {"validation": dev_t}
    if oof:
        from vifeedback.training.trainer import TrainConfig

        cfg = TrainConfig(
            task=task, model_key="phobert-base", preprocessing=preprocessing, seed=seed
        )
        typer.echo(f"OOF: {k} folds over train ({len(tr)} rows)")
        r = EA.oof_predictions(
            cfg,
            tr["sentence"].tolist(),
            tr[task].to_numpy(),
            dv["sentence"].tolist(),
            dv[task].to_numpy(),
            k=k,
            fold_seed=seed,
        )
        tr_t = EA.prediction_table(
            tr["sentence_raw"].tolist(),
            tr["sentence"].tolist(),
            tr[task].to_numpy(),
            r["oof_probs"],
            task,
            split="train",
            source="oof",
        )
        tr_t["fold"] = r["fold_id"]
        tables["train"] = tr_t
        dis = EA.fold_disagreement(r["dev_probs"])
        dev_t["fold_vote_agreement"] = dis["vote_agreement"]
        dev_t["fold_prob_std_max"] = dis["prob_std_max"]
        summary["oof"] = {
            "k": k,
            "config_hash": cfg.config_hash(),
            "folds": r["folds"],
            "gpu_seconds": r["total_seconds"],
            "metrics": M.evaluate(tr_t.gold_id, tr_t.pred_id, task, y_prob=r["oof_probs"]),
            "dev_fold_ensemble_metrics": M.evaluate(
                dev_t.gold_id,
                r["dev_probs"].mean(axis=0).argmax(axis=1),
                task,
                y_prob=r["dev_probs"].mean(axis=0),
            ),
        }
        issues = EA.suspected_label_issues(tr_t, top_k=200)
        summary["suspected_issues_top200_by_gold_pred"] = (
            issues.groupby(["gold", "pred"]).size().rename("n").reset_index().to_dict("records")
        )

    samples = []
    for split, t in tables.items():
        t.to_csv(
            local / f"{split}_predictions_with_text.csv", index_label="row", encoding="utf-8-sig"
        )
        # 6 significant digits, not fixed decimals: a probability of 3e-6 rounded to 5 decimals
        # becomes 0, and NLL / temperature fitting on the saved file would then be wrong.
        text_free = t.drop(columns=["text", "model_input"])
        text_free.to_csv(out / f"{split}_predictions.csv", index_label="row", float_format="%.6g")
        summary[f"error_rate_by_flag_{split}"] = EA.error_rate_by_flag(t).to_dict("records")
        summary[f"error_rate_by_flag_{split}_neutral"] = EA.error_rate_by_flag(
            t, gold_class="neutral"
        ).to_dict("records")
        samples.append(
            EA.stratified_audit_sample(
                t, per_error_cell=8, per_correct_class=5, n_random=20, seed=seed
            )
        )

    sample = pd.concat(samples, ignore_index=True)
    sheet = EA.export_annotation_sheet(sample, local / "audit_sheet.csv")
    sample[["split", "example_index", "stratum", "gold", "pred"]].to_csv(
        out / "audit_sample_index.csv", index=False
    )
    summary["audit_sample"] = {
        "n": len(sample),
        "by_split_stratum": sample.groupby(["split", "stratum"]).size().to_dict(),
    }
    summary["audit_sample"]["by_split_stratum"] = {
        f"{a}|{b}": int(v) for (a, b), v in summary["audit_sample"]["by_split_stratum"].items()
    }

    from vifeedback.evaluation.report import yaml_safe

    (out / "summary.json").write_text(
        json.dumps(yaml_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(f"  dev macro-F1 {summary['dev_metrics']['macro_f1']:.4f}")
    if oof:
        typer.echo(f"  OOF macro-F1 {summary['oof']['metrics']['macro_f1']:.4f}")
    typer.echo(f"  audit sheet ({len(sample)} rows): {sheet}")
    typer.echo(f"  summary: {out / 'summary.json'}")


@study_app.command("calibration")
def study_calibration(
    n_bins: int = typer.Option(15, help="ECE bins; disclosed in the output"),
    seed: int = typer.Option(42),
) -> None:
    """E09: calibration of saved predictions, temperature cross-fitted on disjoint halves.

    Reads Study A's prediction files, so it needs no GPU. Two views: the deployed checkpoint on
    validation, and the fold models on their held-out train folds (OOF, if present).
    """
    import pandas as pd

    from vifeedback.constants import label_names
    from vifeedback.evaluation import calibration as C
    from vifeedback.evaluation.report import yaml_safe

    src = paths.RESULTS / "studies" / "study_a"
    out = paths.RESULTS / "studies" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    cols = [f"p_{n}" for n in label_names("sentiment")]
    report: dict = {"n_bins": n_bins, "protocol": "cross-fit on stratified halves", "views": {}}

    for view, fname in (("checkpoint_on_validation", "validation"), ("oof_on_train", "train")):
        f = src / f"{fname}_predictions.csv"
        if not f.exists():
            typer.echo(f"  skip {view}: {f.name} not found")
            continue
        df = pd.read_csv(f)
        probs, y = df[cols].to_numpy(), df["gold_id"].to_numpy()
        r = C.cross_fit_temperature(C.probs_to_logits(probs), y, seed=seed, n_bins=n_bins)
        r.pop("calibrated_probs")
        r["risk_coverage_uncalibrated"] = C.risk_coverage(probs, y)
        r["aurc"] = C.selective_scores(probs, y)  # E10: which confidence signal ranks errors best
        report["views"][view] = r
        u, t = r["uncalibrated"], r["temperature_scaled"]
        typer.echo(
            f"  {view:26s} n={u['n']:5d}  T={r['temperatures']}  "
            f"NLL {u['nll']:.4f}->{t['nll']:.4f}  Brier {u['brier']:.4f}->{t['brier']:.4f}  "
            f"ECE(w) {u['ece_equal_width']:.4f}->{t['ece_equal_width']:.4f}"
        )

    (out / "summary.json").write_text(
        json.dumps(yaml_safe(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(f"  written {out / 'summary.json'}")


@study_app.command("robustness")
def study_robustness(
    checkpoint: str = typer.Option("models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp"),
    segmenter: str = typer.Option("pyvi", help="the checkpoint's training segmenter"),
    split: str = typer.Option("validation", help="test is reserved for final confirmation"),
    seed: int = typer.Option(42),
    n_boot: int = typer.Option(2000),
) -> None:
    """Study B: perturbation suites and predefined slices, paired against clean predictions.

    Perturbations are applied to raw text and then segmented, as the deployed pipeline would.
    """

    import numpy as np

    from vifeedback.constants import label_names
    from vifeedback.data.loader import load
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import robustness as R
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.preprocess.segment import get_segmenter

    if split == "test":
        raise typer.BadParameter("test is reserved for the final confirmation run")

    task = "sentiment"
    k = len(label_names(task))
    df = load(split)
    raw, y = df["sentence"].tolist(), df[task].to_numpy()
    seg = get_segmenter(segmenter)
    out = paths.RESULTS / "studies" / "robustness"
    out.mkdir(parents=True, exist_ok=True)

    clean = EA.predict_proba(checkpoint, seg(raw)).argmax(1)
    report: dict = {
        "suite_version": R.SUITE_VERSION,
        "slices_version": R.SLICES_VERSION,
        "checkpoint": checkpoint,
        "segmenter": segmenter,
        "split": split,
        "seed": seed,
        "suites": {},
        "slices": R.slice_report(y, clean, R.slice_masks(raw), k),
    }
    for suite in R.SUITES:
        pert, changed = R.perturb(raw, suite, seed=seed)
        pred = EA.predict_proba(checkpoint, seg(pert)).argmax(1)
        entry = R.paired_delta(y, clean, pred, k, n_boot=n_boot, seed=seed)
        entry["changed_share"] = float(changed.mean())
        if changed.sum() >= 30:
            entry["changed_only"] = R.paired_delta(
                y[changed], clean[changed], pred[changed], k, n_boot=n_boot, seed=seed
            )
        # Where predictions go under shift. A rising minority-class recall can mean the model is
        # dumping unreadable input into that class, which the aggregate delta alone would hide.
        entry["pred_share_clean"] = (np.bincount(clean, minlength=k) / len(clean)).tolist()
        entry["pred_share_shifted"] = (np.bincount(pred, minlength=k) / len(pred)).tolist()
        entry["neutral_recall_clean"] = float((clean[y == 1] == 1).mean())
        entry["neutral_recall_shifted"] = float((pred[y == 1] == 1).mean())
        report["suites"][suite] = entry
        typer.echo(
            f"  {suite:16s} changed {entry['changed_share']:6.1%}  macro-F1 "
            f"{entry['macro_f1_a']:.4f} -> {entry['macro_f1_b']:.4f}  "
            f"delta {entry['delta']:+.4f} [{entry['delta_ci'][0]:+.4f}, {entry['delta_ci'][1]:+.4f}]"
        )
    probe = paths.DATA / "probes" / "negation_v1.csv"
    if probe.exists():
        import pandas as pd

        report["negation_probe"] = R.negation_probe(
            pd.read_csv(probe),
            lambda xs: EA.predict_proba(checkpoint, seg(xs)).argmax(1),
        )
        for name, r in report["negation_probe"].items():
            typer.echo(
                f"  negation {name:11s} pairs {r['n_pairs']:2d}  pair-acc {r['pair_accuracy']:.3f}  "
                f"flip {r['flip_rate']:.3f}  base-acc {r['base_accuracy']:.3f}"
            )
    if hasattr(seg, "close"):
        seg.close()

    (out / f"{split}.json").write_text(
        json.dumps(yaml_safe(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(f"  written {out / f'{split}.json'}")


@study_app.command("tables")
def study_tables() -> None:
    """Regenerate result tables and the declared paired comparisons from the registry."""
    from vifeedback.evaluation import compare_runs as CR
    from vifeedback.evaluation.report import yaml_safe

    out = paths.RESULTS / "studies" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    r = CR.build()
    r["table"].to_csv(out / "conditions.csv", index=False, float_format="%.6g")
    (out / "conditions.md").write_text(
        "<!-- generated by `vifeedback study tables`; do not edit by hand -->\n\n"
        + CR.to_markdown(r["table"])
        + "\n",
        encoding="utf-8",
    )
    (out / "comparisons.json").write_text(
        json.dumps(yaml_safe({"hygiene": r["hygiene"], "comparisons": r["comparisons"]}), indent=2),
        encoding="utf-8",
    )
    h = r["hygiene"]
    typer.echo(
        f"  registry rows {h['rows_in_registry']} -> used {h['rows_used']}  "
        f"(duplicate ids {len(h['duplicate_run_ids'])}, re-run seeds {h['reruns']['n_condition_seeds']}, "
        f"max re-run disagreement {h['reruns']['max_abs_macro_f1_disagreement']})"
    )
    for c in r["comparisons"]:
        if "p" in c:
            typer.echo(
                f"  {c['name']:32s} diff {c['mean_diff']:+.4f} "
                f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  wins {c['wins_b']}/{c['n']}  "
                f"p {c['p']:.4f}  BH {'yes' if c['significant_bh'] else 'no'}"
            )
    typer.echo(f"  written {out}")


@study_app.command("neutral-diagnosis")
def study_neutral_diagnosis() -> None:
    """Study A analysis: neutral error structure and a post-hoc decision-boundary test.

    CPU only; reads the outputs of `study neutral-audit`. The boundary bias is tuned on the OOF
    train predictions and evaluated on validation, so it is never scored on its own tuning data.
    """
    import pandas as pd

    from vifeedback.constants import label_names
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation.report import yaml_safe

    task = "sentiment"
    names = label_names(task)
    k, neutral = len(names), names.index("neutral")
    src = paths.RESULTS / "studies" / "study_a"
    cols = [f"p_{n}" for n in names]
    out: dict = {}

    tables = {}
    for split in ("validation", "train"):
        f = src / f"{split}_predictions.csv"
        if f.exists():
            tables[split] = pd.read_csv(f)
            out[f"{split}_neutral"] = EA.neutral_diagnosis(tables[split])

    if "train" in tables:
        tr, dv = tables["train"], tables["validation"]
        tuned = EA.tune_class_bias(tr[cols].to_numpy(), tr.gold_id.to_numpy(), neutral, k)
        before = M.evaluate(dv.gold_id, dv.pred_id, task)
        after_pred = EA.apply_class_bias(dv[cols].to_numpy(), neutral, tuned["bias"])
        after = M.evaluate(dv.gold_id, after_pred, task)
        out["boundary_test"] = {
            "tuned_on": "OOF train predictions (fold models)",
            "evaluated_on": "validation, deployed checkpoint",
            "bias_on_neutral_logprob": tuned["bias"],
            "oof_macro_f1_at_zero": tuned["at_zero"],
            "oof_macro_f1_tuned": tuned["macro_f1"],
            "dev_macro_f1_before": before["macro_f1"],
            "dev_macro_f1_after": after["macro_f1"],
            "dev_neutral_f1_before": before["per_class"]["neutral"]["f1"],
            "dev_neutral_f1_after": after["per_class"]["neutral"]["f1"],
            "dev_neutral_recall_before": before["per_class"]["neutral"]["recall"],
            "dev_neutral_recall_after": after["per_class"]["neutral"]["recall"],
            "dev_neutral_precision_before": before["per_class"]["neutral"]["precision"],
            "dev_neutral_precision_after": after["per_class"]["neutral"]["precision"],
        }
        ng = tr[tr.gold == "neutral"]
        out["train_neutral_oof"] = {
            "n": len(ng),
            "confidently_other_share": float(((~ng.correct) & (ng.p_pred >= 0.9)).mean()),
            "p_gold_below_0.1_share": float((ng.p_gold < 0.1).mean()),
        }
        if "fold_vote_agreement" in dv.columns:
            dn = dv[dv.gold == "neutral"]
            out["dev_neutral_fold_agreement"] = {
                "all_folds_agree_share": float((dn.fold_vote_agreement == 1.0).mean()),
                "all_folds_agree_and_wrong_share": float(
                    ((dn.fold_vote_agreement == 1.0) & ~dn.correct).mean()
                ),
            }

    (src / "diagnosis.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(json.dumps(yaml_safe(out), indent=2, ensure_ascii=False)[:4000])


@study_app.command("cycle1")
def study_cycle1() -> None:
    """Apply the decision rules declared in configs/experiments/cycle1.yaml to the p8 runs."""
    import pandas as pd

    from vifeedback.evaluation import decisions as D
    from vifeedback.evaluation.report import yaml_safe

    runs = D.load_runs(phase=8)
    reg = pd.read_csv(paths.REGISTRY)

    def single_task(task: str) -> dict[int, float]:
        q = reg[
            (reg.model == "phobert-base")
            & (reg.preprocessing == "seg_pyvi")
            & (reg.recipe == "base")
            & (reg.task == task)
            & (reg.split == "validation")
            # The declared controls are the P4 laptop runs (cycle1.yaml). Not "the last row for
            # the seed": a later Kaggle re-run of the same config on another GPU differs by 0.001.
            & reg.run_id.astype(str).str.startswith("p4-")
        ].drop_duplicates("seed", keep="first")
        return {int(s): float(v) for s, v in zip(q.seed, q.macro_f1, strict=True)}

    registry_ce = single_task("sentiment")
    single = {"sentiment": registry_ce, "topic": single_task("topic")}

    out = {
        "runs_found": {k: sorted(v) for k, v in runs.items()},
        "H1": D.h1(runs, registry_ce),
        "H2": D.h2(runs),
        "H3": D.h3(reg),
        "H4": D.h4(D.load_multitask(8), single),
    }
    dst = paths.RESULTS / "studies" / "cycle1"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "decisions.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(json.dumps(yaml_safe(out), indent=2, ensure_ascii=False)[:6000])


@study_app.command("latency")
def study_latency(
    checkpoint: str = typer.Option("models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp"),
    onnx_dir: str = typer.Option("models/serve/sentiment", help="a released artifact (manifest)"),
    timed: int = typer.Option(300, help="timed single-sentence calls per repeat"),
    repeats: int = typer.Option(5),
    extra_onnx: str = typer.Option(
        "", help="comma-separated extra ONNX files to time, e.g. an INT8 graph that failed its gate"
    ),
    out_file: str = typer.Option(
        "", "--out", help="output JSON; default results/studies/latency/reference_cpu.json"
    ),
) -> None:
    """Steady-state CPU latency ladder on the reference machine (review R10, § 8.4).

    Model-only timing on inputs segmented once up front, so every configuration runs the same
    workload. Configurations run in two passes, forward then reversed, so a machine that warms up
    over the session shows as a pass disagreement instead of favouring whichever ran first.
    Throughput is texts/s. Peak RSS is recorded. Never run this while training.
    """
    import os
    import subprocess
    from pathlib import Path

    import psutil
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback import env
    from vifeedback.data.loader import load
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.inference import benchmark as BM
    from vifeedback.inference.onnx_export import OnnxClassifier
    from vifeedback.preprocess.segment import get_segmenter

    torch.set_grad_enabled(False)
    try:  # nvidia-smi rather than torch.cuda.utilization(), which needs nvidia-ml-py
        util = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.split()
        gpu_busy = any(int(u) > 5 for u in util)
    except (OSError, subprocess.SubprocessError, ValueError):
        gpu_busy = False
        typer.secho("  could not read GPU utilization; make sure nothing is training", fg="yellow")
    if gpu_busy:
        raise typer.BadParameter("the GPU is busy: a latency benchmark during training is invalid")

    seg = get_segmenter("pyvi")
    texts = seg(load("test")["sentence"].tolist())  # timing inputs only; no labels are read
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).eval()
    onnx = OnnxClassifier(Path(onnx_dir))

    def torch_dynamic(batch):
        return model(
            **tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=96)
        ).logits

    def torch_padmax(batch):
        return model(
            **tok(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=96)
        ).logits

    configs = {
        "L0 torch fp32, pad to 96": torch_padmax,
        "L1 torch fp32, dynamic padding": torch_dynamic,
        f"L3 onnx fp32 ({onnx.path.name}), dynamic padding": onnx.logits,
    }
    for i, f in enumerate(p for p in extra_onnx.split(",") if p):
        fp = Path(f)
        clf = OnnxClassifier(fp.parent, model_file=fp.name)
        # Timed only: an extra graph here need not have passed its release gate (it is labelled).
        configs[f"X{i} {fp.parent.parent.name}/{fp.name} (not released)"] = clf.logits
    proc = psutil.Process(os.getpid())
    passes = []
    for order in (list(configs), list(reversed(configs))):
        res = {}
        for name in order:
            r = BM.time_callable(configs[name], texts, timed=timed, repeats=repeats)
            r["rss_mb_after"] = round(proc.memory_info().rss / 1e6, 1)
            r["texts_per_s_b32"] = BM.throughput(configs[name], texts, batch_sizes=(32,))[
                "batch32_texts_per_s"
            ]
            res[name] = r
            typer.echo(
                f"  {name:45s} p50 {r['p50_ms']:7.2f}  p95 {r['p95_ms']:7.2f} ms  "
                f"spread {r['p95_spread_across_repeats']:.1%}  {r['texts_per_s_b32']} texts/s"
            )
        passes.append(res)

    summary = BM.summarize_passes(passes, baseline="L0 torch fp32, pad to 96")
    report = {
        "protocol": "model-only, pre-segmented test inputs, 200 warmup, median-of-repeats p95, "
        "two passes in rotated order; texts/s at batch 32; RSS after each config",
        "summary": summary,
        "passes": passes,
        "peak_rss_mb": round(proc.memory_info().rss / 1e6, 1),
        "environment": env.capture(),
    }
    out = (
        Path(out_file) if out_file else paths.RESULTS / "studies" / "latency" / "reference_cpu.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(yaml_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in summary.items():
        if v["reportable"]:
            typer.echo(
                f"  {k:45s} p50 {v['p50_ms']:7.2f}  p95 {v['p95_ms']:7.2f} ms  [{v['basis']}]"
            )
        else:
            typer.secho(f"  {k:45s} NOT REPORTABLE: no steady pass", fg="red")
    typer.echo(f"  written {out}")


@study_app.command("audit-sheet")
def study_audit_sheet() -> None:
    """Rebuild the local neutral-audit sheet from committed, text-free files. No GPU.

    The sheet holds corpus text, so it lives in the gitignored local/ folder and is lost with it.
    Everything needed to rebuild it is committed: which rows were sampled (audit_sample_index.csv)
    and their predictions (split prediction files, keyed by row). Text is re-joined from data/raw.
    """
    import pandas as pd

    from vifeedback.data.loader import load
    from vifeedback.evaluation import error_analysis as EA

    src = paths.RESULTS / "studies" / "study_a"
    idx = pd.read_csv(src / "audit_sample_index.csv")
    parts = []
    for split, g in idx.groupby("split", sort=False):
        preds = pd.read_csv(src / f"{split}_predictions.csv").set_index("row")
        text = load(split)["sentence"]
        rows = preds.loc[g.example_index].copy()
        rows["text"] = text.iloc[g.example_index.to_numpy()].to_numpy()
        rows["stratum"] = g.stratum.to_numpy()
        rows.index.name = "example_index"
        parts.append(rows.reset_index())
    sample = pd.concat(parts, ignore_index=True)
    assert (sample.gold.to_numpy() == idx.gold.to_numpy()).all(), "row alignment check failed"
    sheet = EA.export_annotation_sheet(sample, src / "local" / "audit_sheet.csv")
    typer.echo(f"  rebuilt {len(sample)} rows -> {sheet}")


@study_app.command("audit-report")
def study_audit_report(
    sheet: str = typer.Option(
        "results/studies/study_a/local/audit_sheet.csv", help="the filled annotation sheet"
    ),
    second: str = typer.Option("", help="a second pass over >= 50 of the same rows, for kappa"),
    kind: str = typer.Option(
        "intra", help="inter (two people) | intra (one person, >= 24 h apart)"
    ),
) -> None:
    """Analyse the filled neutral-audit sheet and apply the decision tree frozen in cycle2.yaml.

    Refuses an incomplete sheet. Writes counts only (no text) to study_a/audit_report.json.
    """
    import pandas as pd

    from vifeedback.evaluation import audit as AU
    from vifeedback.evaluation.report import yaml_safe

    first = pd.read_csv(sheet, encoding="utf-8-sig", keep_default_na=False)
    other = pd.read_csv(second, encoding="utf-8-sig", keep_default_na=False) if second else None
    try:
        r = AU.report(first, other, kind)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    out = paths.RESULTS / "studies" / "study_a" / "audit_report.json"
    out.write_text(json.dumps(yaml_safe(r), indent=2), encoding="utf-8")
    rs = r["random_stratum"]
    typer.echo(
        f"  random stratum (n={rs['n']}): incorrect gold {rs['incorrect']['rate']:.2f} "
        f"{tuple(round(x, 2) for x in rs['incorrect']['wilson_95'])}, "
        f"ambiguous {rs['ambiguous']['rate']:.2f}"
    )
    if "agreement" in r:
        a = r["agreement"]
        typer.echo(
            f"  {a['kind']}-annotator kappa {a['kappa']:.2f} on {a['n']} rows {a['warnings']}"
        )
    d = r["decision"]
    typer.echo(
        f"  scope n={d['n']}: incorrect {d['share_incorrect']:.2f}, ambiguous "
        f"{d['share_ambiguous']:.2f} -> {d['branch']}: {d['next']}"
    )
    typer.echo(f"  -> {out}")


@study_app.command("closing-gate")
def study_closing_gate(
    ce_checkpoint: str = typer.Option("models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp"),
    aug_glob: str = typer.Option("p9-sent-phobert-base-seg_pyvi-aug-diac-teen-s42-*-ckp"),
    reason: str = typer.Option("Cycle 1 closing gate (cycle1.yaml): finalists on test, once"),
) -> None:
    """Cycle 1 closing gate: the finalists on test, once, with everything frozen beforehand.

    Per checkpoint (deployed CE model; H2-augmented model, seed 42): test metrics on the official split
    and on the overlap-excluded slice (configs/data/eval_slices_v1.json), calibration with the
    temperature fitted on validation and applied to test, and the robustness suites on test. Plus the
    5-seed paired test comparison from the registry. Every test touch is logged.
    """
    from datetime import UTC, datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from vifeedback.data.loader import load
    from vifeedback.evaluation import calibration as C
    from vifeedback.evaluation import decisions as D
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation import robustness as R
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.preprocess.segment import get_segmenter
    from vifeedback.preprocess.variants import load_variant

    aug = sorted(paths.MODELS.glob(aug_glob))
    if not aug:
        raise typer.BadParameter(f"no augmented checkpoint matching {aug_glob} under models/")
    checkpoints = {"ce_deployed": Path(ce_checkpoint), "h2_augmented": aug[-1]}

    seg = get_segmenter("pyvi")
    te, dv = load_variant("seg_pyvi", "test"), load_variant("seg_pyvi", "validation")
    y_te, y_dv = te.sentiment.to_numpy(), dv.sentiment.to_numpy()
    raw_te = load("test")["sentence"].tolist()
    slices = json.loads(
        (paths.CONFIGS / "data" / "eval_slices_v1.json").read_text(encoding="utf-8")
    )
    keep = np.setdiff1d(np.arange(len(y_te)), slices["test"]["overlapping_train_indices"])

    out: dict = {"reason": reason, "slice_version": slices["version"], "checkpoints": {}}
    for name, ck in checkpoints.items():
        p_te = EA.predict_proba(ck, te.sentence.tolist())
        p_dv = EA.predict_proba(ck, dv.sentence.tolist())
        t = C.fit_temperature(C.probs_to_logits(p_dv), y_dv)  # fitted on validation only
        cal_te = C.apply_temperature(C.probs_to_logits(p_te), t)
        full = M.evaluate(y_te, p_te.argmax(1), "sentiment", y_prob=p_te)
        sl = M.evaluate(y_te[keep], p_te[keep].argmax(1), "sentiment")
        rob = {}
        for suite in ("nodiacritic", "nodiacritic-50", "teencode-100", "charnoise-5"):
            pert, _ = R.perturb(raw_te, suite, seed=42)
            pred = EA.predict_proba(ck, seg(pert)).argmax(1)
            rob[suite] = {
                "macro_f1": M.macro_f1(y_te, pred, 3),
                "pred_share_neutral": float((pred == 1).mean()),
            }
        out["checkpoints"][name] = {
            "checkpoint": str(ck),
            "test": {
                "macro_f1": full["macro_f1"],
                "weighted_f1": full["weighted_f1"],
                "per_class_f1": {c: v["f1"] for c, v in full["per_class"].items()},
            },
            "test_excluding_train_overlap": {
                "n": len(keep),
                "macro_f1": sl["macro_f1"],
                "neutral_f1": sl["per_class"]["neutral"]["f1"],
            },
            "calibration": {
                "temperature_fit_on_validation": t,
                "uncalibrated": {
                    k: v
                    for k, v in C.summary(p_te, y_te).items()
                    if k in ("nll", "brier", "ece_equal_width", "ece_equal_mass")
                },
                "calibrated": {
                    k: v
                    for k, v in C.summary(cal_te, y_te).items()
                    if k in ("nll", "brier", "ece_equal_width", "ece_equal_mass")
                },
            },
            "robustness_test": rob,
        }
        with open(paths.TEST_EVAL_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}\t{ck.name}\tP9\t{reason}\n")
        typer.echo(
            f"  {name:13s} test macro-F1 {full['macro_f1']:.4f}  (excl. overlap {sl['macro_f1']:.4f})  "
            f"T={t:.2f}  no-diacritic {rob['nodiacritic']['macro_f1']:.3f}"
        )

    reg = pd.read_csv(paths.REGISTRY)
    tr = reg[reg.split == "test"]

    def by_seed(q):
        return {int(s): float(m) for s, m in zip(q.seed, q.macro_f1, strict=True)}

    ce5 = by_seed(
        tr[tr.run_id.astype(str).str.match(r"p4-sent-phobert-base-seg_pyvi-base-s\d+-tes")]
    )
    aug5 = by_seed(
        tr[tr.run_id.astype(str).str.startswith("p9-sent-phobert-base-seg_pyvi-aug-diac-teen")]
    )
    out["five_seed_test"] = {
        "ce": ce5,
        "augmented": aug5,
        "paired_aug_minus_ce": D.paired(ce5, aug5),
    }
    dst = paths.RESULTS / "studies" / "closing_gate"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    p5 = out["five_seed_test"]["paired_aug_minus_ce"]
    if p5.get("n"):
        typer.echo(
            f"  5-seed test, augmented - CE: {p5['mean_delta']:+.4f} {p5.get('ci95')} "
            f"wins {p5['wins']}/{p5['n']}"
        )
    typer.echo(f"  written {dst / 'summary.json'}")


def _print_stacking(result: dict) -> None:
    for name in ("dense", "sparse", "stacked"):
        pc = result[name]["per_class_f1"]
        typer.echo(
            f"  {name:8s} macro-F1 {result[name]['macro_f1']:.4f}  "
            f"facility {pc['facility']:.3f}  others {pc['others']:.3f}"
        )
    pb = result["paired_stacked_minus_dense"]
    typer.echo(
        f"  stacked - dense: {pb['observed_diff']:+.4f} [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}]"
        f"  supported={result['supported']}"
    )
    fc = result.get("descriptive_facility_sparse_minus_dense")
    if fc:
        typer.echo(
            f"  facility F1, sparse - dense (descriptive): {fc['observed_diff']:+.4f} "
            f"[{fc['ci_low']:+.4f}, {fc['ci_high']:+.4f}]"
        )


@study_app.command("topic-stacking")
def study_topic_stacking(
    seed: int = typer.Option(42),
    k: int = typer.Option(5),
    reuse: bool = typer.Option(
        False, help="re-score from the saved validation features (no training, no refit)"
    ),
) -> None:
    """Cycle 2 H5: out-of-fold PhoBERT + TF-IDF stacking for topic (5 fold fine-tunes, ~20 GPU-min)."""
    import numpy as np
    import pandas as pd

    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import stacking as ST
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.preprocess.variants import load_variant
    from vifeedback.training.trainer import TrainConfig

    task = "topic"
    out = paths.RESULTS / "studies" / "topic_stacking"
    if reuse:
        v = pd.read_csv(out / "validation_features.csv")
        prev = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        cols_of = lambda p: v.filter(regex=rf"^{p}_\d+$").to_numpy()  # noqa: E731
        result = ST.compare(
            v.gold.to_numpy(),
            {
                "dense": cols_of("dense").argmax(1),
                "sparse": cols_of("sparse").argmax(1),
                "stacked": v.stacked_pred.to_numpy(),
            },
            task,
        )
        result["folds"], result["gpu_seconds"] = prev["folds"], prev["gpu_seconds"]
        (out / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
        _print_stacking(result)
        return

    tr, dv = load_variant("seg_pyvi", "train"), load_variant("seg_pyvi", "validation")
    y_tr, y_dv = tr[task].to_numpy(), dv[task].to_numpy()
    cfg = TrainConfig(task=task, model_key="phobert-base", preprocessing="seg_pyvi", seed=seed)

    typer.echo(f"dense OOF: {k} folds of phobert-base / topic")
    r = EA.oof_predictions(
        cfg, tr.sentence.tolist(), y_tr, dv.sentence.tolist(), y_dv, k=k, fold_seed=seed
    )
    dense_oof, fold_id = r["oof_probs"], r["fold_id"]
    dense_dv = r["dev_probs"].mean(axis=0)

    typer.echo("sparse OOF: TF-IDF B4, same folds, C fixed")
    sparse_oof = ST.sparse_oof_scores(tr.sentence_raw.tolist(), y_tr, fold_id, seed)
    sparse_full = ST.sparse_model(seed).fit(np.asarray(tr.sentence_raw, dtype=object), y_tr)
    sparse_dv = sparse_full.decision_function(np.asarray(dv.sentence_raw, dtype=object))

    meta = ST.fit_meta(ST.features(dense_oof, sparse_oof), y_tr, seed)
    stacked_dv = meta.predict(ST.features(dense_dv, sparse_dv))
    result = ST.compare(
        y_dv,
        {"dense": dense_dv.argmax(1), "sparse": sparse_dv.argmax(1), "stacked": stacked_dv},
        task,
    )
    result["folds"] = r["folds"]
    result["gpu_seconds"] = r["total_seconds"]

    out.mkdir(parents=True, exist_ok=True)
    cols = lambda p, a: {f"{p}_{i}": a[:, i] for i in range(a.shape[1])}  # noqa: E731
    pd.DataFrame(
        {"gold": y_tr, "fold": fold_id, **cols("dense", dense_oof), **cols("sparse", sparse_oof)}
    ).to_csv(out / "train_oof_features.csv", index_label="row", float_format="%.6g")
    pd.DataFrame(
        {
            "gold": y_dv,
            **cols("dense", dense_dv),
            **cols("sparse", sparse_dv),
            "stacked_pred": stacked_dv,
        }
    ).to_csv(out / "validation_features.csv", index_label="row", float_format="%.6g")
    (out / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
    _print_stacking(result)


results_app = typer.Typer(pretty_exceptions_show_locals=False, help="Results housekeeping")
app.add_typer(results_app, name="results")


@study_app.command("challenge")
def study_challenge(
    ce: str = typer.Option(
        "models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp", help="seed-42 CE checkpoint (served)"
    ),
    aug: str = typer.Option(
        "models/p9-sent-phobert-base-seg_pyvi-aug-diac-teen-s42-599cf21f-ckp",
        help="seed-42 H2-augmented checkpoint",
    ),
) -> None:
    """Cycle 2 H6 on the frozen challenge set: per-category results and the declared serving rule."""
    from vifeedback.constants import label_names
    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation.report import yaml_safe

    df = CH.load()  # raises if the file differs from the hash frozen in cycle2.yaml
    x = CH.pipeline("seg_pyvi")(df.text.tolist())
    probs = {name: EA.predict_proba(ckp, x) for name, ckp in (("ce", ce), ("augmented", aug))}
    pred = {name: p.argmax(1) for name, p in probs.items()}

    result: dict = {
        "challenge_sha256": CH.declared_sha256(),
        "checkpoints": {"ce": ce, "augmented": aug},
        "h6": CH.h6(df, pred["ce"], pred["augmented"]),
    }
    for name in probs:
        result[name] = {
            **CH.category_report(df, pred[name]),
            "negation_pairs": CH.negation_pairs(df, pred[name]),
            "out_of_scope_confidence": CH.out_of_scope_confidence(df, probs[name]),
        }

    out = paths.RESULTS / "studies" / "challenge"
    out.mkdir(parents=True, exist_ok=True)
    names = label_names("sentiment")
    table = df[["id", "category", "pair_id", "text", "sentiment"]].copy()
    for name, p in probs.items():
        table[f"{name}_pred"] = [names[i] for i in pred[name]]
        table[f"{name}_conf"] = p.max(1).round(4)
    table.to_csv(out / "predictions.csv", index=False)  # constructed text only; no corpus rows
    (out / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")

    cats = sorted(result["ce"]["by_category"])
    typer.echo(f"{'category':22s} {'n':>4s} {'CE':>6s} {'aug':>6s}")
    for c in cats:
        a, b = result["ce"]["by_category"][c], result["augmented"]["by_category"][c]
        typer.echo(f"{c:22s} {a['n']:4d} {a['accuracy']:6.3f} {b['accuracy']:6.3f}")
    for name in probs:
        r = result[name]
        typer.echo(
            f"{name:10s} macro-F1 {r['macro_f1']:.4f}  neutral F1 {r['per_class_f1']['neutral']:.3f}  "
            f"negation pairs both-correct {r['negation_pairs']['both_correct']:.2f}"
        )
    h = result["h6"]
    pb = h["typed_paired_aug_minus_ce"]
    typer.echo(
        f"H6 typed rows ({h['typed_rows']}): aug - CE {pb['observed_diff']:+.3f} "
        f"[{pb['ci_low']:+.3f}, {pb['ci_high']:+.3f}]; other rows {h['other_diff']:+.3f} "
        f"-> {h['decision']}"
    )


@study_app.command("llm-prompt-dev")
def study_llm_prompt_dev(
    model: str = typer.Option("Qwen/Qwen3-1.7B", help="the pilot; never an API model"),
    force: bool = typer.Option(False, help="overwrite a frozen prompt_dev.json"),
) -> None:
    """Cycle 2 H7 prompt development: score the prompt variants on a fixed train subset, freeze the best."""
    import datetime as dt

    from vifeedback.data.loader import load
    from vifeedback.evaluation import llm_reference as L
    from vifeedback.evaluation import metrics as M

    if L.PROMPT_DEV_FILE.exists() and not force:
        raise typer.BadParameter(
            f"{L.PROMPT_DEV_FILE} exists: the prompt is frozen (--force to redo)"
        )
    tr = load("train")
    y = tr.sentiment.to_numpy()
    idx = L.prompt_dev_subset(y)
    texts = tr.sentence.iloc[idx].tolist()
    scorer = L.HFScorer(model)
    rows = {}
    for variant in L.VARIANTS:
        r = L.run(scorer, texts, variant)
        ev = M.evaluate(y[idx], r["probs"].argmax(1), "sentiment")
        rows[variant] = {
            "macro_f1": ev["macro_f1"],
            "per_class_f1": {c: v["f1"] for c, v in ev["per_class"].items()},
            "seconds": r["seconds"],
        }
        typer.echo(
            f"  {variant:30s} macro-F1 {ev['macro_f1']:.4f}  "
            f"neutral F1 {rows[variant]['per_class_f1']['neutral']:.3f}"
        )
    best = max(L.VARIANTS, key=lambda v: rows[v]["macro_f1"])  # ties keep the earlier, simpler one
    per_class = {L.LABELS[c]: int((y[idx] == c).sum()) for c in range(len(L.LABELS))}
    L.write(
        L.PROMPT_DEV_FILE,
        {
            "model": model,
            "revision": scorer.revision,
            "subset": {"split": "train", "indices": idx.tolist(), "per_class": per_class},
            "variants": rows,
            "frozen_variant": best,
            "frozen_on": dt.date.today().isoformat(),
            "prompts": {v: "\n\n".join(parts) for v, parts in L.VARIANTS.items()},
        },
    )
    typer.echo(f"frozen: {best} -> {L.PROMPT_DEV_FILE}")


@study_app.command("llm-reference")
def study_llm_reference(
    model: str = typer.Option("Qwen/Qwen3-1.7B"),
    backend: str = typer.Option("hf", help="hf | openai"),
    data: str = typer.Option("challenge", help="challenge | validation"),
    shots: int = typer.Option(0, help="0, or 6 demonstrations (2 per class)"),
    demo_seed: int = typer.Option(1, help="demonstration draw (declared: 1 and 2)"),
    batch_size: int = typer.Option(8),
    licence_confirmed: bool = typer.Option(
        False, help="owner confirmed UIT-VSFC may be sent to the API (cycle2.yaml data_egress)"
    ),
) -> None:
    """Cycle 2 H7: score one LLM configuration with the frozen prompt; compare with the encoders."""
    import numpy as np
    import pandas as pd

    from vifeedback.data.loader import load
    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import llm_reference as L
    from vifeedback.evaluation import metrics as M

    if backend == "openai" and (data != "challenge" or shots) and not licence_confirmed:
        raise typer.BadParameter(
            "UIT-VSFC text (validation, or train demonstrations) goes to the API only after the "
            "owner confirms the licence allows it; pass --licence-confirmed once that is settled"
        )
    variant = L.frozen_variant()
    frozen = json.loads(L.PROMPT_DEV_FILE.read_text(encoding="utf-8"))
    tr = load("train")
    tr_texts, tr_y = tr.sentence.tolist(), tr.sentiment.to_numpy()
    demo_idx = (
        L.draw_demo_indices(
            tr_texts, tr_y, demo_seed, shots // 3, exclude=frozen["subset"]["indices"]
        )
        if shots
        else []
    )
    demos = [(tr_texts[i], L.LABELS[int(tr_y[i])]) for i in demo_idx]

    if data == "challenge":
        df = CH.load()
        texts = df.text.tolist()
        scored = df.scored.to_numpy()
        y = df.y.to_numpy()
        enc = pd.read_csv(
            paths.RESULTS / "studies" / "challenge" / "predictions.csv", keep_default_na=False
        )
        enc_pred = {n: enc[f"{n}_pred"].map(L.LABELS.index).to_numpy() for n in ("ce", "augmented")}
    else:
        from vifeedback.evaluation import error_analysis as EA
        from vifeedback.preprocess.variants import load_variant

        dv = load("validation")
        texts = dv.sentence.tolist()
        y = dv.sentiment.to_numpy().astype(float)
        scored = np.ones(len(y), dtype=bool)
        # Computed once from the local checkpoints and committed (labels only, no text), so a
        # Kaggle session without the checkpoints compares against the same predictions.
        cache = L.OUT / "encoder_validation_preds.csv"
        if not cache.exists():
            x = load_variant("seg_pyvi", "validation").sentence.tolist()
            ckp = {
                "ce": "models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp",
                "augmented": "models/p9-sent-phobert-base-seg_pyvi-aug-diac-teen-s42-599cf21f-ckp",
            }
            pd.DataFrame(
                {f"{n}_pred": EA.predict_proba(p, x).argmax(1) for n, p in ckp.items()}
            ).to_csv(cache, index_label="row")
        cached = pd.read_csv(cache)
        enc_pred = {n: cached[f"{n}_pred"].to_numpy() for n in ("ce", "augmented")}

    import gc

    import torch

    gc.collect()  # the encoders' weights must leave the 4 GB GPU before the LLM arrives
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scorer = L.OpenAIScorer(model) if backend == "openai" else L.HFScorer(model)
    r = L.run(scorer, texts, variant, demos, batch_size=batch_size)
    pred = r["probs"].argmax(1)
    ys, ps = y[scored].astype(int), pred[scored]
    ev = M.evaluate(ys, ps, "sentiment", y_prob=r["probs"][scored])
    summary: dict = {
        "model": model,
        "revision": scorer.revision,
        "backend": backend,
        "data": data,
        "variant": variant,
        "shots": shots,
        "demo_seed": demo_seed if shots else None,
        # Train row indices and labels only: corpus text is never written to a result file.
        "demos": [
            {"train_index": i, "label": lab} for i, (_, lab) in zip(demo_idx, demos, strict=True)
        ],
        "n_scored": int(scored.sum()),
        "macro_f1": ev["macro_f1"],
        "per_class": ev["per_class"],
        "seconds": r["seconds"],
        "seconds_per_1k": r["seconds_per_1k"],
        "vs_encoder": {
            name: L.compare_to_encoder(ys, ps, p[scored]) for name, p in enc_pred.items()
        },
    }
    if isinstance(scorer, L.OpenAIScorer):
        summary["api"] = {
            **scorer.usage,
            "served_models": sorted(scorer.served_models),
            "price_per_m_tokens_usd": L.API_PRICE_PER_M,
            "cost_usd": scorer.cost_usd(),
            "usd_per_1k": 1000 * scorer.cost_usd() / len(texts),
            "note": "latency includes the network round trip",
        }
    else:
        summary["gpu_seconds_per_1k"] = r["seconds_per_1k"]
        summary["batch_size"] = batch_size
    if data == "challenge":
        summary["challenge"] = {
            **CH.category_report(df, pred),
            "negation_pairs": CH.negation_pairs(df, pred),
            "out_of_scope_confidence": CH.out_of_scope_confidence(df, r["probs"]),
        }

    run_name = f"{data}-{variant}-k{shots}" + (f"-s{demo_seed}" if shots else "")
    out = L.OUT / L.slug(model) / run_name
    L.write(out / "summary.json", summary)
    table = pd.DataFrame(r["probs"].round(5), columns=[f"p_{c}" for c in L.LABELS])
    table.insert(0, "pred", [L.LABELS[i] for i in pred])
    if data == "challenge":  # constructed text ids only; corpus text is never written here
        table.insert(0, "id", df.id)
    table.to_csv(out / "predictions.csv", index_label="row")

    pc = ev["per_class"]
    typer.echo(
        f"{model} {run_name}: macro-F1 {ev['macro_f1']:.4f}  neutral F1 {pc['neutral']['f1']:.3f}"
        f"  ({r['seconds_per_1k']:.1f} s per 1k)"
    )
    for name, c in summary["vs_encoder"].items():
        n = c["neutral_f1"]
        typer.echo(
            f"  vs {name:9s} neutral F1 {n['observed_diff']:+.3f} [{n['ci_low']:+.3f}, "
            f"{n['ci_high']:+.3f}] p={n['p_value']:.3f}; macro-F1 {c['macro_f1']['observed_diff']:+.3f}"
        )
    if isinstance(scorer, L.OpenAIScorer):
        typer.echo(f"  API: {scorer.usage['calls']} calls, USD {scorer.cost_usd():.4f}")


@results_app.command("merge")
def results_merge(
    source: str = typer.Argument(..., help="an extracted results folder, e.g. kaggle_results/"),
    dry_run: bool = typer.Option(False, help="report what would be merged, change nothing"),
) -> None:
    """Merge another machine's results into results/: append-only, never overwriting.

    Replaces a manual step that once cost a morning of runs: the Kaggle output was extracted over
    the repository and results/, holding uncommitted local runs, was lost. This command refuses to
    run without a local results/, migrates the registry schema first, appends only run ids not yet
    present, and copies only run directories that do not exist locally.
    """
    import csv
    import shutil
    from pathlib import Path

    import pandas as pd

    from vifeedback.evaluation import report as R

    src = Path(source)
    if not paths.REGISTRY.exists():
        raise typer.BadParameter(
            f"{paths.REGISTRY} is missing. Restore results/ first (git checkout -- results/) instead of "
            "replacing it with another machine's copy; that copy lacks this machine's runs."
        )
    if not (src / "registry.csv").exists():
        raise typer.BadParameter(
            f"{src} has no registry.csv; point at the extracted results folder"
        )

    local = pd.read_csv(paths.REGISTRY)
    other = pd.read_csv(src / "registry.csv")
    new = other[~other.run_id.isin(local.run_id)]
    run_dirs = [
        d
        for d in sorted((src / "runs").glob("*"))
        if d.is_dir() and not (paths.RUNS / d.name).exists()
    ]
    typer.echo(f"  registry: {len(new)} new rows ({len(other) - len(new)} already present)")
    typer.echo(f"  run directories: {len(run_dirs)} new")
    if dry_run:
        for rid in new.run_id:
            typer.echo(f"    + {rid}")
        return

    R._migrate_registry_schema()
    with open(paths.REGISTRY, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=R.REGISTRY_FIELDS)
        for _, row in new.iterrows():
            w.writerow(
                {k: ("" if k not in row or pd.isna(row[k]) else row[k]) for k in R.REGISTRY_FIELDS}
            )
    for d in run_dirs:
        shutil.copytree(d, paths.RUNS / d.name)
    manifests = sorted((src / "studies" / "export").glob("*_manifest.json"))
    if manifests:
        dst = paths.RESULTS / "studies" / "export"
        dst.mkdir(parents=True, exist_ok=True)
        for m in manifests:
            if not (dst / m.name).exists():
                shutil.copy(m, dst / m.name)
    # H7 runs (Kaggle Qwen3-4B): one folder per model and configuration; never overwrite one.
    for run in sorted((src / "studies" / "llm_reference").glob("*/*")):
        dst = paths.RESULTS / "studies" / "llm_reference" / run.parent.name / run.name
        if run.is_dir() and not dst.exists():
            shutil.copytree(run, dst)
            typer.echo(f"  llm_reference: {run.parent.name}/{run.name}")
    typer.echo(
        f"  merged. Now: vifeedback study tables   (registry has {len(pd.read_csv(paths.REGISTRY))} rows)"
    )


@results_app.command("provenance")
def results_provenance() -> None:
    """Resolve every run's recorded source hash to the commit whose code it executed (R11).

    Kaggle runs carry no git SHA; their `source_sha256` is matched against every commit's source.
    Writes results/provenance.json.
    """
    from collections import defaultdict

    from vifeedback.env import source_hash_by_commit

    by_commit = source_hash_by_commit(paths.ROOT)
    runs: dict[str, list[str]] = defaultdict(list)
    for env_file in sorted((paths.RESULTS / "runs").glob("*/env.json")):
        h = json.loads(env_file.read_text(encoding="utf-8")).get("source_sha256") or "none"
        runs[h].append(env_file.parent.name)
    out: dict[str, dict] = {
        h: {
            "first_commit": by_commit[h][0] if h in by_commit else None,
            "commits_with_identical_source": len(by_commit.get(h, [])),
            "note": None if h in by_commit else "no commit has this source: an uncommitted tree",
            "runs": names,
        }
        for h, names in sorted(runs.items())
    }
    (paths.RESULTS / "provenance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for h, r in out.items():
        where = (r["first_commit"] or "uncommitted")[:10]
        typer.echo(f"  {h[:12]}  {len(r['runs']):3d} runs  -> {where}")


if __name__ == "__main__":
    app()
