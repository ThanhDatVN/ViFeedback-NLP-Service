"""`vifeedback serve`: release the ONNX artifact, attach its parts, publish, reproduce, benchmark, run."""

from __future__ import annotations

import json

import typer

from vifeedback import paths
from vifeedback.cli._apps import serve_app


@serve_app.command("export")
def serve_export(
    checkpoint: str = typer.Option(..., help="path to a saved HF checkpoint"),
    task: str = typer.Option("sentiment"),
    preprocessing: str = typer.Option(
        "", help="the checkpoint's training preprocessing; inferred from its name if omitted"
    ),
    max_length: int = typer.Option(96),
    quantize: str = typer.Option("dynamic", help="none | dynamic | static | careful (S5')"),
    acceptance_set: str = typer.Option(
        "validation",
        help="validation (UIT-VSFC) | pooled: UIT-VSFC + NEU-ESC validation (S5', cycle4.yaml v4)",
    ),
    out: str = typer.Option("", help="defaults to models/serve/<task>"),
    prebuilt: str = typer.Option(
        "",
        help="release an already built graph (e.g. the INT8 file S5' accepted) instead of exporting",
    ),
    calib_size: int = typer.Option(300, help="stratified train sentences for static INT8"),
    with_features: bool = typer.Option(
        False, help="also output the sentence feature (for the out-of-scope score, ADR-031)"
    ),
) -> None:
    """Export, verify against the quality contract, and release (review R3).

    Built in a staging directory; the served directory is replaced only if FP32 logits match, or
    INT8 stays within 0.005 macro-F1 of PyTorch on the full validation set.
    """
    from pathlib import Path

    import numpy as np

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
    if acceptance_set == "pooled":
        # S5': UIT-VSFC validation as segmented, plus NEU-ESC validation through the serving
        # transform (lowercase, restorer, pyvi); both are already model input.
        from vifeedback.training.domain import neu_esc, serving_transform

        neu = neu_esc("validation", serving_transform(), in_scope=False)
        accept_x = load_variant(preprocessing, "validation").sentence.tolist() + neu["x"]
        accept_y = np.concatenate([y_dv, neu["y"]])
        accept_pipeline = None
    elif acceptance_set == "validation":
        accept_x, accept_y, accept_pipeline = raw_dv, y_dv, pipeline
    else:
        raise typer.BadParameter("acceptance-set must be validation or pooled")

    manifest = release(
        checkpoint=checkpoint,
        task=task,
        preprocessing=preprocessing,
        quantize=quantize,
        out_dir=Path(out) if out else paths.MODELS / "serve" / task,
        pipeline=pipeline,
        calib_raw=[raw_tr[i] for i in calib_idx],
        accept_raw=accept_x,
        accept_y=accept_y,
        max_length=max_length,
        log=typer.echo,
        with_features=with_features,
        accept_pipeline=accept_pipeline,
        acceptance_set=acceptance_set,
        prebuilt=Path(prebuilt) if prebuilt else None,
    )
    typer.echo(f"  manifest: {manifest['model_file']}  sha256 {manifest['sha256'][:12]}...")


@serve_app.command("add-restorer")
def serve_add_restorer(task: str = typer.Option("sentiment")) -> None:
    """Attach the diacritic restorer to the released artifact (cycle3.yaml v5 S2b passed, ADR-031).

    Built from UIT-VSFC train only. Accepted only if every validation label of the served ONNX graph
    is unchanged with the restorer in front of it; the file's SHA-256 goes into the manifest, and the
    service refuses a restorer that does not match.
    """
    import hashlib
    import shutil

    import numpy as np

    from vifeedback.data.loader import load
    from vifeedback.evaluation import metrics as M
    from vifeedback.inference.onnx_export import OnnxClassifier
    from vifeedback.preprocess.diacritics import Restorer
    from vifeedback.preprocess.normalize import model_text, strip_diacritics
    from vifeedback.preprocess.segment import get_segmenter

    d = paths.MODELS / "serve" / task
    mpath = d / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    clf = OnnxClassifier(d, max_length=manifest.get("max_length", 96))
    segment = get_segmenter("pyvi")
    restorer = Restorer.fit(load("train").sentence.tolist())

    def labels(texts: list[str], with_restorer: bool) -> np.ndarray:
        prep = [restorer(model_text(t)) if with_restorer else model_text(t) for t in texts]
        x = segment(prep)
        return np.concatenate([clf.logits(x[i : i + 64]) for i in range(0, len(x), 64)]).argmax(1)

    dv = load("validation")
    y = dv.sentiment.to_numpy()
    clean = dv.sentence.tolist()
    stripped = [strip_diacritics(t) for t in clean]
    base, with_r = labels(clean, False), labels(clean, True)
    changed = int((base != with_r).sum())
    acceptance = {
        "validation_labels_changed": changed,
        "validation_macro_f1": M.macro_f1(y, with_r, 3),
        "stripped_validation_macro_f1_without": M.macro_f1(y, labels(stripped, False), 3),
        "stripped_validation_macro_f1_with": M.macro_f1(y, labels(stripped, True), 3),
        "passed": changed == 0,
    }
    typer.echo(f"  acceptance: {acceptance}")
    if not acceptance["passed"]:
        raise typer.Exit(1)

    out = d / "restorer.json"
    restorer.save(out)
    shutil.copy2(mpath, d / "manifest.before-restorer.json")
    manifest["restorer"] = {
        "file": out.name,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "threshold": restorer.threshold,
        "fitted_on": "UIT-VSFC train (syllable forms and word bigrams)",
        "decided_by": "cycle3.yaml v5 S2b (NEU-ESC confirmation), ADR-031",
        "acceptance": acceptance,
    }
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    record = paths.RESULTS / "studies" / "export" / "laptop_fp32_augmented_restorer_manifest.json"
    record.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"  restorer attached: {out.name} sha256={manifest['restorer']['sha256'][:12]}")


@serve_app.command("add-ood")
def serve_add_ood(task: str = typer.Option("sentiment")) -> None:
    """Attach the out-of-scope score to a release exported with features (cycle3.yaml v5 S3, ADR-031).

    Mahalanobis parameters from UIT-VSFC train features of the served graph; threshold keeps 95% of
    validation. Accepted only if the declared S3 check holds on the served graph itself: NEU-ESC
    off-topic posts vs validation, AUROC >= 0.90.
    """
    import hashlib
    import shutil

    import numpy as np

    from vifeedback.data.loader import load
    from vifeedback.evaluation import external as X
    from vifeedback.evaluation import ood as OOD
    from vifeedback.inference.onnx_export import OnnxClassifier
    from vifeedback.preprocess.diacritics import Restorer
    from vifeedback.preprocess.normalize import model_text
    from vifeedback.preprocess.segment import get_segmenter

    d = paths.MODELS / "serve" / task
    mpath = d / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    clf = OnnxClassifier(d, max_length=manifest.get("max_length", 96))
    if not clf.has_features:
        raise typer.BadParameter(
            "the served graph has no 'features' output: serve export --with-features"
        )
    restorer = Restorer.load(d / manifest["restorer"]["file"]) if manifest.get("restorer") else None
    segment = get_segmenter("pyvi")

    def encode(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        prep = [model_text(t) for t in texts]
        if restorer is not None:
            prep = [restorer(t) for t in prep]
        x = segment(prep)
        parts = [clf.logits_and_features(x[i : i + 64]) for i in range(0, len(x), 64)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    tr = load("train")
    _, f_tr = encode(tr.sentence.tolist())
    maha = OOD.fit_mahalanobis(f_tr, tr.sentiment.to_numpy())
    l_dv, f_dv = encode(load("validation").sentence.tolist())
    s_dv = OOD.scores(l_dv, f_dv, maha)["neg_mahalanobis"]
    threshold = float(np.quantile(s_dv, 0.05))

    ne = X.load_neu_esc("test")
    off = ne[ne.topic.isin(("Spam", "News", "Jobs & Recruitment", "Club & Events"))].text.tolist()
    l_o, f_o = encode(off)
    check = OOD.evaluate(s_dv, OOD.scores(l_o, f_o, maha)["neg_mahalanobis"])
    acceptance = {**check, "off_topic_rows": len(off), "passed": check["auroc"] >= 0.90}
    typer.echo(f"  acceptance: {acceptance}")
    if not acceptance["passed"]:
        raise typer.Exit(1)

    out = d / "ood.npz"
    np.savez(
        out,
        means=maha["means"].astype(np.float32),
        precision=maha["precision"].astype(np.float32),
        threshold=np.float32(threshold),
    )
    shutil.copy2(mpath, d / "manifest.before-ood.json")
    manifest["ood"] = {
        "file": out.name,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "method": "negative Mahalanobis distance to the nearest class mean, shared covariance (train)",
        "threshold": threshold,
        "keeps_validation": 0.95,
        "decided_by": "cycle3.yaml v5 S3 (NEU-ESC confirmation), ADR-031",
        "acceptance": acceptance,
    }
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    record = (
        paths.RESULTS / "studies" / "export" / "laptop_fp32_augmented_restorer_ood_manifest.json"
    )
    record.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"  out-of-scope score attached: {out.name} sha256={manifest['ood']['sha256'][:12]}")


@serve_app.command("add-scope")
def serve_add_scope(task: str = typer.Option("sentiment")) -> None:
    """Replace the Mahalanobis score behind `in_scope` with B4's topic-aware detector (ADR-034).

    Exports the fitted TF-IDF logistic regression to scope.npz (no pickle at serving time), checks
    the numpy scorer equal to scikit-learn on every B4' evaluation text, and records the file's
    SHA-256, the threshold and B4's acceptance in the manifest. The Mahalanobis entry is removed
    (kept in manifest.before-scope.json).
    """
    import hashlib
    import pickle
    import shutil

    import numpy as np

    from vifeedback.evaluation import scope as SC
    from vifeedback.serving.scope_tfidf import TfidfScope

    d = paths.MODELS / "serve" / task
    mpath = d / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    src = paths.MODELS / "scope" / "b4prime_tfidf_logistic.pkl"
    decision = json.loads((SC.OUT / "decision.json").read_text(encoding="utf-8"))
    if not decision["passed"] or decision["selection"]["chosen"]["candidate"] != "tfidf_logistic":
        raise typer.BadParameter("B4' did not pass with the TF-IDF detector; nothing to attach")
    with open(src, "rb") as fh:  # written by `study b4prime` on this machine
        fitted = pickle.load(fh)
    scope = TfidfScope.from_sklearn(fitted["model"], fitted["threshold"])

    sets = SC.data()
    texts = (
        sets["uit_validation"] + sets["neu_validation"]["x"] + sets["neu_test"]["x"] + sets["u4"]
    )
    ours = scope.decision(texts)
    ref = fitted["model"].decision_function(texts)
    max_diff = float(np.abs(ours - ref).max())
    same_flags = bool(((ours >= scope.threshold) == (ref >= scope.threshold)).all())
    typer.echo(
        f"  numpy vs scikit-learn on {len(texts)} texts: max |diff| {max_diff:.2e}, same flags {same_flags}"
    )
    if max_diff > 1e-9 or not same_flags:
        raise typer.Exit(1)

    out = d / "scope.npz"
    scope.save(out)
    if TfidfScope.load(out).decision(texts[:500]).tolist() != ours[:500].tolist():
        raise RuntimeError("scope.npz does not round-trip")
    shutil.copy2(mpath, d / "manifest.before-scope.json")
    manifest.pop("ood", None)
    manifest["scope"] = {
        "file": out.name,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "method": "logistic regression on TF-IDF word unigrams and bigrams of the model input",
        "threshold": scope.threshold,
        "terms": len(scope.index),
        "decided_by": "cycle4.yaml v3 B4', ADR-034",
        "acceptance": {
            **{k: v["value"] for k, v in decision["rules"].items()},
            "passed": decision["passed"],
            "numpy_vs_sklearn_max_abs_diff": max_diff,
            "texts_checked": len(texts),
        },
    }
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    record = (
        paths.RESULTS / "studies" / "export" / "laptop_fp32_augmented_restorer_scope_manifest.json"
    )
    record.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"  scope detector attached: {out.name} sha256={manifest['scope']['sha256'][:12]}")


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


@serve_app.command("reproduce")
def serve_reproduce(
    repo_id: str = typer.Option("Datk4/vifeedback-sentiment-phobert", help="the published release"),
    revision: str = typer.Option("", help="a Hub commit; default: main"),
    out: str = typer.Option("", help="also write the result as JSON here"),
) -> None:
    """Reproduce the published model's validation macro-F1 from a clean clone (S10, NEXT_PLAN v5 F2).

    Downloads the serving files from the Hub, verifies them against SHA256SUMS and the manifest,
    fetches UIT-VSFC at its pinned revision if needed, and scores validation through the service's
    own pipeline on CPU. Exit code 1 unless the manifest's release-gate macro-F1 is reproduced.
    """
    from pathlib import Path

    from vifeedback.inference.reproduce import reproduce

    r = reproduce(repo_id, revision or None)
    typer.echo(
        f"  {repo_id}@{r['revision']}: {r['files_checked']} files verified; validation macro-F1 "
        f"{r['macro_f1']:.4f} (manifest {r['manifest_macro_f1']:.4f}) -> "
        f"{'REPRODUCED' if r['reproduced'] else 'NOT reproduced'}"
    )
    typer.echo(f"  seconds: {r['seconds']}")
    if out:
        Path(out).write_text(json.dumps(r, indent=2), encoding="utf-8")
    if not r["reproduced"]:
        raise typer.Exit(1)


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
