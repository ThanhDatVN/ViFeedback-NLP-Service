"""`vifeedback study`, Cycle 3: real input (ViLexNorm, NEU-ESC), diacritic restoration,
the out-of-scope score, INT8 recipes and the NEU-ESC confirmation."""

from __future__ import annotations

import json

import typer

from vifeedback import paths
from vifeedback.cli._apps import study_app


def _study_checkpoints(spec: str) -> dict[str, str]:
    """Named checkpoints: explicit `name=path,...`, or every saved CE / augmented sentiment checkpoint."""
    from pathlib import Path

    if spec:
        return dict(item.split("=", 1) for item in spec.split(","))
    found: dict[str, str] = {}
    for ckp in sorted(Path("models").glob("p*-sent-phobert-base-seg_pyvi-*-ckp")):
        name = ckp.name
        kind = (
            "vln"
            if "-seg_pyvi-aug-diac-teen-vln-s" in name
            else "aug"
            if "-seg_pyvi-aug-diac-teen-s" in name
            else "ce"
            if "-seg_pyvi-base-s" in name
            else None
        )
        seed = next(
            (p[1:] for p in ckp.name.split("-") if p.startswith("s") and p[1:].isdigit()), None
        )
        if kind and seed:
            found.setdefault(f"{kind}-s{seed}", str(ckp))
    return found


@study_app.command("challenge-seeds")
def study_challenge_seeds(
    checkpoints: str = typer.Option(
        "", help="name=path,...; default: every saved CE / augmented checkpoint"
    ),
) -> None:
    """Cycle 3 V1: all five seeds of CE and augmented on challenge v1, and the declared drop rule."""

    from vifeedback.constants import label_names
    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation.report import yaml_safe

    ckps = _study_checkpoints(checkpoints)
    df = CH.load()
    x = CH.pipeline("seg_pyvi")(df.text.tolist())
    preds = {name: EA.predict_proba(ckp, x).argmax(1) for name, ckp in ckps.items()}
    ce = {n[4:]: p for n, p in preds.items() if n.startswith("ce-s")}
    aug = {n[5:]: p for n, p in preds.items() if n.startswith("aug-s")}
    result = {"checkpoints": ckps, **CH.seed_paired_drops(df, ce, aug)}
    result["per_model"] = {name: CH.category_report(df, p) for name, p in preds.items()}

    out = paths.RESULTS / "studies" / "challenge_seeds"
    out.mkdir(parents=True, exist_ok=True)
    names = label_names("sentiment")
    table = df[["id", "category"]].copy()
    for name, p in preds.items():
        table[name] = [names[i] for i in p]
    table.to_csv(out / "predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")

    typer.echo(f"seeds: {result['seeds']}")
    typer.echo(
        f"{'category':22s} {'n':>3s} {'CE':>6s} {'aug':>6s} {'aug<CE':>7s} {'p':>7s}  confirmed"
    )
    for cat, r in result["categories"].items():
        typer.echo(
            f"{cat:22s} {r['n']:3d} {r['mean_accuracy']['ce']:6.3f} {r['mean_accuracy']['augmented']:6.3f} "
            f"{r['seeds_augmented_lower']:>5d}/5 {r['pooled_exact_mcnemar_p']:7.4f}  {r['confirmed_drop']}"
        )


@study_app.command("external")
def study_external(
    corpus: str = typer.Option("vilexnorm", help="vilexnorm | case | neu_esc"),
    checkpoints: str = typer.Option(
        "", help="name=path,...; default: every saved CE / augmented checkpoint"
    ),
) -> None:
    """Real-input tests (docs/EVALUATION_DATA.md): invariance on ViLexNorm and on sentence case,
    and NEU-ESC as a domain-shift test. Writes labels and counts only, never text."""
    import numpy as np
    import pandas as pd

    from vifeedback.data.loader import load
    from vifeedback.evaluation import bootstrap as B
    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import external as X
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.preprocess.normalize import basic_clean

    ckps = _study_checkpoints(checkpoints)
    if not ckps:
        raise typer.BadParameter("no checkpoints found under models/")
    from vifeedback.preprocess.segment import get_segmenter

    segment = get_segmenter("pyvi")
    pipelines = {
        # the service before cycle3.yaml's case rule: NFC + whitespace + segmentation, no lowercasing
        "no_lowercase": lambda ts: segment([basic_clean(t) for t in ts]),
        # the service since then (normalize.model_text)
        "lowercased": CH.pipeline("seg_pyvi"),
    }
    predict = lambda ckp, texts: EA.predict_proba(ckp, texts).argmax(1)  # noqa: E731
    out_dir = paths.RESULTS / "studies" / "external" / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"corpus": corpus, "checkpoints": ckps}
    preds: dict[str, np.ndarray] = {}

    if corpus in ("vilexnorm", "case"):
        if corpus == "vilexnorm":
            df = X.load_vilexnorm("test")
            a_texts, b_texts = df.original.tolist(), df.normalized.tolist()
            result["pairs"] = "original vs human-normalized (ViLexNorm test)"
        else:
            v = load("validation")
            variants = X.case_variants(v.sentence.tolist())
            a_texts, b_texts = variants["sentence_case"], variants["as_is"]
            y = v.sentiment.to_numpy()
            result["pairs"] = "UIT-VSFC validation: first letter capitalized vs as in the corpus"
        result["flip_rate"] = {}
        for pname, fn in pipelines.items():
            xa, xb = fn(a_texts), fn(b_texts)
            for name, ckp in ckps.items():
                pa, pb = predict(ckp, xa), predict(ckp, xb)
                preds[f"{pname}:{name}:a"], preds[f"{pname}:{name}:b"] = pa, pb
                r = X.flip_rate(pa, pb)
                if corpus == "case":
                    r["macro_f1_capitalized"] = M.macro_f1(y, pa, 3)
                    r["macro_f1_as_is"] = M.macro_f1(y, pb, 3)
                result["flip_rate"][f"{pname}:{name}"] = r
                typer.echo(
                    f"  {pname:10s} {name:10s} flips {r['flips']:4d}/{r['n']} ({r['rate']:.3f})"
                )
        # augmented vs CE at the same seed, no-lowercase pipeline (the harder case), pooled over seeds
        pairs = [
            (n, "aug-" + n[3:]) for n in ckps if n.startswith("ce-") and "aug-" + n[3:] in ckps
        ]
        if pairs:
            fa = np.concatenate(
                [preds[f"no_lowercase:{c}:a"] != preds[f"no_lowercase:{c}:b"] for c, _ in pairs]
            )
            fb = np.concatenate(
                [preds[f"no_lowercase:{a}:a"] != preds[f"no_lowercase:{a}:b"] for _, a in pairs]
            )
            result["ce_vs_aug_pooled"] = {
                "seeds": [c[3:] for c, _ in pairs],
                **X.paired_flip_test(fa, fb),
            }
    elif corpus == "neu_esc":
        df = X.load_neu_esc("test")
        y = df.sentiment.map({"negative": 0, "neutral": 1, "positive": 2}).to_numpy()
        views = {
            "all": np.ones(len(df), bool),
            "no_toxic": (df.source_label != "Toxic").to_numpy(),
            "course_topics": df.topic.isin(X.NEU_ESC_COURSE_TOPICS).to_numpy(),
        }
        result["label_counts"] = df.source_label.value_counts().to_dict()
        result["views"] = {k: int(m.sum()) for k, m in views.items()}
        result["scores"] = {}
        for pname, fn in pipelines.items():
            x = fn(df.text.tolist())
            for name, ckp in ckps.items():
                p = predict(ckp, x)
                preds[f"{pname}:{name}"] = p
                for view, m in views.items():
                    ev = M.evaluate(y[m], p[m], "sentiment")
                    result["scores"][f"{pname}:{name}:{view}"] = {
                        "macro_f1": ev["macro_f1"],
                        "per_class_f1": {c: v["f1"] for c, v in ev["per_class"].items()},
                    }
                s = result["scores"][f"{pname}:{name}:all"]
                typer.echo(f"  {pname:10s} {name:10s} macro-F1 {s['macro_f1']:.4f}")
        if "lowercased:ce-s42" in preds and "lowercased:aug-s42" in preds:
            result["aug_minus_ce_s42"] = B.paired_bootstrap(
                y,
                preds["lowercased:aug-s42"],
                preds["lowercased:ce-s42"],
                3,
                n_resamples=5000,
                seed=42,
            )
    else:
        raise typer.BadParameter("corpus must be vilexnorm, case or neu_esc")

    pd.DataFrame({k: v for k, v in preds.items()}).to_csv(
        out_dir / "predictions.csv", index_label="row"
    )
    (out_dir / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
    typer.echo(f"  -> {out_dir}")


@study_app.command("s2a-gate")
def study_s2a_gate() -> None:
    """Cycle 3 S2a development gate (cycle3.yaml v3): the ViLexNorm-lexicon recipe vs the served
    augmented recipe at the same five seeds. Passing makes it a candidate, not the served model."""
    import numpy as np
    import pandas as pd

    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import external as X
    from vifeedback.evaluation.report import yaml_safe

    ckps = _study_checkpoints("")
    seeds = sorted({n.split("-s")[-1] for n in ckps if n.startswith("vln-")}, key=int)
    seeds = [s for s in seeds if f"aug-s{s}" in ckps]
    if len(seeds) < 5:
        raise typer.BadParameter(f"need 5 seeds of both recipes, found {seeds}")
    pipe = CH.pipeline("seg_pyvi")
    predict = lambda ckp, x: EA.predict_proba(ckp, x).argmax(1)  # noqa: E731

    # 1) ViLexNorm invariance, pooled over seeds
    vl = X.load_vilexnorm("test")
    xa, xb = pipe(vl.original.tolist()), pipe(vl.normalized.tolist())
    flips: dict[str, list[np.ndarray]] = {"aug": [], "vln": []}
    per_seed_flip = {}
    for s in seeds:
        for kind in ("aug", "vln"):
            ckp = ckps[f"{kind}-s{s}"]
            f = predict(ckp, xa) != predict(ckp, xb)
            flips[kind].append(f)
            per_seed_flip[f"{kind}-s{s}"] = float(f.mean())
    fa, fv = np.concatenate(flips["aug"]), np.concatenate(flips["vln"])
    mc = X.paired_flip_test(fv, fa)  # first = vln
    vilexnorm = {
        "flip_rate": {"control_aug": float(fa.mean()), "vln": float(fv.mean())},
        "per_seed": per_seed_flip,
        **mc,
        "passed": bool(fv.mean() < fa.mean() and mc["exact_mcnemar_p"] < 0.05),
    }

    # 2) validation macro-F1, seed-paired against the registry
    reg = pd.read_csv(paths.REGISTRY)
    val = reg[reg.run_id.str.endswith("-val") & reg.run_id.str.contains("seg_pyvi")]

    def f1(prefix: str, s: str) -> float:
        row = val[val.run_id.str.startswith(prefix) & val.run_id.str.contains(f"-s{s}-")]
        return float(row.macro_f1.iloc[0])

    diffs = [
        f1("p11-sent-phobert-base-seg_pyvi-aug-diac-teen-vln-", s)
        - f1("p9-sent-phobert-base-seg_pyvi-aug-diac-teen-s", s)
        for s in seeds
    ]
    validation = {
        "per_seed_diff": dict(zip(seeds, diffs, strict=True)),
        "mean_diff": float(np.mean(diffs)),
        "passed": bool(np.mean(diffs) >= -0.005),
    }

    # 3) challenge v1 categories, 5-seed means
    df = CH.load()
    xc = pipe(df.text.tolist())
    acc: dict[str, dict[str, list[float]]] = {}
    s_mask = df.scored.to_numpy()
    y = df.y.to_numpy()
    for kind in ("aug", "vln"):
        for s in seeds:
            p = predict(ckps[f"{kind}-s{s}"], xc)
            for cat in sorted(df[s_mask].category.unique()):
                m = s_mask & (df.category == cat).to_numpy()
                acc.setdefault(cat, {"aug": [], "vln": []})[kind].append(
                    float((p[m] == y[m].astype(int)).mean())
                )
    cats = {
        c: {
            "aug": float(np.mean(v["aug"])),
            "vln": float(np.mean(v["vln"])),
            "diff": float(np.mean(v["vln"]) - np.mean(v["aug"])),
        }
        for c, v in acc.items()
    }
    worst = min(cats.values(), key=lambda v: v["diff"])["diff"]
    challenge = {
        "categories": cats,
        "unaccented_diff": cats["unaccented_typed"]["diff"],
        "worst_category_diff": worst,
        "passed": bool(cats["unaccented_typed"]["diff"] >= -0.02 and worst >= -0.05),
    }

    gate = bool(vilexnorm["passed"] and validation["passed"] and challenge["passed"])
    result = {
        "seeds": seeds,
        "vilexnorm": vilexnorm,
        "validation": validation,
        "challenge_v1": challenge,
        "gate_passed": gate,
    }
    out = paths.RESULTS / "studies" / "s2a_gate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
    typer.echo(
        f"ViLexNorm flip rate: control {fa.mean():.4f} -> vln {fv.mean():.4f}, pooled p = {mc['exact_mcnemar_p']:.4g}  [{vilexnorm['passed']}]"
    )
    typer.echo(
        f"validation macro-F1, vln - control: {np.mean(diffs):+.4f} per seed {[round(d, 4) for d in diffs]}  [{validation['passed']}]"
    )
    for c, v in cats.items():
        typer.echo(f"  {c:22s} control {v['aug']:.3f}  vln {v['vln']:.3f}  {v['diff']:+.3f}")
    typer.echo(
        f"challenge v1: unaccented {challenge['unaccented_diff']:+.3f}, worst {worst:+.3f}  [{challenge['passed']}]"
    )
    typer.echo(f"S2a development gate passed: {gate}")


@study_app.command("restore-dev")
def study_restore_dev() -> None:
    """Cycle 3 S2b development: the served model with and without diacritic restoration, on
    validation (clean and with diacritics stripped) and challenge v1 unaccented rows (cycle3.yaml)."""
    import hashlib
    import time

    from vifeedback.data.loader import load
    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import external as X
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.preprocess.diacritics import Restorer
    from vifeedback.preprocess.normalize import model_text, strip_diacritics
    from vifeedback.preprocess.segment import get_segmenter

    tr, dv = load("train"), load("validation")
    restorer = Restorer.fit(tr.sentence.tolist())
    path = paths.MODELS / "diacritics" / "restorer.json"  # built from corpus text: git-ignored
    restorer.save(path)
    segment = get_segmenter("pyvi")
    plain = lambda ts: segment([model_text(t) for t in ts])  # noqa: E731
    restored = lambda ts: segment([restorer(model_text(t)) for t in ts])  # noqa: E731

    manifest = json.loads(
        (paths.MODELS / "serve" / "sentiment" / "manifest.json").read_text(encoding="utf-8")
    )
    served = str(paths.ROOT / manifest["checkpoint"])
    predict = lambda x: EA.predict_proba(served, x).argmax(1)  # noqa: E731
    y = dv.sentiment.to_numpy()
    clean = dv.sentence.tolist()
    stripped = [strip_diacritics(s) for s in clean]

    p_clean, p_clean_r = predict(plain(clean)), predict(restored(clean))
    p_str, p_str_r = predict(plain(stripped)), predict(restored(stripped))
    ok = tot = 0
    for s, g in zip(stripped, clean, strict=True):
        r = restorer.restore(s).split()
        ok += sum(a == b for a, b in zip(r, g.split(), strict=False))
        tot += len(g.split())
    t0 = time.perf_counter()
    for s in stripped:
        restorer(s)
    ms = 1000 * (time.perf_counter() - t0) / len(stripped)

    ch = CH.load()
    m = (ch.category == "unaccented_typed").to_numpy()
    yc = ch.y.to_numpy()[m].astype(int)
    texts = ch.text[m].tolist()
    a, b = predict(plain(texts)) == yc, predict(restored(texts)) == yc
    mc = X.paired_flip_test(~a, ~b)  # errors without vs with restoration
    result = {
        "restorer": {
            "threshold": restorer.threshold,
            "keys": len(restorer.candidates),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "syllable_accuracy_on_stripped_validation": ok / tot,
            "ms_per_sentence": ms,
        },
        "served_checkpoint": manifest["checkpoint"],
        "validation_clean": {
            "touched": int(sum(restorer.needs_restoring(model_text(s)) for s in clean)),
            "predictions_changed": int((p_clean != p_clean_r).sum()),
            "macro_f1": M.macro_f1(y, p_clean, 3),
        },
        "validation_stripped": {
            "macro_f1_without": M.macro_f1(y, p_str, 3),
            "macro_f1_with": M.macro_f1(y, p_str_r, 3),
        },
        "challenge_v1_unaccented": {
            "n": int(m.sum()),
            "accuracy_without": float(a.mean()),
            "accuracy_with": float(b.mean()),
            "errors_only_without": mc["only_first_flips"],
            "errors_only_with": mc["only_second_flips"],
            "exact_mcnemar_p": mc["exact_mcnemar_p"],
        },
    }
    out = paths.RESULTS / "studies" / "restoration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "development.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
    v, s, c = (
        result["validation_clean"],
        result["validation_stripped"],
        result["challenge_v1_unaccented"],
    )
    typer.echo(
        f"restorer: syllable accuracy {ok / tot:.4f}, {ms:.2f} ms per sentence, threshold {restorer.threshold:.3f}"
    )
    typer.echo(
        f"validation clean: {v['touched']} touched, {v['predictions_changed']} predictions changed"
    )
    typer.echo(
        f"validation stripped: macro-F1 {s['macro_f1_without']:.4f} -> {s['macro_f1_with']:.4f}"
    )
    typer.echo(
        f"challenge v1 unaccented: {c['accuracy_without']:.3f} -> {c['accuracy_with']:.3f} (p = {c['exact_mcnemar_p']:.4g})"
    )


@study_app.command("restore-compare")
def study_restore_compare() -> None:
    """Cycle 3 S2b-prime development (cycle3.yaml v4): CE + restoration vs augmented + restoration,
    five seeds each. Descriptive; the decision is taken on challenge v2."""
    import numpy as np

    from vifeedback.data.loader import load
    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import external as X
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.preprocess.diacritics import Restorer
    from vifeedback.preprocess.normalize import model_text, strip_diacritics
    from vifeedback.preprocess.segment import get_segmenter

    restorer = Restorer.load(paths.MODELS / "diacritics" / "restorer.json")
    segment = get_segmenter("pyvi")
    pipe = lambda ts: segment([restorer(model_text(t)) for t in ts])  # noqa: E731
    ckps = _study_checkpoints("")
    seeds = sorted({n.split("-s")[-1] for n in ckps if n.startswith("ce-")}, key=int)
    seeds = [s for s in seeds if f"aug-s{s}" in ckps]

    dv = load("validation")
    y = dv.sentiment.to_numpy()
    x_clean = pipe(dv.sentence.tolist())
    x_strip = pipe([strip_diacritics(t) for t in dv.sentence])
    ch = CH.load()
    x_ch = pipe(ch.text.tolist())
    scored = ch.scored.to_numpy()
    yc = ch.y.to_numpy()

    per: dict[str, dict] = {}
    right: dict[str, dict[str, np.ndarray]] = {"ce": {}, "aug": {}}
    for kind in ("ce", "aug"):
        for s in seeds:
            ckp = ckps[f"{kind}-s{s}"]
            pc = EA.predict_proba(ckp, x_ch).argmax(1)
            right[kind][s] = pc == np.where(scored, yc, -1)
            per[f"{kind}-s{s}"] = {
                "validation_clean": M.macro_f1(y, EA.predict_proba(ckp, x_clean).argmax(1), 3),
                "validation_stripped": M.macro_f1(y, EA.predict_proba(ckp, x_strip).argmax(1), 3),
                **{
                    c: float(right[kind][s][scored & (ch.category == c).to_numpy()].mean())
                    for c in sorted(ch[scored].category.unique())
                },
            }
            typer.echo(f"  {kind}-s{s}: stripped {per[f'{kind}-s{s}']['validation_stripped']:.4f}")
    cats = sorted(ch[scored].category.unique())
    summary = {}
    for key in ("validation_clean", "validation_stripped", *cats):
        summary[key] = {
            k: float(np.mean([per[f"{k}-s{s}"][key] for s in seeds])) for k in ("ce", "aug")
        }
        summary[key]["diff_ce_minus_aug"] = summary[key]["ce"] - summary[key]["aug"]
    m = scored & (ch.category == "mixed_aspect").to_numpy()
    contrast = X.paired_flip_test(
        np.concatenate([~right["ce"][s][m] for s in seeds]),
        np.concatenate([~right["aug"][s][m] for s in seeds]),
    )  # errors: first = CE + restorer
    result = {"seeds": seeds, "per_model": per, "mean": summary, "contrast_pooled": contrast}
    out = paths.RESULTS / "studies" / "restoration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ce_vs_aug.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
    for key, v in summary.items():
        typer.echo(
            f"  {key:22s} CE+R {v['ce']:.4f}  aug+R {v['aug']:.4f}  {v['diff_ce_minus_aug']:+.4f}"
        )
    typer.echo(
        f"contrast, errors only CE+R {contrast['only_first_flips']} vs only aug+R {contrast['only_second_flips']}, p = {contrast['exact_mcnemar_p']:.4g}"
    )


@study_app.command("ood-dev")
def study_ood_dev(
    checkpoint: str = typer.Option("", help="default: the served checkpoint (release manifest)"),
) -> None:
    """Cycle 3 S3 development: which out-of-scope score separates off-topic input, and where the
    95%-retention threshold falls. Development data only; confirmation is challenge v2 (cycle3.yaml)."""
    from vifeedback.data.loader import load
    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import ood as OOD
    from vifeedback.evaluation.report import yaml_safe

    if not checkpoint:
        manifest = json.loads(
            (paths.MODELS / "serve" / "sentiment" / "manifest.json").read_text(encoding="utf-8")
        )
        checkpoint = str(paths.ROOT / manifest["checkpoint"])
    pipe = CH.pipeline("seg_pyvi")
    tr, dv = load("train"), load("validation")
    u4 = json.loads(
        (paths.RESULTS / "studies" / "ood" / "u4_offtopic_dev.json").read_text(encoding="utf-8")
    )
    ch = CH.load()
    off_v1 = ch[ch.category == CH.OUT_OF_SCOPE].text.tolist()
    off_u4 = [r["text"] for r in u4["rows"]]

    f_tr, _ = OOD.encode(checkpoint, pipe(tr.sentence.tolist()))
    maha = OOD.fit_mahalanobis(f_tr, tr.sentiment.to_numpy())
    f_dv, l_dv = OOD.encode(checkpoint, pipe(dv.sentence.tolist()))
    f_o, l_o = OOD.encode(checkpoint, pipe(off_v1 + off_u4))
    s_in, s_out = OOD.scores(l_dv, f_dv, maha), OOD.scores(l_o, f_o, maha)

    result: dict = {
        "checkpoint": checkpoint,
        "in_domain": "UIT-VSFC validation",
        "out_of_scope": {"challenge_v1": len(off_v1), "u4_generated": len(off_u4)},
        "methods": {},
    }
    for method in s_in:
        r = OOD.evaluate(s_in[method], s_out[method])
        r["challenge_v1_only"] = OOD.evaluate(s_in[method], s_out[method][: len(off_v1)])
        result["methods"][method] = r
        typer.echo(
            f"  {method:16s} AUROC {r['auroc']:.3f}  caught {r['out_of_scope_caught']:.2f} of off-topic at "
            f"{r['in_domain_flagged']:.2f} of validation flagged  (v1 rows only: AUROC {r['challenge_v1_only']['auroc']:.3f})"
        )
    best = max(result["methods"], key=lambda m: result["methods"][m]["auroc"])
    result["development_choice"] = best
    out = paths.RESULTS / "studies" / "ood"
    (out / "development.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
    typer.echo(
        f"development choice: {best} (confirmation: challenge v2, AUROC >= 0.90, <= 5% of validation flagged)"
    )


@study_app.command("int8-recipes")
def study_int8_recipes(
    work: str = typer.Option("models/int8_candidates", help="build directory (git-ignored)"),
) -> None:
    """Cycle 3 S5: build the INT8 recipes, choose one on fidelity to FP32 (train subset), then accept
    or reject it once on validation (cycle3.yaml). Latency is measured separately, on an idle machine."""
    from pathlib import Path

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.data.loader import load
    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import llm_reference as LR
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.inference import int8_recipes as Q
    from vifeedback.inference import onnx_export as OX

    served = json.loads(
        (paths.MODELS / "serve" / "sentiment" / "manifest.json").read_text(encoding="utf-8")
    )
    ckp = paths.ROOT / served["checkpoint"]
    w = Path(work)
    plain_dir = w / "fp32_plain"
    if not (plain_dir / "pre.onnx").exists():
        model = AutoModelForSequenceClassification.from_pretrained(ckp)
        tok = AutoTokenizer.from_pretrained(ckp)
        plain = OX.export_fp32(model, tok, plain_dir)
        tok.save_pretrained(plain_dir)
        OX.preprocess_for_quantization(plain, plain_dir / "pre.onnx")
    fp32 = OX.OnnxClassifier(plain_dir, model_file="model.onnx")
    pipeline = CH.pipeline("seg_pyvi")

    def logits(clf, texts: list[str], batch: int = 64):
        # one session call per batch: a single call over 1,583 sentences asked ORT for a 700 MB buffer
        import numpy as np

        return np.concatenate(
            [clf.logits(texts[i : i + batch]) for i in range(0, len(texts), batch)]
        )

    tr = load("train")
    idx = LR.prompt_dev_subset(tr.sentiment.to_numpy(), n=2000, per_class_min=400, seed=0)
    x_tr = pipeline(tr.sentence.iloc[idx].tolist())
    ref = logits(fp32, x_tr)

    candidates = {}
    for recipe in Q.RECIPES:
        info = Q.build(plain_dir / "pre.onnx", plain_dir, w / recipe, recipe)
        clf = OX.OnnxClassifier(w / recipe, model_file="model.int8.onnx")
        candidates[recipe] = {**info, **Q.fidelity(ref, logits(clf, x_tr))}
        c = candidates[recipe]
        typer.echo(
            f"  {recipe:15s} {c['size_mb']:6.1f} MB  agreement {c['label_agreement']:.4f}  "
            f"neutral {c['neutral_agreement']:.4f}  |dlogit| {c['mean_abs_logit_diff']:.4f}"
        )
    chosen = Q.select(candidates)
    result: dict = {
        "checkpoint": served["checkpoint"],
        "fidelity_subset": {"split": "train", "n": len(idx), "per_class_min": 400, "seed": 0},
        "candidates": candidates,
        "chosen": chosen,
    }
    if chosen:
        dv = load("validation")
        x_dv = pipeline(dv.sentence.tolist())
        int8 = OX.OnnxClassifier(w / chosen, model_file="model.int8.onnx")
        result["validation"] = Q.acceptance(
            dv.sentiment.to_numpy(), logits(fp32, x_dv).argmax(1), logits(int8, x_dv).argmax(1)
        )
        v = result["validation"]
        typer.echo(
            f"chosen {chosen}: macro-F1 drop {v['macro_f1_drop']:+.4f} (upper {v['macro_f1_drop_upper_95_one_sided']:+.4f}), "
            f"neutral F1 {v['neutral_f1_fp32']:.3f} -> {v['neutral_f1_int8']:.3f}; quality passed: {v['quality_passed']}"
        )
    else:
        typer.echo("no recipe is at or under the size limit")
    out = paths.RESULTS / "studies" / "int8_recipes"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
    typer.echo(
        f"  -> {out} (latency: study latency --extra-onnx {w / (chosen or '')}/model.int8.onnx)"
    )


@study_app.command("neu-esc-confirm")
def study_neu_esc_confirm() -> None:
    """Cycle 3 confirmation on NEU-ESC (cycle3.yaml v5, ADR-030): S2b, S2b-prime and S3 rules."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from vifeedback.data.loader import load
    from vifeedback.evaluation import bootstrap as B
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import external as X
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation import ood as OOD
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.preprocess.diacritics import Restorer
    from vifeedback.preprocess.normalize import model_text, strip_diacritics
    from vifeedback.preprocess.segment import get_segmenter

    ne = X.load_neu_esc("test")
    y = ne.sentiment.map({"negative": 0, "neutral": 1, "positive": 2}).to_numpy()
    words = ne.text.str.split().str.len().to_numpy()
    contrast_re = r"(?<!\w)(?:nhưng|tuy|mặc dù|song)(?!\w)"
    cells = {
        "all": np.ones(len(ne), bool),
        "contrast": ne.text.str.contains(contrast_re, regex=True).to_numpy(),
        "short_neutral": (y == 1) & (words <= 10),
    }
    off_topic = ne.topic.isin(("Spam", "News", "Jobs & Recruitment", "Club & Events")).to_numpy()
    sizes = {k: int(v.sum()) for k, v in cells.items()}
    sizes.update({"unaccented": len(ne), "off_topic": int(off_topic.sum())})
    typer.echo(f"cells: {sizes}")

    restorer = Restorer.load(paths.MODELS / "diacritics" / "restorer.json")
    segment = get_segmenter("pyvi")

    def plain(ts: list[str]) -> list[str]:
        return segment([model_text(t) for t in ts])

    def restored(ts: list[str]) -> list[str]:
        return segment([restorer(model_text(t)) for t in ts])

    texts = ne.text.tolist()
    stripped = [strip_diacritics(t) for t in texts]
    x_r, x_rs = restored(texts), restored(stripped)
    dv = load("validation")
    x_dv = restored(dv.sentence.tolist())

    ckps = _study_checkpoints("")
    seeds = sorted({n.split("-s")[-1] for n in ckps if n.startswith("ce-")}, key=int)
    seeds = [s for s in seeds if f"aug-s{s}" in ckps]
    pred: dict[str, np.ndarray] = {}
    val_f1: dict[str, float] = {}
    for kind in ("ce", "aug"):
        for s in seeds:
            ckp = ckps[f"{kind}-s{s}"]
            pred[f"{kind}-s{s}:R"] = EA.predict_proba(ckp, x_r).argmax(1)
            pred[f"{kind}-s{s}:R_stripped"] = EA.predict_proba(ckp, x_rs).argmax(1)
            p_dv = EA.predict_proba(ckp, x_dv).argmax(1)
            val_f1[f"{kind}-s{s}"] = M.macro_f1(dv.sentiment.to_numpy(), p_dv, 3)
            typer.echo(f"  {kind}-s{s} done")
    manifest = json.loads(
        (paths.MODELS / "serve" / "sentiment" / "manifest.json").read_text(encoding="utf-8")
    )
    served = str(paths.ROOT / manifest["checkpoint"])
    pred["served:plain"] = EA.predict_proba(served, plain(texts)).argmax(1)
    pred["served:plain_stripped"] = EA.predict_proba(served, plain(stripped)).argmax(1)
    served_name = next(n for n, p in ckps.items() if Path(p).name == Path(served).name)

    def f1(p: np.ndarray, m: np.ndarray) -> float:
        return M.macro_f1(y[m], p[m], 3)

    def acc(p: np.ndarray, m: np.ndarray) -> float:
        return float((p[m] == y[m]).mean())

    result: dict = {"cells": sizes, "seeds": seeds, "served": served_name}

    # S2b: the served model with vs without the restorer
    s_r, s_rs = pred[f"{served_name}:R"], pred[f"{served_name}:R_stripped"]
    pb = B.paired_bootstrap(y, s_rs, pred["served:plain_stripped"], 3, n_resamples=5000, seed=42)
    all_diff = f1(s_r, cells["all"]) - f1(pred["served:plain"], cells["all"])
    result["S2b_restoration"] = {
        "unaccented_macro_f1": {
            "without": f1(pred["served:plain_stripped"], cells["all"]),
            "with": f1(s_rs, cells["all"]),
        },
        "paired": pb,
        "all_cell_diff": all_diff,
        "passed": bool(pb["ci_low"] > 0 and abs(all_diff) <= 0.002),
    }

    # S2b-prime: CE + restorer vs augmented + restorer, five seeds
    def mean(kind: str, key: str, fn, m: np.ndarray) -> float:
        return float(np.mean([fn(pred[f"{kind}-s{s}:{key}"], m) for s in seeds]))

    c = cells["contrast"]
    ce_err = np.concatenate([pred[f"ce-s{s}:R"][c] != y[c] for s in seeds])
    aug_err = np.concatenate([pred[f"aug-s{s}:R"][c] != y[c] for s in seeds])
    mc = X.paired_flip_test(
        aug_err, ce_err
    )  # first: only augmented+R wrong; second: only CE+R wrong
    checks: dict[str, dict] = {
        "contrast_accuracy": {
            "ce": mean("ce", "R", acc, c),
            "aug": mean("aug", "R", acc, c),
            "only_aug_wrong": mc["only_first_flips"],
            "only_ce_wrong": mc["only_second_flips"],
            "p": mc["exact_mcnemar_p"],
            "passed": bool(
                sizes["contrast"] >= 100
                and mc["exact_mcnemar_p"] < 0.05
                and mc["only_first_flips"] > mc["only_second_flips"]
            ),
        },
        "unaccented_macro_f1": {
            "ce": mean("ce", "R_stripped", f1, cells["all"]),
            "aug": mean("aug", "R_stripped", f1, cells["all"]),
        },
        "all_macro_f1": {
            "ce": mean("ce", "R", f1, cells["all"]),
            "aug": mean("aug", "R", f1, cells["all"]),
        },
        "short_neutral_accuracy": {
            "ce": mean("ce", "R", acc, cells["short_neutral"]),
            "aug": mean("aug", "R", acc, cells["short_neutral"]),
        },
        "validation_macro_f1": {
            "ce": float(np.mean([val_f1[f"ce-s{s}"] for s in seeds])),
            "aug": float(np.mean([val_f1[f"aug-s{s}"] for s in seeds])),
        },
    }
    margins = {
        "unaccented_macro_f1": 0.02,
        "all_macro_f1": 0.005,
        "short_neutral_accuracy": 0.05,
        "validation_macro_f1": 0.005,
    }
    for key, margin in margins.items():
        d = checks[key]
        d["diff"] = d["ce"] - d["aug"]
        big_enough = key != "short_neutral_accuracy" or sizes["short_neutral"] >= 100
        d["passed"] = bool(d["diff"] >= -margin and big_enough)
    result["S2b_prime"] = {"checks": checks, "passed": all(v["passed"] for v in checks.values())}

    # S3: Mahalanobis on the served model, threshold as in development
    tr = load("train")
    f_tr, _ = OOD.encode(served, plain(tr.sentence.tolist()))
    maha = OOD.fit_mahalanobis(f_tr, tr.sentiment.to_numpy())
    f_dv, l_dv = OOD.encode(served, plain(dv.sentence.tolist()))
    off_texts = [t for t, m in zip(texts, off_topic, strict=True) if m]
    f_o, l_o = OOD.encode(served, plain(off_texts))
    s_in, s_out = OOD.scores(l_dv, f_dv, maha), OOD.scores(l_o, f_o, maha)
    s3 = {m: OOD.evaluate(s_in[m], s_out[m]) for m in s_in}
    result["S3_out_of_scope"] = {
        "methods": s3,
        "passed": bool(s3["neg_mahalanobis"]["auroc"] >= 0.90),
    }

    out = paths.RESULTS / "studies" / "neu_esc_confirm"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
    pd.DataFrame(pred).to_csv(out / "predictions.csv", index_label="row")  # labels only, no text
    r2 = result["S2b_restoration"]
    typer.echo(
        f"S2b: unaccented macro-F1 {r2['unaccented_macro_f1']['without']:.4f} -> "
        f"{r2['unaccented_macro_f1']['with']:.4f} [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}]; "
        f"all-cell diff {all_diff:+.4f} -> passed {r2['passed']}"
    )
    for k, v in checks.items():
        typer.echo(f"S2b': {k:24s} CE+R {v['ce']:.4f}  aug+R {v['aug']:.4f}  passed {v['passed']}")
    typer.echo(
        f"S2b' passed: {result['S2b_prime']['passed']} "
        f"(contrast p = {checks['contrast_accuracy']['p']:.4g})"
    )
    typer.echo(
        f"S3: Mahalanobis AUROC {s3['neg_mahalanobis']['auroc']:.3f} "
        f"(max-prob {s3['max_probability']['auroc']:.3f}) -> passed {result['S3_out_of_scope']['passed']}"
    )
