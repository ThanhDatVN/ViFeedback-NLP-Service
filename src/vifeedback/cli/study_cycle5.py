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


H10B_S42 = "models/p15-sent-phobert-base-seg_pyvi-h10b-anchored_orig-s42-063770c8-ckp"


@study_app.command("s7b-trim")
def study_s7b_trim(checkpoint: str = typer.Option(H10B_S42)) -> None:
    """S7b (cycle5.yaml v7): the served H10b checkpoint with its vocabulary cut to 17,500 entries."""
    import json
    from pathlib import Path

    from vifeedback import paths
    from vifeedback.inference import vocab_trim as V
    from vifeedback.inference.phobert_tokenizer import PhobertBPE
    from vifeedback.training.domain import serving_transform

    src = Path(checkpoint)
    tok = PhobertBPE(src)
    used: set[str] = set()
    sources = {}
    for name, texts in V.coverage_texts(serving_transform()).items():
        pieces = V.used_pieces(tok, texts) & set(tok.encoder)
        sources[name] = {"texts": len(texts), "pieces_in_vocab": len(pieces)}
        used |= pieces
    keep = V.choose(V.read_vocab(src), used)
    dst = paths.MODELS / "trimmed" / src.name.replace("-ckp", f"-v{V.N_ENTRIES}-ckp")
    record = V.write_trimmed(src, dst, keep)
    record.update(sources=sources, used_pieces=len(used), checkpoint=dst.name)
    V.OUT.mkdir(parents=True, exist_ok=True)
    (V.OUT / "trim.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    typer.echo(f"  {len(used)} used pieces, {record['entries']} entries -> {dst}")


@study_app.command("s7b-fidelity")
def study_s7b_fidelity(
    candidate: str = typer.Option("models/candidate/sentiment"),
    served: str = typer.Option("models/serve/sentiment"),
) -> None:
    """S7b (cycle5.yaml v7): the trimmed graph against the served FP32 graph on held-out text."""
    import json
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd

    from vifeedback import paths
    from vifeedback.data.loader import load
    from vifeedback.evaluation import challenge as C
    from vifeedback.evaluation import external as X
    from vifeedback.evaluation import metrics as M
    from vifeedback.inference import vocab_trim as V
    from vifeedback.inference.onnx_export import OnnxClassifier
    from vifeedback.inference.phobert_tokenizer import PhobertBPE
    from vifeedback.preprocess.normalize import strip_diacritics
    from vifeedback.preprocess.segment import get_segmenter
    from vifeedback.preprocess.variants import load_variant
    from vifeedback.serving import pipeline as SP
    from vifeedback.training.domain import neu_esc, serving_transform

    cdir, sdir = Path(candidate), Path(served)
    mc = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))
    ms = json.loads((sdir / "manifest.json").read_text(encoding="utf-8"))
    clf_c = OnnxClassifier(cdir, max_length=mc["max_length"], model_file=mc["model_file"])
    clf_s = OnnxClassifier(sdir, max_length=ms["max_length"], model_file=ms["model_file"])
    labels = ms["labels"]
    restorer = SP.load_restorer(sdir, ms["restorer"])
    seg = get_segmenter("pyvi")
    transform = serving_transform()

    sets: dict[str, tuple[list[str], np.ndarray | None]] = {}
    val = load_variant("seg_pyvi", "validation")
    y_val = val["sentiment"].to_numpy()
    sets["uit_validation"] = (val.sentence.tolist(), y_val)
    stripped = [strip_diacritics(t) for t in load("validation").sentence]
    sets["uit_validation_stripped"] = (SP.prepare(stripped, restorer, seg), y_val)
    neu = neu_esc("validation", transform, in_scope=False)
    sets["neu_esc_validation"] = (neu["x"], neu["y"])
    vl = X.load_vilexnorm("train")
    for name, index in (("dev", "vilexnorm_dev_index.csv"), ("confirm", "h10b_confirm_index.csv")):
        rows = pd.read_csv(paths.RESULTS / "studies" / "cycle5" / index).row.to_numpy()
        for form in ("original", "normalized"):
            sets[f"vilexnorm_{name}_{form}"] = (transform(vl[form].iloc[rows].tolist()), None)
    ch = C.load()
    ch = ch[ch.sentiment.isin(labels)]
    y_ch = ch.sentiment.map({c: i for i, c in enumerate(labels)}).to_numpy()
    sets["challenge_v1"] = (SP.prepare(ch.text.tolist(), restorer, seg), y_ch)

    full_tok = PhobertBPE(sdir)
    kept = set(PhobertBPE(cdir).encoder)
    results: dict[str, Any] = {}
    for name, (x, y) in sets.items():
        pc = V.logits(clf_c, x).argmax(1)
        ps = V.logits(clf_s, x).argmax(1)
        r: dict[str, Any] = {"n": len(x), "agreement": float((pc == ps).mean())}
        r.update(V.removed_share(full_tok, kept, x))
        if y is not None:
            r["macro_f1_candidate"] = M.macro_f1(y, pc, 3)
            r["macro_f1_served"] = M.macro_f1(y, ps, 3)
        results[name] = r
        typer.echo(
            f"  {name:34s} n={len(x):5d} agreement {r['agreement']:.4f} "
            f"pieces removed {r['pieces_removed']:.4f}"
            + (
                f"  F1 {r['macro_f1_candidate']:.4f} vs {r['macro_f1_served']:.4f}"
                if y is not None
                else ""
            )
        )

    size_mb = (cdir / mc["model_file"]).stat().st_size / 1e6
    rules = {
        "1_size_mb_le_200": {"value": size_mb, "passed": size_mb <= 200},
        "2_release_gate": {"passed": bool(mc.get("acceptance", {}).get("passed", False))},
        "3_agreement_ge_0.99": {
            "min": min(r["agreement"] for r in results.values()),
            "passed": all(r["agreement"] >= 0.99 for r in results.values()),
        },
        "4_macro_f1_within_0.005": {
            k: results[k]["macro_f1_candidate"] - results[k]["macro_f1_served"]
            for k in ("uit_validation", "neu_esc_validation")
        },
    }
    rules["4_macro_f1_within_0.005"]["passed"] = all(
        abs(v) <= 0.005 for k, v in rules["4_macro_f1_within_0.005"].items() if k != "passed"
    )
    passed = all(v["passed"] for v in rules.values())
    out = {"candidate": mc["model_file"], "sha256": mc["sha256"], "sets": results, "rules": rules}
    out["passed"] = passed
    V.OUT.mkdir(parents=True, exist_ok=True)
    (V.OUT / "fidelity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    typer.echo(f"  size {size_mb:.1f} MB; S7b {'PASSED' if passed else 'NOT passed'} -> {V.OUT}")
