"""Export → verify → release, with a staging boundary — the quality contract from review R3.

What the earlier export did wrong, and what this module does instead:

| Before (review R3) | Now |
|---|---|
| INT8 calibration and parity used **raw** text; the service feeds pyvi-segmented text | Both go through the checkpoint's own preprocessing |
| Parity on the first 64 validation sentences, label agreement only | The full validation set, in mixed-length batches *and* one at a time |
| `logits_close` computed but never enforced | FP32 must be logit-close or the release fails |
| No quality check of INT8 | INT8 must stay within the pre-registered budget: ≤ 0.005 macro-F1 below the PyTorch model |
| Calibration drew on validation, the acceptance set | Calibration draws on a stratified **train** subset; acceptance stays independent |
| Files written straight into the served directory | Built in a staging directory; the served directory is replaced only after every check passes |
| The loader picked `model.quant.onnx` if one existed, even a stale one | A manifest names the one served file and its SHA-256; the loader refuses a mismatch |

The acceptance set is validation, not test: the test split is reserved for gate evaluations.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST = "manifest.json"
FP32_ATOL = 1e-3
INT8_MACRO_F1_BUDGET = 0.005  # docs/EVALUATION_PROTOCOL.md § Accuracy budget
MIN_LABEL_AGREEMENT_INT8 = 0.99


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_model_file(model_dir: Path) -> Path:
    """The file to serve. With a manifest: exactly the file it names, checksum-verified. Without one
    (artifacts exported before the manifest existed): the legacy preference order."""
    model_dir = Path(model_dir)
    m = model_dir / MANIFEST
    if m.exists():
        manifest = json.loads(m.read_text(encoding="utf-8"))
        path = model_dir / manifest["model_file"]
        if not path.exists():
            raise FileNotFoundError(f"manifest names {path.name}, which is missing")
        if sha256(path) != manifest["sha256"]:
            raise ValueError(f"{path.name} does not match the SHA-256 recorded in the manifest")
        return path
    for c in ("model.quant.onnx", "model.opt.onnx", "model.onnx"):
        if (model_dir / c).exists():
            return model_dir / c
    raise FileNotFoundError(f"no .onnx found in {model_dir}")


def stratified_subset(y: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Indices of an n-example subset with every class represented in proportion, minimum 10.

    Static INT8 calibration only sees what it is given; a random 200 sentences would hold about
    eight neutral ones.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    take = np.maximum(10, np.round(n * counts / counts.sum()).astype(int))
    idx = [
        rng.choice(np.flatnonzero(y == c), size=min(t, cnt), replace=False)
        for c, t, cnt in zip(classes, take, counts, strict=True)
    ]
    return np.sort(np.concatenate(idx))


def acceptance(
    ref_logits: np.ndarray,
    got_logits: np.ndarray,
    y: np.ndarray,
    k: int,
    quantized: bool,
    single_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    """The release decision, as a pure function of the logits so it can be tested without ONNX."""
    from vifeedback.evaluation.metrics import macro_f1

    ref_pred, got_pred = ref_logits.argmax(1), got_logits.argmax(1)
    r = {
        "n": len(y),
        "max_abs_logit_diff": float(np.abs(ref_logits - got_logits).max()),
        "logits_close": bool(np.allclose(ref_logits, got_logits, atol=FP32_ATOL)),
        "label_agreement": float((ref_pred == got_pred).mean()),
        "macro_f1_torch": macro_f1(y, ref_pred, k),
        "macro_f1_onnx": macro_f1(y, got_pred, k),
    }
    r["macro_f1_drop"] = r["macro_f1_torch"] - r["macro_f1_onnx"]
    if single_logits is not None:
        # Batch-of-one must equal the padded batch: a dynamic-axis bug shows up exactly here.
        r["batch_vs_single_max_diff"] = float(
            np.abs(single_logits - got_logits[: len(single_logits)]).max()
        )

    failures = []
    if quantized:
        if r["macro_f1_drop"] > INT8_MACRO_F1_BUDGET:
            failures.append(f"INT8 macro-F1 drop {r['macro_f1_drop']:.4f} > {INT8_MACRO_F1_BUDGET}")
        if r["label_agreement"] < MIN_LABEL_AGREEMENT_INT8:
            failures.append(
                f"INT8 label agreement {r['label_agreement']:.4f} < {MIN_LABEL_AGREEMENT_INT8}"
            )
    elif not r["logits_close"]:
        failures.append(
            f"FP32 logits diverge: max |diff| {r['max_abs_logit_diff']:.2e} > {FP32_ATOL}"
        )
    if single_logits is not None and r["batch_vs_single_max_diff"] > (
        0.5 if quantized else FP32_ATOL * 10
    ):
        failures.append(f"batch vs single mismatch {r['batch_vs_single_max_diff']:.2e}")
    r["failures"] = failures
    r["passed"] = not failures
    return r


def _torch_logits(
    model, tok, texts: list[str], max_length: int, batch_size: int = 32
) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(
                texts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            out.append(model(**enc).logits.float().numpy())
    return np.concatenate(out)


def release(
    *,
    checkpoint: str | Path,
    task: str,
    preprocessing: str,
    quantize: str,
    out_dir: Path,
    pipeline: Callable[[list[str]], list[str]],
    calib_raw: list[str],
    accept_raw: list[str],
    accept_y: np.ndarray,
    max_length: int = 96,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Build in staging, verify, and only then replace `out_dir`. Returns the manifest.

    On failure the served directory is untouched and the staging directory is kept for inspection.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.constants import label_names
    from vifeedback.inference import onnx_export as OX

    if quantize not in ("none", "dynamic", "static"):
        raise ValueError(f"quantize must be none | dynamic | static, got {quantize!r}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    staging = out_dir.parent / f".staging-{out_dir.name}-{stamp}"
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    tok = AutoTokenizer.from_pretrained(str(checkpoint))

    fp32 = OX.export_fp32(model, tok, staging, max_length)
    served = OX.optimize_graph(fp32, staging / "model.opt.onnx")
    if quantize == "dynamic":
        served = OX.quantize_dynamic_int8(served, staging / "model.quant.onnx")
    elif quantize == "static":
        calib = pipeline(calib_raw)
        served = OX.quantize_static_int8(
            served, staging / "model.quant.onnx", calib, tok, max_length
        )
    for f in staging.glob("*.onnx"):
        if f != served:
            f.unlink()  # one model file per release: nothing stale left to be picked up
    log(f"  staged {served.name}  {served.stat().st_size / 1e6:.1f} MB  in {staging}")

    accept = pipeline(accept_raw)
    clf = OX.OnnxClassifier(staging, max_length=max_length, model_file=served.name)
    got = np.concatenate([clf.logits(accept[i : i + 32]) for i in range(0, len(accept), 32)])
    single = np.concatenate([clf.logits([t]) for t in accept[:32]])
    ref = _torch_logits(model, tok, accept, max_length)
    k = len(label_names(task))
    verdict = acceptance(ref, got, np.asarray(accept_y), k, quantize != "none", single)
    log(
        f"  acceptance on {verdict['n']} validation sentences: macro-F1 torch "
        f"{verdict['macro_f1_torch']:.4f} / onnx {verdict['macro_f1_onnx']:.4f}, label agreement "
        f"{verdict['label_agreement']:.2%}, max logit diff {verdict['max_abs_logit_diff']:.2e}"
    )

    import onnxruntime
    import transformers

    manifest = {
        "task": task,
        "model_file": served.name,
        "sha256": sha256(served),
        "quantization": quantize,
        "preprocessing": preprocessing,
        "max_length": max_length,
        "labels": label_names(task),
        "checkpoint": str(checkpoint),
        "calibration": f"{len(calib_raw)} stratified train sentences"
        if quantize == "static"
        else None,
        "acceptance": verdict,
        "created": stamp,
        "software": {
            "onnxruntime": onnxruntime.__version__,
            "transformers": transformers.__version__,
        },
    }
    (staging / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not verdict["passed"]:
        raise RuntimeError(
            "release blocked — " + "; ".join(verdict["failures"]) + f". Staging kept at {staging}"
        )

    previous = out_dir.parent / f".previous-{out_dir.name}"
    if out_dir.exists():
        if previous.exists():
            shutil.rmtree(previous)
        out_dir.rename(previous)
    staging.rename(out_dir)
    log(
        f"  released -> {out_dir}"
        + (f" (previous kept at {previous})" if previous.exists() else "")
    )
    return manifest
