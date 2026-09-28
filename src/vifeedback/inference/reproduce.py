"""Reproduce the published model's headline number from a clean clone (ROADMAP S10, NEXT_PLAN v5 F2).

Downloads the release from the Hugging Face Hub, checks every file against the release's
`SHA256SUMS` and the manifest, and scores UIT-VSFC validation through the service's own pipeline
(`serving.pipeline`: lowercase, restorer, pyvi, ONNX). Passes when the macro-F1 equals the one the
release gate recorded in the manifest. Needs no GPU, no PyTorch and no local checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vifeedback import paths

# The files serving needs; the PyTorch checkpoint is not downloaded.
SERVING_FILES = (
    "manifest.json",
    "SHA256SUMS",
    "vocab.txt",
    "bpe.codes",
    "added_tokens.json",
    "tokenizer_config.json",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(repo_id: str, revision: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Fetch the serving files of a release into models/hub/<name>/ and verify them."""
    from huggingface_hub import hf_hub_download

    dst = paths.MODELS / "hub" / repo_id.replace("/", "__")
    dst.mkdir(parents=True, exist_ok=True)

    def get(name: str) -> Path:
        return Path(hf_hub_download(repo_id, name, revision=revision, local_dir=dst))

    for name in SERVING_FILES:
        get(name)
    manifest = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    extra = [manifest["model_file"]] + [
        manifest[k]["file"] for k in ("restorer", "ood", "scope") if manifest.get(k)
    ]
    for name in extra:
        get(name)

    sums: dict[str, str] = {}
    for line in (dst / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split("  ", 1)
            sums[name] = sha
    checked = {}
    for name in [*SERVING_FILES[2:], "manifest.json", *extra]:
        if name in sums:
            ok = _sha256(dst / name) == sums[name]
            checked[name] = ok
            if not ok:
                raise ValueError(f"{name} does not match SHA256SUMS")
    if _sha256(dst / manifest["model_file"]) != manifest["sha256"]:
        raise ValueError(f"{manifest['model_file']} does not match the manifest")
    return dst, {"files_checked": len(checked), "model_sha256": manifest["sha256"]}


def evaluate(model_dir: Path) -> dict[str, Any]:
    """Validation macro-F1 through the served pipeline, from raw corpus text."""
    from vifeedback.data.loader import load
    from vifeedback.evaluation import metrics as M
    from vifeedback.inference.onnx_export import OnnxClassifier
    from vifeedback.preprocess.segment import get_segmenter
    from vifeedback.preprocess.variants import VARIANTS
    from vifeedback.serving import pipeline as SP

    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    clf = OnnxClassifier(model_dir, max_length=manifest.get("max_length", 96))
    restorer = (
        SP.load_restorer(model_dir, manifest["restorer"]) if manifest.get("restorer") else None
    )
    ood = SP.load_ood(model_dir, manifest["ood"], clf) if manifest.get("ood") else None
    seg = get_segmenter(VARIANTS[manifest["preprocessing"]][0])

    dv = load("validation")
    texts, y = dv.sentence.tolist(), dv.sentiment.to_numpy()
    pred = np.concatenate(
        [
            SP.score(clf, SP.prepare(texts[i : i + 64], restorer, seg), ood)[0]
            for i in range(0, len(texts), 64)
        ]
    )
    macro = M.macro_f1(y, pred, len(manifest["labels"]))
    expected = manifest["acceptance"]["macro_f1_onnx"]
    return {
        "validation_rows": len(y),
        "macro_f1": macro,
        "manifest_macro_f1": expected,
        "abs_diff": abs(macro - expected),
        "reproduced": abs(macro - expected) <= 1e-4,
    }


def reproduce(repo_id: str, revision: str | None = None) -> dict[str, Any]:
    """Download, verify, evaluate; with the wall time of each step."""
    from vifeedback.data.loader import fetch

    t0 = time.perf_counter()
    model_dir, verified = download(repo_id, revision)
    t1 = time.perf_counter()
    if not (paths.DATA_RAW / "validation.parquet").exists():
        fetch()
    t2 = time.perf_counter()
    result = evaluate(model_dir)
    t3 = time.perf_counter()
    return {
        "repo_id": repo_id,
        "revision": revision or "main",
        **verified,
        **result,
        "seconds": {
            "download_and_verify": round(t1 - t0, 1),
            "data": round(t2 - t1, 1),
            "evaluate": round(t3 - t2, 1),
            "total": round(t3 - t0, 1),
        },
    }
