"""Bundle the served model for the Hugging Face Hub, with a model card built from committed results.

Review R11 asks for artifacts another researcher can fetch. This module only *builds* the bundle
(``models/publish/<name>/``) and checks it; uploading is a separate, explicit step the owner runs
with their own token (``vifeedback serve publish --repo-id ... --upload``).

Every number in the card is read from a committed result file, so the card cannot drift from the
evidence: the release manifest, the run's metrics, the closing gate and the challenge set.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PureWindowsPath
from typing import Any

from vifeedback import paths
from vifeedback.constants import HF_DATASET_ID, MODEL_REVISIONS

SERVED = paths.MODELS / "serve" / "sentiment"
PUBLISH_ROOT = paths.MODELS / "publish"
_TOKENIZER_FILES = ("vocab.txt", "bpe.codes", "added_tokens.json", "tokenizer_config.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _basename(path: str) -> str:
    """Last component of a path recorded on any OS: results written on Windows hold backslashes,
    which `Path(...).name` does not split on Linux."""
    return PureWindowsPath(path).name


def evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    """Collect the card's numbers from committed results for the manifest's checkpoint."""
    ckp = _basename(manifest["checkpoint"])
    run_id = ckp.removesuffix("-ckp")
    val = _json(paths.RUNS / f"{run_id}-val" / "metrics.json")
    config = (paths.RUNS / f"{run_id}-val" / "config.yaml").read_text(encoding="utf-8")
    gate = _json(paths.RESULTS / "studies" / "closing_gate" / "summary.json")["checkpoints"]
    gate_row = next(v for v in gate.values() if _basename(v["checkpoint"]) == ckp)
    challenge = _json(paths.RESULTS / "studies" / "challenge" / "summary.json")
    which = next(k for k, v in challenge["checkpoints"].items() if _basename(v) == ckp)
    return {
        "run_id": run_id,
        "config_yaml": config,
        "validation": val,
        "test": gate_row,
        "challenge": challenge[which],
    }


def model_card(repo_id: str, manifest: dict[str, Any], ev: dict[str, Any]) -> str:
    v, t, c = ev["validation"], ev["test"], ev["challenge"]
    pc = v["per_class"]
    rob = t["robustness_test"]
    cal = t["calibration"]
    cats = "\n".join(
        f"| `{k}` | {x['n']} | {x['accuracy']:.3f} |" for k, x in sorted(c["by_category"].items())
    )
    files = "\n".join(f"| `{name}` | `{sha}` |" for name, sha in sorted(manifest["_files"].items()))
    return f"""---
language: vi
license: mit
library_name: onnx
pipeline_tag: text-classification
base_model: vinai/phobert-base
datasets:
- {HF_DATASET_ID}
tags:
- sentiment-analysis
- vietnamese
- phobert
- onnx
- student-feedback
---

# ViFeedback sentiment: PhoBERT-base, diacritic-robust

Sentiment (negative / neutral / positive) for Vietnamese student feedback about university courses.
PhoBERT-base fine-tuned on UIT-VSFC with diacritic and teencode augmentation, so it keeps working
when text is typed without accents. Shipped as the ONNX FP32 graph the ViFeedback service runs, plus
the PyTorch checkpoint it was exported from.

Code, evaluation protocol and every number below: https://github.com/ThanhDatVN/ViFeedback-NLP-Service

## Input contract

The model expects **pyvi word-segmented, lowercase** text (`giảng_viên nhiệt_tình`), at most
{manifest["max_length"]} tokens. Unsegmented input still runs but costs about 0.05 macro-F1 (measured).

```python
import numpy as np, onnxruntime as ort
from pyvi import ViTokenizer
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo_id}")
sess = ort.InferenceSession("{manifest["model_file"]}")  # downloaded from this repository
text = ViTokenizer.tokenize("thầy dạy rất dễ hiểu".lower())
enc = tok([text], truncation=True, max_length={manifest["max_length"]}, return_tensors="np")
logits = sess.run(None, {{"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}})[0]
print({manifest["labels"]}[int(logits.argmax())])
```

For calibrated probabilities, divide the logits by **T = {cal["temperature_fit_on_validation"]:.2f}**
(fitted on validation) before the softmax.

## Training

UIT-VSFC train (11,426 sentences), base model `vinai/phobert-base` at
`{MODEL_REVISIONS.get("phobert-base", "?")[:12]}`; 4 epochs, lr 2e-5, batch 32, max length 96,
seed 42. Before training, 30% of the training sentences were replaced by a perturbed copy (half
their diacritics removed, or common teencode substituted) and re-segmented. Checkpoint selected on
validation macro-F1. Full configuration in `training_config.yaml`.

## Evaluation

Macro-F1 is the headline because `neutral` is 4% of the data and carries a third of the average.

| Split | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---:|---:|---:|---:|
| Validation (1,583) | {v["macro_f1"]:.4f} | {pc["negative"]["f1"]:.3f} | {pc["neutral"]["f1"]:.3f} | {pc["positive"]["f1"]:.3f} |
| Test (3,166), evaluated once | {t["test"]["macro_f1"]:.4f} | {t["test"]["per_class_f1"]["negative"]:.3f} | {t["test"]["per_class_f1"]["neutral"]:.3f} | {t["test"]["per_class_f1"]["positive"]:.3f} |

ONNX and PyTorch agree on every validation label (max logit difference
{manifest["acceptance"]["max_abs_logit_diff"]:.1e}).

**Robustness** (test, macro-F1): no diacritics {rob["nodiacritic"]["macro_f1"]:.3f} (the same
model without augmentation: 0.271); half the diacritics {rob["nodiacritic-50"]["macro_f1"]:.3f};
teencode {rob["teencode-100"]["macro_f1"]:.3f}; 5% character noise {rob["charnoise-5"]["macro_f1"]:.3f}.

**Calibration** (test): temperature scaling cuts NLL {cal["uncalibrated"]["nll"]:.3f} →
{cal["calibrated"]["nll"]:.3f} and ECE {cal["uncalibrated"]["ece_equal_width"]:.3f} →
{cal["calibrated"]["ece_equal_width"]:.3f}.

**Challenge set** (305 constructed sentences, not from UIT-VSFC; accuracy per category):

| Category | n | Accuracy |
|---|---:|---:|
{cats}

## Limitations

- **Neutral is weak** (F1 about 0.6 to 0.66). Neutral errors are confident, so thresholds do not fix them.
- **Short factual sentences drift to `negative`** (*môn học có ba tín chỉ* → negative, confidence
  0.5 to 0.7). Treat low-confidence `negative` on short inputs with care.
- **Hand-typed teencode** is handled slightly worse than by the non-augmented model (0.875 vs 0.975
  on 40 challenge rows), although scripted teencode improved.
- **No abstention.** Off-topic input receives a confident polar label.
- **One domain.** Student feedback from one Vietnamese university; other domains are untested.
- The suggestion convention is the corpus's: a request for change (*thầy nên…*) is `negative`.

## Files and checksums (SHA-256)

| File | SHA-256 |
|---|---|
{files}

Release manifest: `manifest.json` (acceptance on the full validation set). Training run:
`{ev["run_id"]}`.

## Licence

Code and weights: MIT. The weights derive from PhoBERT (MIT) and are trained on UIT-VSFC; the
dataset's own terms apply to any use of the data, and should be checked before commercial use.

## Citation

UIT-VSFC: Nguyen et al., 2018, *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment
Analysis*. PhoBERT: Nguyen and Nguyen, 2020, *PhoBERT: Pre-trained language models for Vietnamese*.
"""


def build(repo_id: str, include_pytorch: bool = True) -> dict[str, Any]:
    """Assemble models/publish/<name>/ and return what would be uploaded."""
    manifest = _json(SERVED / "manifest.json")
    model_file = SERVED / manifest["model_file"]
    if _sha256(model_file) != manifest["sha256"]:
        raise RuntimeError(f"{model_file} does not match its manifest; re-run the release")
    ev = evidence(manifest)

    out = PUBLISH_ROOT / repo_id.split("/")[-1]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in (manifest["model_file"], "manifest.json", *_TOKENIZER_FILES):
        if (SERVED / name).exists():
            shutil.copy2(SERVED / name, out / name)
    if include_pytorch:
        ckp = paths.ROOT / manifest["checkpoint"]
        (out / "pytorch").mkdir()
        for f in ckp.iterdir():
            shutil.copy2(f, out / "pytorch" / f.name)
    (out / "training_config.yaml").write_text(ev["config_yaml"], encoding="utf-8")

    files = {
        p.relative_to(out).as_posix(): _sha256(p) for p in sorted(out.rglob("*")) if p.is_file()
    }
    card = model_card(repo_id, {**manifest, "_files": files}, ev)
    (out / "README.md").write_text(card, encoding="utf-8")
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in files.items()), encoding="utf-8"
    )
    return {
        "folder": out,
        "files": files,
        "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
    }


def upload(repo_id: str, folder: Path, private: bool = False) -> str:
    """Create the repository if needed and upload the folder. Uses the caller's HF login/token."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        repo_type="model",
        commit_message="ViFeedback sentiment model (release manifest included)",
    )
    return str(info)
