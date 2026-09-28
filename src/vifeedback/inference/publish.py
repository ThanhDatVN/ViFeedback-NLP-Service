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
    scope = paths.RESULTS / "studies" / "cycle4_step0" / "served_neu_esc_validation.json"
    return {
        "run_id": run_id,
        "config_yaml": config,
        "validation": val,
        "test": gate_row,
        "challenge": challenge[which],
        # ADR-032: what `in_scope` does on another institution's text (NEXT_PLAN v5 A2).
        "scope_check": _json(scope) if scope.exists() else None,
    }


def _optional_parts(manifest: dict[str, Any], scope_check: dict[str, Any] | None = None) -> str:
    """Model-card section for the restorer and the out-of-scope score, when the release has them."""
    items = []
    r = manifest.get("restorer")
    if r:
        a = r["acceptance"]
        items.append(
            f"- `{r['file']}`: a diacritic restorer (word-bigram Viterbi over UIT-VSFC train text). "
            "The service applies it after lowercasing and before segmentation, and only to "
            f"essentially unaccented sentences (accented share below {r['threshold']:.3f}). "
            "Validation with the diacritics stripped: macro-F1 "
            f"{a['stripped_validation_macro_f1_without']:.3f} → "
            f"{a['stripped_validation_macro_f1_with']:.3f}; clean validation labels changed: "
            f"{a['validation_labels_changed']}. "
            "Load it with `vifeedback.preprocess.diacritics.Restorer`."
        )
    o = manifest.get("ood")
    if o:
        a = o["acceptance"]
        items.append(
            f"- `{o['file']}`: an out-of-scope score over the graph's `features` output (last-layer "
            "`<s>` vector): the negative Mahalanobis distance to the nearest class mean (shared "
            f"covariance, fitted on train). Below {o['threshold']:.1f} the input is out of scope; "
            f"the threshold keeps {o['keeps_validation']:.0%} of validation. On "
            f"{a['off_topic_rows']} real off-topic forum posts (NEU-ESC spam, news, jobs, club "
            f"events) against validation: AUROC {a['auroc']:.3f}, "
            f"{a['out_of_scope_caught']:.0%} flagged. The service reports it as `in_scope` and "
            "`scope_score` next to the label and never refuses an input."
        )
        if scope_check:
            items[-1] += (
                " **It measures resemblance to the training surveys, not topic** (ADR-032): on "
                "another university's forum posts it flags "
                f"{scope_check['in_scope_topics']['flagged']:.0%} of the posts that are in scope, and "
                "separates in-scope from off-topic posts there only at AUROC "
                f"{scope_check['within_source_auroc_in_scope_vs_off_topic']:.3f}. Do not use it to "
                "filter another institution's feedback."
            )
    sc = manifest.get("scope")
    if sc:
        a = sc["acceptance"]
        items.append(
            f"- `{sc['file']}`: a topic-aware scope detector (ADR-034): logistic regression on TF-IDF "
            f"word unigrams and bigrams of the model input ({sc['terms']:,} terms), trained to tell "
            "course-related student text (UIT-VSFC, and in-scope posts of another university's "
            "forum, NEU-ESC) from spam, news, job and club posts. Below "
            f"{sc['threshold']:.3f} the input is out of scope. On NEU-ESC test it separates in-scope "
            f"from off-topic posts at AUROC {a['1_within_source_auroc']:.3f}, flags "
            f"{a['2_in_scope_flagged']:.0%} of in-scope posts and catches "
            f"{a['3_off_topic_caught']:.0%} of off-topic ones; it flags "
            f"{a['4_uit_validation_flagged']:.1%} of UIT-VSFC validation. Served by a numpy "
            "re-implementation checked equal to scikit-learn. The service reports it as `in_scope` "
            "and `scope_score` next to the label and never refuses an input."
        )
    if not items:
        return ""
    head = (
        "\n## Optional parts of the release\n\n"
        "Each is declared in `manifest.json` with a SHA-256; the service refuses to start on a "
        "mismatch.\n\n"
    )
    return head + "\n".join(items) + "\n"


def _graph_kind(manifest: dict[str, Any]) -> str:
    if manifest.get("quantization") == "careful":
        return (
            "INT8 ONNX graph (per-channel dynamic quantization, classifier head and last two "
            "encoder layers kept in FP32; S5-prime, ADR-035)"
        )
    if manifest.get("quantization") in ("dynamic", "static"):
        return f"{manifest['quantization']} INT8 ONNX graph"
    return "ONNX FP32 graph"


def _agreement(manifest: dict[str, Any]) -> str:
    a = manifest["acceptance"]
    if manifest.get("quantization") in (None, "none"):
        return (
            "ONNX and PyTorch agree on every validation label (max logit difference "
            f"{a['max_abs_logit_diff']:.1e})."
        )
    where = (
        "UIT-VSFC + NEU-ESC validation"
        if manifest.get("acceptance_set") == "pooled"
        else "UIT-VSFC validation"
    )
    return (
        f"The INT8 graph agrees with PyTorch on {a['label_agreement']:.1%} of {a['n']:,} {where} "
        f"texts; its macro-F1 drop is {a['macro_f1_drop']:+.4f}, with a one-sided 95% upper bound "
        f"of {a['macro_f1_drop_upper95']:.4f} (non-inferiority margin 0.005)."
    )


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
license: cc-by-nc-4.0
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
when text is typed without accents. Shipped as the {_graph_kind(manifest)} the ViFeedback service runs, plus
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
{_optional_parts(manifest, ev.get("scope_check"))}
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

{_agreement(manifest)}

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
- **Contrast sentences** (*A nhưng B*) are handled worse than by the same model trained without
  augmentation: 0.800 vs 0.875 accuracy on 40 constructed sentences, lower in 4 of 5 seeds
  (p = 0.004). Found by a declared rule but not pre-specified. On 414 real forum posts containing a
  contrast (NEU-ESC) the difference did not replicate (0.437 vs 0.440, p = 0.68).
- **Short factual sentences may drift to `negative`** (*môn học có ba tín chỉ* → negative, confidence
  0.5 to 0.7): 0.66 vs 0.75, not consistent across seeds (3 of 5). Treat low-confidence `negative`
  on short inputs with care.
- **Real informal typing is unstable.** On 1,045 real social-media comments (ViLexNorm), about 17% of
  labels change between a comment and its human-normalized version; the augmentation does not reduce
  this. Missing diacritics alone are handled much better (0.66 vs 0.33 on constructed text, 5 of 5
  seeds), and better still with the restorer (NEU-ESC posts stripped of diacritics: 0.270 → 0.374).
- **Input is lowercased** before scoring, as the training data is; capitalized input otherwise changed
  about 1% of labels.
- **Off-topic input still gets a label**, often a confident one; `in_scope` (above) is the signal
  to act on. It judges topic from words, so a short or unusual course comment can be flagged, and
  off-topic text that talks like a course comment can pass.
- **Real student text from another university is much harder.** On 6,613 forum posts from NEU-ESC
  (human labels) this model's macro-F1 is 0.46 (0.43 averaged over five seeds): annotators called 69%
  of posts neutral, this model 27%. Forum posts are mostly non-evaluative; the model was trained
  on course surveys. A zero-shot gpt-4o-mini reaches 0.60 on the same posts, mostly on neutral.
- **One training domain.** Student feedback from one Vietnamese university.
- The suggestion convention is the corpus's: a request for change (*thầy nên…*) is `negative`.

## Files and checksums (SHA-256)

| File | SHA-256 |
|---|---|
{files}

Release manifest: `manifest.json` (acceptance on the full validation set). Training run:
`{ev["run_id"]}`.

## Intended use

Research and experiments on Vietnamese student feedback about courses: aggregate sentiment over
many comments, error analysis, robustness studies. It is not meant for decisions about an individual
student or lecturer, and its labels follow UIT-VSFC's annotation guide (a request for change is
`negative`), which other institutions' annotators may not share.

## Licence

Weights: **CC BY-NC 4.0**, for research and other non-commercial use. They are trained on UIT-VSFC,
which its authors release for research purposes, and derive from PhoBERT (MIT). The code that
produced them is MIT-licensed in the GitHub repository.

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
    optional = [manifest[k]["file"] for k in ("restorer", "ood", "scope") if manifest.get(k)]
    for name in (manifest["model_file"], "manifest.json", *_TOKENIZER_FILES, *optional):
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
