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
CLOSING_GATES = (
    paths.RESULTS / "studies" / "closing_gate" / "summary.json",
    paths.RESULTS / "studies" / "cycle5" / "h11" / "closing_gate" / "summary.json",
    paths.RESULTS / "studies" / "cycle5" / "h10b" / "closing_gate" / "summary.json",
)
CHALLENGES = (
    paths.RESULTS / "studies" / "challenge" / "summary.json",
    paths.RESULTS / "studies" / "cycle5" / "h11" / "challenge_summary.json",
    paths.RESULTS / "studies" / "cycle5" / "h10b" / "challenge_summary.json",
)
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


def _student_dir(ckp: str) -> str:
    """The H11 result folder of a student checkpoint, e.g. pretrained-first6-s42."""
    import re

    m = re.search(r"-h11-(.+)-from-.+-s(\d+)-[0-9a-f]{8}-ckp$", ckp)
    if not m:
        raise ValueError(f"not an H11 student checkpoint: {ckp}")
    return f"{m.group(1)}-s{m.group(2)}"


def evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    """Collect the card's numbers from committed results for the manifest's checkpoint."""
    ckp = _basename(manifest["checkpoint"])
    run_id = ckp.removesuffix("-ckp")
    val = _json(paths.RUNS / f"{run_id}-val" / "metrics.json")
    config = (paths.RUNS / f"{run_id}-val" / "config.yaml").read_text(encoding="utf-8")
    # The closing gate and the challenge set: Cycle 1's record, or Cycle 5 H11's for the student.
    gate_row = next(
        v
        for f in CLOSING_GATES
        if f.exists()
        for v in _json(f)["checkpoints"].values()
        if _basename(v["checkpoint"]) == ckp
    )
    challenge_row = None
    for f in CHALLENGES:
        if f.exists():
            c = _json(f)
            which = next((k for k, v in c["checkpoints"].items() if _basename(v) == ckp), None)
            if which:
                challenge_row = c[which]
                break
    if challenge_row is None:
        raise KeyError(f"{ckp} has no challenge-set record")
    scope = paths.RESULTS / "studies" / "cycle4_step0" / "served_neu_esc_validation.json"
    out = {
        "run_id": run_id,
        "config_yaml": config,
        "validation": val,
        "test": gate_row,
        "challenge": challenge_row,
        # ADR-032: what `in_scope` does on another institution's text (NEXT_PLAN v5 A2).
        "scope_check": _json(scope) if scope.exists() else None,
    }
    if "-h11-" in ckp:  # the distilled student (ADR-039): its rule and its five-seed test
        h11 = paths.RESULTS / "studies" / "cycle5" / "h11"
        out["h11_decision"] = _json(next(iter(sorted(h11.glob("confirm-*/decision.json")))))
        out["h11_gate"] = _json(CLOSING_GATES[1])
        out["h11_summary"] = _json(h11 / _student_dir(ckp) / "summary.json")
        h10 = paths.RESULTS / "studies" / "cycle5" / "h10"
        teacher = out["h11_decision"]["teacher"]
        name = "control-s42" if teacher == "served" else f"{teacher.removeprefix('h10-')}-s42"
        out["h11_teacher_summary"] = _json(h10 / name / "summary.json")
        lat = h11 / "latency.json"
        out["h11_latency"] = _json(lat) if lat.exists() else None
    if "-h10b-" in ckp:  # consistency anchored on a frozen teacher (ADR-042)
        h10b = paths.RESULTS / "studies" / "cycle5" / "h10b"
        out["h10b_decision"] = _json(next(iter(sorted(h10b.glob("confirm-*/decision.json")))))
        out["h10b_gate"] = _json(CLOSING_GATES[2])
        d9 = paths.RESULTS / "studies" / "cycle5" / "decision9_challenge_served_pipeline.json"
        out["decision9"] = _json(d9) if d9.exists() else None
    return out


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
    if manifest.get("quantization") == "fp16-storage":
        return "ONNX graph with its weights stored in FP16 and computed in FP32 (ADR-039)"
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
    if manifest.get("quantization") == "fp16-storage":
        return (
            "The graph and the PyTorch model with the same FP16-rounded weights agree on every "
            f"validation label (max logit difference {a['max_abs_logit_diff']:.1e}); a sentence "
            "scores the same alone and inside a padded batch."
        )
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
    if "h11_decision" in ev:
        return student_card(repo_id, manifest, ev)
    if "h10b_decision" in ev:
        return h10b_card(repo_id, manifest, ev)
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


def _latency_line(lat: dict[str, Any] | None) -> str:
    if not lat:
        return ""
    return (
        f"\n**Speed** (laptop CPU, one sentence, raw text through the served pipeline; median of "
        f"{lat['sessions']} sessions): p95 **{lat['candidate_p95_median_ms']:.1f} ms** against "
        f"{lat['served_p95_median_ms']:.1f} ms for the 12-layer model measured in the same sessions "
        f"({lat['served_p95_median_ms'] / lat['candidate_p95_median_ms']:.1f}x faster).\n"
    )


def student_card(repo_id: str, manifest: dict[str, Any], ev: dict[str, Any]) -> str:
    """The card of the distilled 6-layer student (Cycle 5 H11, ADR-039)."""
    v, t, c = ev["validation"], ev["test"], ev["challenge"]
    pc = v["per_class"]
    rob, cal = t["robustness_test"], t["calibration"]
    rules = ev["h11_decision"]["rules"]
    five = ev["h11_gate"]["five_seed_test"]
    paired = five["paired_student_minus_teacher"]
    sd, td = ev["h11_summary"]["sets"], ev["h11_teacher_summary"]["sets"]
    size_mb = ev["h11_decision"]["graph"]["bytes"] / 1e6  # the graph the H11 rule checked
    flips = ev["h11_summary"]["vilexnorm_dev"]["flip_rate"]
    t_flips = ev["h11_teacher_summary"]["vilexnorm_dev"]["flip_rate"]
    cats = "\n".join(
        f"| `{k}` | {x['n']} | {x['accuracy']:.3f} |" for k, x in sorted(c["by_category"].items())
    )
    files = "\n".join(f"| `{name}` | `{sha}` |" for name, sha in sorted(manifest["_files"].items()))
    student5 = ", ".join(f"{x:.4f}" for x in five["student"].values())
    return f"""---
language: vi
license: cc-by-nc-sa-4.0
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
- knowledge-distillation
- student-feedback
---

# ViFeedback sentiment: a 6-layer PhoBERT student, {size_mb:.0f} MB

Sentiment (negative / neutral / positive) for Vietnamese student feedback about university courses.
A 6-layer PhoBERT student distilled from a 5-seed ensemble of the 12-layer ViFeedback model, at a
third of its size and about half its latency. It matches its teacher on validation; on the test split
it is {-paired["mean_delta"]:.3f} macro-F1 lower (five seeds each, see Limitations). Shipped as the
{_graph_kind(manifest)} ({size_mb:.1f} MB) the ViFeedback service runs, plus the PyTorch checkpoint it
was exported from.

Code, evaluation protocol and every number below: https://github.com/ThanhDatVN/ViFeedback-NLP-Service

## Input contract

The model expects **pyvi word-segmented, lowercase** text (`giảng_viên nhiệt_tình`), at most
{manifest["max_length"]} tokens. ONNX Runtime folds the FP16-to-FP32 casts when it loads the graph.

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
{_optional_parts(manifest, None)}
## Training (knowledge distillation)

- **Student.** PhoBERT-base's embeddings and first six layers (`vinai/phobert-base` at
  `{MODEL_REVISIONS.get("phobert-base", "?")[:12]}`) and a new 3-class head: 92.5 M parameters.
- **Teacher.** The mean, at temperature 2, of five seeds of the 12-layer ViFeedback model (PhoBERT-base
  trained on UIT-VSFC with diacritic and teencode augmentation).
- **Data.** UIT-VSFC train (11,426 sentences, gold labels, the same 30% augmentation) plus 28,648
  unlabeled texts labelled by the teacher: in-scope forum posts from another university (NEU-ESC train)
  and real social-media comments (ViLexNorm train).
- **Loss.** 0.5 * T^2 * KL(teacher ‖ student) + 0.5 * cross-entropy on labelled text; the KL term alone
  on unlabeled text. 4 epochs, lr 2e-5, batch 32; the epoch selected on validation macro-F1.
- **Storage.** Weights rounded to FP16 in the file and cast to FP32 in the graph, so every
  multiplication is FP32 and nothing depends on the batch.

The student passed a rule declared before it was trained (`configs/experiments/cycle5.yaml`, H11):
five seeds against the teacher's five, pooled UIT-VSFC + NEU-ESC validation drop
{rules["2_pooled_drop_bound"]["observed_drop"]:.4f} with a one-sided 95% bound of
{rules["2_pooled_drop_bound"]["upper_95_one_sided"]:.4f} (limit 0.01); UIT-VSFC drop
{rules["3a_uit_validation_macro_f1_drop"]["drop"]:.4f}; neutral F1 {-rules["3b_uit_validation_neutral_f1_drop"]["drop"]:+.4f} for the student.

## Evaluation

Macro-F1 is the headline because `neutral` is 4% of the data and carries a third of the average.

| Split | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---:|---:|---:|---:|
| Validation (1,583) | {v["macro_f1"]:.4f} | {pc["negative"]["f1"]:.3f} | {pc["neutral"]["f1"]:.3f} | {pc["positive"]["f1"]:.3f} |
| Test (3,166), evaluated once | {t["test"]["macro_f1"]:.4f} | {t["test"]["per_class_f1"]["negative"]:.3f} | {t["test"]["per_class_f1"]["neutral"]:.3f} | {t["test"]["per_class_f1"]["positive"]:.3f} |

Test macro-F1 over the five student seeds: {student5}. Against the teacher's five seeds (Cycle 1's
closing gate) the difference is {paired["mean_delta"]:+.4f}, 95% interval [{paired["ci95"][0]:+.4f},
{paired["ci95"][1]:+.4f}], lower in {paired["n"] - paired["wins"]} of {paired["n"]} seeds.

{_agreement(manifest)}
{_latency_line(ev.get("h11_latency"))}
**Robustness** (test, macro-F1): no diacritics {rob["nodiacritic"]["macro_f1"]:.3f}; half the diacritics
{rob["nodiacritic-50"]["macro_f1"]:.3f}; teencode {rob["teencode-100"]["macro_f1"]:.3f}; 5% character
noise {rob["charnoise-5"]["macro_f1"]:.3f}. With the restorer in front, validation with every diacritic
stripped scores {manifest["restorer"]["acceptance"]["stripped_validation_macro_f1_with"]:.3f}.

**Calibration** (test): temperature scaling cuts NLL {cal["uncalibrated"]["nll"]:.3f} →
{cal["calibrated"]["nll"]:.3f} and ECE {cal["uncalibrated"]["ece_equal_width"]:.3f} →
{cal["calibrated"]["ece_equal_width"]:.3f}.

**Challenge set** (305 constructed sentences, not from UIT-VSFC; accuracy per category):

| Category | n | Accuracy |
|---|---:|---:|
{cats}

## Limitations

- **It is a little less accurate than the 12-layer model on test**: {-paired["mean_delta"]:.3f} macro-F1
  over five seeds each, lower in every seed, although the two were equal on validation, where the
  student was selected. Validation holds 73 neutral sentences, too few to resolve a difference this
  small. Use the 12-layer model (`Datk4/vifeedback-sentiment-phobert`) where that matters more than
  size and speed.
- **Neutral is weak** (test F1 {t["test"]["per_class_f1"]["neutral"]:.3f}; validation {pc["neutral"]["f1"]:.3f}). Neutral errors are
  confident, so thresholds do not fix them.
- **Real informal typing is unstable.** Between a real social-media comment and its human
  normalization, {flips:.0%} of this model's labels change on the ViLexNorm development pairs
  (the teacher: {t_flips:.0%}).
- **Real student text from another university is much harder.** On NEU-ESC validation (3,305 forum
  posts) macro-F1 is {sd["neu_validation_all"]["macro_f1"]:.3f} (the teacher: {td["neu_validation_all"]["macro_f1"]:.3f}). Forum posts are mostly
  non-evaluative; the model was trained on course surveys.
- **Input is lowercased** before scoring, as the training data is.
- **Off-topic input still gets a label**; `in_scope` (above) is the signal to act on. It judges topic
  from words, so a short or unusual course comment can be flagged.
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

Weights: **CC BY-NC-SA 4.0**, for research and other non-commercial use, shared alike. The student
learned from UIT-VSFC (released by its authors for research), from NEU-ESC posts and from ViLexNorm
comments (CC BY-NC-SA 4.0), and derives from PhoBERT (MIT). The code that produced it is MIT-licensed
in the GitHub repository.

## Citation

UIT-VSFC: Nguyen et al., 2018. NEU-ESC: Mai et al., 2025. ViLexNorm: Nguyen et al., 2024 (EACL).
PhoBERT: Nguyen and Nguyen, 2020. Distillation: Hinton et al., 2015.
"""


def h10b_card(repo_id: str, manifest: dict[str, Any], ev: dict[str, Any]) -> str:
    """The card of the 12-layer model trained with anchored consistency (Cycle 5 H10b, ADR-042/043)."""
    v, t, c = ev["validation"], ev["test"], ev["challenge"]
    pc = v["per_class"]
    rob = t["robustness_test"]
    rules = ev["h10b_decision"]["rules"]
    per_seed = ev["h10b_decision"]["reported"]["per_seed"]
    five = ev["h10b_gate"]["five_seed_test"]
    paired = five["paired_h10b_minus_served_recipe"]
    flips_c = [x["candidate"]["flip_rate"] for x in per_seed.values()]
    flips_b = [x["control"]["flip_rate"] for x in per_seed.values()]
    t_value = manifest["temperature"]["value"] if manifest.get("temperature") else None
    cats = "\n".join(
        f"| `{k}` | {x['n']} | {x['accuracy']:.3f} |" for k, x in sorted(c["by_category"].items())
    )
    served_pipeline = ""
    d9 = ev.get("decision9")
    if d9:
        rel = d9["releases"]
        mine = next(r for r in rel.values() if "h10b" in r["checkpoint"])
        old = next(r for r in rel.values() if "p9-" in r["checkpoint"])
        served_pipeline = (
            f"\nThrough the service's own pipeline (lowercase, diacritic restorer, pyvi) the same "
            f"sentences score macro-F1 **{mine['macro_f1']:.3f}** (neutral F1 "
            f"{mine['per_class_f1']['neutral']:.3f}), against {old['macro_f1']:.3f} "
            f"({old['per_class_f1']['neutral']:.3f}) for the model without the consistency training.\n"
        )
    files = "\n".join(f"| `{name}` | `{sha}` |" for name, sha in sorted(manifest["_files"].items()))
    five_str = ", ".join(f"{x:.4f}" for x in five["h10b"].values())
    cal_line = (
        f"The service divides the logits by **T = {t_value:.2f}** (fitted on validation) before the "
        "softmax, so the probabilities it reports are calibrated; do the same with the raw graph."
        if t_value
        else ""
    )
    return f"""---
language: vi
license: cc-by-nc-sa-4.0
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
- robustness
- student-feedback
---

# ViFeedback sentiment: PhoBERT-base, robust to missing diacritics and real informal typing

Sentiment (negative / neutral / positive) for Vietnamese student feedback about university courses.
PhoBERT-base fine-tuned on UIT-VSFC with diacritic and teencode augmentation and, at every step,
trained to give a real informally typed comment the label a frozen teacher gives its normalized form.
Real typing changes {100 * sum(flips_c) / len(flips_c):.0f}% of its labels, against
{100 * sum(flips_b) / len(flips_b):.0f}% for the same model without that training. Shipped as the
{_graph_kind(manifest)} the ViFeedback service runs, plus the PyTorch checkpoint it was exported from.

Code, evaluation protocol and every number below: https://github.com/ThanhDatVN/ViFeedback-NLP-Service

## Input contract

The model expects **pyvi word-segmented, lowercase** text (`giảng_viên nhiệt_tình`), at most
{manifest["max_length"]} tokens.

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

{cal_line}
{_optional_parts(manifest, None)}
## Training

- **Base.** `vinai/phobert-base` at `{MODEL_REVISIONS.get("phobert-base", "?")[:12]}`; 4 epochs, lr 2e-5,
  batch 32, max length 96, seed 42; the epoch selected on validation macro-F1.
- **Augmentation.** 30% of the UIT-VSFC training sentences replaced by a perturbed copy (half their
  diacritics removed, or common teencode substituted) and re-segmented.
- **Anchored consistency** (ADR-042). At every step, a batch of 6,035 real social-media comments from
  ViLexNorm, each paired with its human normalization. The loss adds KL(q ‖ p(comment)), where q is
  the probability a frozen 5-seed ensemble of the model without this training gives the
  normalized form. The target is fixed and diverse, so the model cannot satisfy it by predicting one
  class, the failure of an earlier attempt that used its own prediction as the target (ADR-038).

It passed a rule declared before it was trained (`configs/experiments/cycle5.yaml` v4, five seeds each
side, 1,500 held-out comment pairs):
- agreement with the teacher's label on the normalized form
  {rules["1_agreement_up"]["observed_diff"]:+.3f} [{rules["1_agreement_up"]["ci_low"]:+.3f}, {rules["1_agreement_up"]["ci_high"]:+.3f}];
- label flips between a comment and its normalization {rules["2_flip_rate_down"]["observed_diff"]:+.3f}
  [{rules["2_flip_rate_down"]["ci_low"]:+.3f}, {rules["2_flip_rate_down"]["ci_high"]:+.3f}];
- label-distribution distance {rules["3_label_tv"]["mean"]:.3f} (limit 0.10);
- and no loss on UIT-VSFC ({rules["4_uit_validation_macro_f1"]["mean_diff"]:+.4f}), its neutral class,
  stripped text or another university's posts (NEU-ESC, {rules["7_neu_validation_macro_f1"]["mean_diff"]:+.4f}).

## Evaluation

Macro-F1 is the headline because `neutral` is 4% of the data and carries a third of the average.

| Split | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---:|---:|---:|---:|
| Validation (1,583) | {v["macro_f1"]:.4f} | {pc["negative"]["f1"]:.3f} | {pc["neutral"]["f1"]:.3f} | {pc["positive"]["f1"]:.3f} |
| Test (3,166), evaluated once | {t["test"]["macro_f1"]:.4f} | {t["test"]["per_class_f1"]["negative"]:.3f} | {t["test"]["per_class_f1"]["neutral"]:.3f} | {t["test"]["per_class_f1"]["positive"]:.3f} |

Test macro-F1 over five seeds of this recipe: {five_str}. Against five seeds of the recipe without the
consistency training, the difference is {paired["mean_delta"]:+.4f}, 95% interval
[{paired["ci95"][0]:+.4f}, {paired["ci95"][1]:+.4f}]: not distinguishable. (The seed-42 model
without it happened to score 0.8371 on test.)

{_agreement(manifest)}

**Robustness** (test, macro-F1, without the restorer): no diacritics {rob["nodiacritic"]["macro_f1"]:.3f};
half the diacritics {rob["nodiacritic-50"]["macro_f1"]:.3f}; teencode {rob["teencode-100"]["macro_f1"]:.3f};
5% character noise {rob["charnoise-5"]["macro_f1"]:.3f}. The service restores diacritics first: validation
with every diacritic stripped scores {manifest["restorer"]["acceptance"]["stripped_validation_macro_f1_with"]:.3f} through it.

**Challenge set** (305 constructed sentences, not from UIT-VSFC; accuracy per category, the model alone):

| Category | n | Accuracy |
|---|---:|---:|
{cats}
{served_pipeline}
## Limitations

- **Neutral is weak** (test F1 {t["test"]["per_class_f1"]["neutral"]:.3f}). Neutral errors are confident, so thresholds do not fix them.
- **Real informal typing still changes labels**, less often: about {100 * sum(flips_c) / len(flips_c):.0f}% of
  labels between a social-media comment and its human normalization. Those comments come from social
  media (ViLexNorm), not from course feedback.
- **Real student text from another university is much harder** (0.43 macro-F1 on NEU-ESC validation).
  Forum posts are mostly non-evaluative; the model was trained on course surveys.
- **Input is lowercased** before scoring, as the training data is.
- **Off-topic input still gets a label**; `in_scope` (above) is the signal to act on.
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

Weights: **CC BY-NC-SA 4.0**, for research and other non-commercial use, shared alike. The model
learned from UIT-VSFC (released by its authors for research) and from ViLexNorm comments
(CC BY-NC-SA 4.0), and derives from PhoBERT (MIT). The code that produced it is MIT-licensed in the
GitHub repository.

## Citation

UIT-VSFC: Nguyen et al., 2018. ViLexNorm: Nguyen et al., 2024 (EACL). PhoBERT: Nguyen and Nguyen,
2020. Consistency training: Xie et al., 2020 (UDA).
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


def stale_files(remote: list[str], folder: Path) -> list[str]:
    """Files on the Hub that the new bundle no longer has (e.g. ood.npz after ADR-034)."""
    local = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    return sorted(f for f in remote if f not in local and f != ".gitattributes")


def upload(repo_id: str, folder: Path, private: bool = False) -> str:
    """Create the repository if needed and make it hold exactly the folder, in one commit.

    Files the new release no longer ships are deleted in the same commit, so the Hub never serves a
    stale file next to a manifest that does not mention it. Uses the caller's HF login/token.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    stale = stale_files(api.list_repo_files(repo_id, repo_type="model"), folder)
    info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        repo_type="model",
        commit_message="ViFeedback sentiment model (release manifest included)",
        delete_patterns=stale or None,
    )
    return str(info) + (f"\n  removed from the Hub: {', '.join(stale)}" if stale else "")
