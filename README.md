# ViFeedback NLP Service

Vietnamese feedback understanding: **sentiment** (3-class) and **topic** (4-class) classification on
[UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback), benchmarking classical
baselines against fine-tuned PhoBERT, with an explicit focus on **CPU inference cost** and on
**where Vietnamese models actually break** (negation, teencode, missing diacritics).

> **Status:** planning complete, implementation not started.
> Start at [docs/ROADMAP.md](docs/ROADMAP.md).

---

## Why this project is framed the way it is

Most public work on UIT-VSFC reports ~92–94% F1 and stops. Those are **weighted** F1 numbers on a
dataset that is 4.3% neutral — a model that never predicts `neutral` at all still scores ~93% weighted.
This project reports **macro-F1 as the headline metric**, which makes the minority class impossible
to hide, and pairs it with a latency budget so the result is a *deployable* artifact, not a notebook cell.

Three claims this project sets out to test (any of them may come back negative — that is fine, and
a well-documented negative result is kept):

| # | Hypothesis | Why it matters |
|---|------------|----------------|
| H1 | Macro-F1 is the binding constraint on UIT-VSFC, and the gap between TF-IDF and PhoBERT is far larger in macro-F1 than in weighted F1. | Turns a crowded benchmark into an honest, differentiated result. |
| H2 | VnCoreNLP word segmentation — mandated by PhoBERT's own model card — is not necessary for this task, and is a *larger* share of p95 latency than the transformer itself. | Ablation and deployment become the same experiment. See [RESEARCH_NOTES](docs/RESEARCH_NOTES.md#h2-prior-art). |
| H3 | INT8 dynamic quantization may be *slower* than FP32 on a consumer laptop CPU without AVX512-VNNI. | Forces a real measurement instead of a repeated blog-post claim. |

---

## Documents

| Document | Contents |
|---|---|
| [docs/ROADMAP.md](docs/ROADMAP.md) | Objectives, 8-phase plan, week-by-week schedule, exit gates, risk register, repo layout |
| [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md) | The full experiment matrix, run-ID scheme, baseline ladder, result tables to fill |
| [docs/DATA_CARD.md](docs/DATA_CARD.md) | Dataset provenance, splits, class distribution, known limitations, validation checks |
| [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) | Metrics, seed policy, significance testing, latency harness spec, error taxonomy, software test strategy |
| [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) | Dataset and model landscape, prior results, tooling decisions, sources |

---

## Stack

PyTorch · Transformers · PhoBERT · VnCoreNLP · ONNX Runtime · FastAPI · Hugging Face Hub · Docker

**Compute:** personal laptop (CPU — all benchmarking, the API, and classical baselines) +
Google Colab (GPU — all transformer fine-tuning). The split is deliberate: latency numbers are only
meaningful on fixed, documented hardware, so the laptop is the *reference machine* and never runs training.

---

## CV snippet — fill after Gate G6

> Fine-tuned PhoBERT for Vietnamese sentiment/topic classification on 16k+ labeled sentences;
> raised macro-F1 from `[baseline]` to `[result]` and reduced CPU p95 latency by `[x]`% with `[method]`.

Alternative framings and the rules for filling the placeholders honestly are in
[ROADMAP.md § 9](docs/ROADMAP.md#9-cv-snippet-and-claim-discipline).

---

## CV-readiness checklist · 0/10

Each item is owned by exactly one phase gate; see [ROADMAP.md § 7](docs/ROADMAP.md#7-checklist-to-gate-mapping).

- [ ] 1. Problem definition + data card + split — *Gate G0*
- [ ] 2. TF-IDF baseline — *Gate G1*
- [ ] 3. PhoBERT fine-tuning & reproduction — *Gate G2*
- [ ] 4. Macro/per-class F1 + confusion matrix — *Gate G1 (harness), G2 (first full report)*
- [ ] 5. Word segmentation / preprocessing ablation — *Gate G3*
- [ ] 6. ≥30 error cases categorized by linguistic features — *Gate G5*
- [ ] 7. ONNX / quantization benchmark — *Gate G6*
- [ ] 8. API + Docker + CI — *Gate G7*
- [ ] 9. HF model card + README — *Gate G7*
- [ ] 10. Reproducible end-to-end run from a clean clone — *Gate G7*

---

## License and citation

Dataset usage follows the UIT NLP Group's terms — see [docs/DATA_CARD.md § Licensing](docs/DATA_CARD.md#11-licensing-and-citation).
Cite Nguyen et al. (KSE 2018) for UIT-VSFC and Nguyen & Nguyen (EMNLP Findings 2020) for PhoBERT.
