# ViFeedback NLP Service

Vietnamese feedback understanding: **sentiment** (3-class) and **topic** (4-class) classification on
[UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback), benchmarking classical
baselines against fine-tuned PhoBERT, with an explicit focus on **CPU inference cost** and on
**where Vietnamese models actually break** (negation, teencode, missing diacritics).

> **Status:** Gates G0–G3 complete · 47 runs · 5/10 checklist · zero external GPU used ·
> one published conclusion falsified (ADR-012) and one of our own retracted (ADR-015).
> Next: Phase 4 improvement ladder.
> Progress, open problems and next experiments: **[docs/STATUS.md](docs/STATUS.md)**.

---

## Why this project is framed the way it is

Most public work on UIT-VSFC reports ~92–94% F1 and stops. Those are **weighted** F1 numbers on a
dataset that is 4.3% neutral — a model that never predicts `neutral` at all still scores ~93% weighted.
This project reports **macro-F1 as the headline metric**, which makes the minority class impossible
to hide, and pairs it with a latency budget so the result is a *deployable* artifact, not a notebook cell.

Three claims this project sets out to test (any of them may come back negative — that is fine, and
a well-documented negative result is kept):

| # | Hypothesis | Status |
|---|------------|--------|
| H1 | Macro-F1 is the binding constraint on UIT-VSFC. | **Supported.** A perfect-except-neutral classifier scores 0.947 accuracy / 0.922 weighted F1 / **0.649 macro-F1** on the real test split. The measured baseline gap is 0.906 weighted vs 0.782 macro. |
| H2 | VnCoreNLP word segmentation is unnecessary here, and costs more p95 latency than the transformer itself. | **FALSIFIED, both halves.** Segmentation is worth **+0.023 macro-F1** (t=8.58, p=0.001, 5/5 seeds) and costs **0.6 ms p95** — 1.2% of the model's 50.8 ms. The published "unnecessary" result replicates on *accuracy* (+0.96 pp) and collapses on *macro-F1* (+2.34 pp) and *neutral F1* (+5.42 pp). |
| H3 | INT8 dynamic quantization may be *slower* than FP32 on a consumer laptop CPU without AVX512-VNNI. | **Likely.** Reference CPU measured: AMD Ryzen 5 6600H, `avx2=true`, **`avx512_vnni=false`, `avx_vnni=false`**. |

### Results so far — dev split

| Task | Model | macro-F1 | weighted F1 | minority-class F1 |
|---|---|---|---|---|
| Sentiment | TF-IDF word+char + cross-fitted priors | 0.7708 | 0.906 | neutral 0.497 |
| Sentiment | PhoBERT-base, raw, 5 seeds | 0.8436 ± 0.0079 | 0.9427 | neutral 0.614 ± 0.022 |
| Sentiment | **PhoBERT-base + segmentation, 5 seeds** | **0.8670 ± 0.0072** | 0.9529 | neutral **0.668 ± 0.018** |
| Topic | TF-IDF LinearSVC *(honest best)* | 0.768 | 0.874 | others 0.479 |
| Topic | **PhoBERT-base, 5 seeds** | **0.7971 ± 0.0018** | 0.8889 | others **0.568 ± 0.011** |

PhoBERT's advantage is concentrated almost entirely in the minority class: weighted F1 moves +0.037,
macro-F1 +0.062, **neutral F1 +0.117**. Read only through weighted F1, the transformer would look
barely worth the GPU. Two results that cut the other way are reported rather than dropped: TF-IDF
**beats** PhoBERT on the `facility` topic (0.921 vs 0.905), and the topic lift (+0.023) is a quarter
of the sentiment lift.

Full ladder and analysis: [EXPERIMENT_MATRIX § 5.1](docs/EXPERIMENT_MATRIX.md#51-baseline-ladder--measured-at-gate-g1-dev-split-seed-42).
Three pre-registered predictions were **falsified** at G1 and the success criteria revised in ADR-008 —
see the [scorecard](docs/EXPERIMENT_MATRIX.md#pre-registration-scorecard--gate-g1).

---

## Documents

| Document | Contents |
|---|---|
| [docs/ROADMAP.md](docs/ROADMAP.md) | Objectives, 8-phase plan, week-by-week schedule, exit gates, risk register, repo layout |
| [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md) | The full experiment matrix, run-ID scheme, baseline ladder, result tables to fill |
| [docs/DATA_CARD.md](docs/DATA_CARD.md) | Dataset provenance, splits, class distribution, known limitations, validation checks |
| [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) | Metrics, seed policy, significance testing, latency harness spec, error taxonomy, software test strategy |
| [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) | Dataset and model landscape, prior results, tooling decisions, sources |
| **[docs/STATUS.md](docs/STATUS.md)** | **Progress, open problems, next experiments, compute plan** |
| [docs/BENCHMARK_COMPARISON.md](docs/BENCHMARK_COMPARISON.md) | Where we stand against published UIT-VSFC results, and what is not yet claimable |
| [docs/PROPOSALS.md](docs/PROPOSALS.md) | Techniques, models and workflow changes, each anchored to a measurement |
| [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) | Executed EDA — 23 cells, 4 figures, every decision traced to an ADR |
| [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md) | Step-by-step for the two models that exceed the 4.29 GB laptop GPU |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log (ADR-001 … ADR-014) — every plan correction forced by measurement |

## Reproducing what exists

```bash
pip install -e ".[dev]"
vifeedback data fetch        # -> data/raw/*.parquet + SHA256 manifest
vifeedback data report       # -> results/data_report.json + figures
pytest -q                    # 62 tests
vifeedback baseline run --task sentiment
vifeedback baseline run --task topic
vifeedback baseline registry
```

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

## CV-readiness checklist · 5/10

Each item is owned by exactly one phase gate; see [ROADMAP.md § 7](docs/ROADMAP.md#7-checklist-to-gate-mapping).

- [x] 1. Problem definition + data card + split — *Gate G0* ✅
- [x] 2. TF-IDF baseline — *Gate G1* ✅
- [x] 3. PhoBERT fine-tuning & reproduction — *Gate G2* ✅ (5 seeds, both tasks)
- [x] 4. Macro/per-class F1 + confusion matrix — *Gate G1* ✅ (harness + 22 runs)
- [x] 5. Word segmentation / preprocessing ablation — *Gate G3* ✅ (4 conditions × 5 seeds + latency)
- [ ] 6. ≥30 error cases categorized by linguistic features — *Gate G5*
- [ ] 7. ONNX / quantization benchmark — *Gate G6*
- [ ] 8. API + Docker + CI — *Gate G7*
- [ ] 9. HF model card + README — *Gate G7*
- [ ] 10. Reproducible end-to-end run from a clean clone — *Gate G7*

---

## License and citation

Dataset usage follows the UIT NLP Group's terms — see [docs/DATA_CARD.md § Licensing](docs/DATA_CARD.md#11-licensing-and-citation).
Cite Nguyen et al. (KSE 2018) for UIT-VSFC and Nguyen & Nguyen (EMNLP Findings 2020) for PhoBERT.
