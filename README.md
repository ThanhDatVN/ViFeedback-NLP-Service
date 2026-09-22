# ViFeedback NLP Service

Vietnamese feedback classification — **sentiment** (3-class) and **topic** (4-class) on
[UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) — benchmarking
TF-IDF against fine-tuned PhoBERT, with a CPU latency budget and a measurement protocol strict
enough to have falsified three of its own predictions.

[![CI](https://github.com/ThanhDatVN/ViFeedback-NLP-Service/actions/workflows/ci.yml/badge.svg)](https://github.com/ThanhDatVN/ViFeedback-NLP-Service/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Result

**Test split, 5 seeds, PhoBERT-base + word segmentation.** The test set was untouched until Gate G4
and every evaluation is logged.

| | Ours | Published best | Honest TF-IDF baseline |
|---|---|---|---|
| **Macro-F1** | **0.8373 ± 0.0031** | 0.8341 *(BamiBERT, 2026)* | 0.7450 |
| Weighted F1 | 0.9391 ± 0.0020 | ~0.94 | 0.8817 |
| Accuracy | 0.9421 ± 0.0023 | 93.86% | 0.8834 |
| Neutral-class F1 | 0.5955 ± 0.0070 | not reported | 0.3530 |

**+0.0032 over the published best is one seed-standard-deviation, with 4 of 5 seeds above it. That
is competitive, not better — the project does not claim state of the art.**

It would have been easy to. On **dev** the same model scored **0.8670**, a 3.3-point gap that reads
like a comfortable win. The dev→test drop is **−0.0298**, and the TF-IDF baseline dropped **−0.026**
in the same direction, so the gap belongs to the split, not the model. Nothing but running the
locked test split would have shown that.

---

## Why macro-F1

UIT-VSFC sentiment is **4.32% neutral**. On the real test split, a classifier that never predicts
`neutral` but is otherwise perfect scores:

| Accuracy | Weighted F1 | **Macro-F1** |
|---|---|---|
| 0.947 | 0.922 | **0.649** |

Three metrics, one model, and only one notices an entire class is missing. Most published UIT-VSFC
results are weighted F1 or accuracy — within reach of a model that learned nothing about neutral.
*(Pinned by `tests/unit/test_metrics.py`, so the claim cannot drift.)*

---

## Three hypotheses, two falsified

| | Hypothesis | Outcome |
|---|---|---|
| **H1** | Macro-F1 is the binding constraint | **Supported.** PhoBERT's advantage over TF-IDF is +0.037 weighted F1 but **+0.24 neutral F1** |
| **H2** | Word segmentation is unnecessary and dominates p95 latency | **Falsified, both halves.** Worth **+0.0234 macro-F1** (t = 8.58, p = 0.0010, 5/5 seeds) and costs **0.31 ms p95** — 0.6% of the model's 50.8 ms |
| **H3** | INT8 may be slower than FP32 without AVX512-VNNI | **Open.** Reference CPU measured: AMD Ryzen 5 6600H, `avx2=true`, **`avx512_vnni=false`** |

### The most interesting finding is about the literature

The published conclusion *"word segmentation is unnecessary for Vietnamese sentiment classification"*
([arXiv:2301.00418](https://arxiv.org/abs/2301.00418)) **replicates exactly on the metric it
reported** and collapses on macro-F1:

| Metric | raw → segmented |
|---|---|
| Accuracy | +0.96 pp ← *"under 1 percentage point", as published* |
| Weighted F1 | +1.02 pp ← *as published* |
| **Macro-F1** | **+2.34 pp** |
| **Neutral F1** | **+5.42 pp** |

A field-level conclusion that turns out to be an artifact of aggregating over a 4% class.

---

## What was got wrong, and corrected

The decision log records four occasions where measurement overturned a plan — including one
retraction of a result already written up as a success.

| ADR | What was claimed | What measurement showed |
|---|---|---|
| [007](docs/DECISIONS.md) | Teencode and missing diacritics are the error-analysis targets | They are **0.16%** and **0.14%** of the corpus. Reframed as robustness targets measured by induced perturbation |
| [008](docs/DECISIONS.md) | The TF-IDF baseline would reach ~0.70 macro-F1 | It reached **0.78**. Three pre-registered ranges falsified, all low; success criteria revised against the measured baseline |
| [012](docs/DECISIONS.md) | Removing segmentation would be the biggest latency win | Segmentation **helps accuracy** and costs 0.6% of p95. The headline hypothesis was wrong |
| [015](docs/DECISIONS.md) | Threshold tuning is "the single largest lever", +0.035 | Cross-fitted, the gain on PhoBERT is **zero**. Retracted — and the correction *raised* the reported lift, because the inflated baseline had been understating the model |

ADR-015 is the one worth reading. The bug was not convenient, and it surfaced because a result that
had already been written up got re-tested.

---

## Quick start

```bash
git clone https://github.com/ThanhDatVN/ViFeedback-NLP-Service.git
cd ViFeedback-NLP-Service
make install          # editable install with dev extras
make data             # fetch UIT-VSFC + run the integrity suite
make test             # 141 fast tests
make report           # Phase 0 profiling + EDA figures
make baseline         # TF-IDF ladder
make train            # fine-tune PhoBERT (needs a GPU; ~5 min/seed on an RTX 3050)
```

`make help` lists every target. Models too large for a 4.29 GB GPU go to Kaggle —
see [KAGGLE_GUIDE](docs/KAGGLE_GUIDE.md).

### Serving

```bash
make export           # checkpoint -> ONNX, optimize, quantize, verify parity
make docker && make docker-run
curl -s localhost:8000/v1/classify \
  -H 'content-type: application/json' \
  -d '{"texts":["giảng viên nhiệt tình với sinh viên ."]}'
```

---

## Repository

```
├── src/vifeedback/
│   ├── data/          loading · integrity · leakage · profiling · lexical analysis
│   ├── preprocess/    normalizers · 4 segmentation backends · materialized variants
│   ├── models/        TF-IDF ladder (B0–B5) · cross-fitted prior tuning
│   ├── training/      fine-tuning loop · losses (focal, logit-adjust, R-Drop, FGM) · seeding
│   ├── evaluation/    metrics · bootstrap · ordinal metrics · run registry
│   ├── inference/     ONNX export · quantization · parity checks · latency harness
│   ├── serving/       FastAPI app · request/response contracts
│   └── cli.py         every experiment is a CLI call
├── notebooks/         EDA and results, both executed with outputs
├── configs/           one YAML per experiment
├── docs/              11 documents — see below
├── tests/             unit · data · contract · integration · packaging guards
├── results/           registry.csv (append-only) · run artifacts · figures
└── Dockerfile · docker-compose.yml · Makefile · .github/workflows/ci.yml
```

| Document | Contents |
|---|---|
| **[STATUS](docs/STATUS.md)** | **Progress, open problems, next experiments, compute plan** |
| [ROADMAP](docs/ROADMAP.md) | Objectives, 8 phases, exit gates, risk register |
| [DECISIONS](docs/DECISIONS.md) | 15 ADRs — every plan correction forced by measurement |
| [DATA_CARD](docs/DATA_CARD.md) | Provenance, splits, distributions, 11 measured limitations |
| [EVALUATION_PROTOCOL](docs/EVALUATION_PROTOCOL.md) | Metrics, seeds, significance, latency harness, error taxonomy |
| [EXPERIMENT_MATRIX](docs/EXPERIMENT_MATRIX.md) | Run-ID scheme, pre-registration scorecard, all result tables |
| [BENCHMARK_COMPARISON](docs/BENCHMARK_COMPARISON.md) | Against published work — and what is not yet claimable |
| [PROPOSALS](docs/PROPOSALS.md) | 6 techniques, 7 models, 6 workflow changes, each anchored to a measurement |
| [RESEARCH_NOTES](docs/RESEARCH_NOTES.md) | Landscape survey, 24 cited sources |
| [KAGGLE_GUIDE](docs/KAGGLE_GUIDE.md) | Step-by-step for the models that exceed the laptop GPU |
| [01_eda](notebooks/01_eda.ipynb) · [02_results](notebooks/02_results.ipynb) | Executed notebooks with figures |

---

## Measurement protocol

What every number in this repository is held to
([EVALUATION_PROTOCOL](docs/EVALUATION_PROTOCOL.md)):

- **5 fixed seeds**, mean ± std. A single-seed number is never a headline.
- **Per-class bootstrap CI.** Neutral F1 on dev is 0.688 **[0.593, 0.772]** — 9× wider than the
  majority classes. A point estimate on 73 examples invites over-reading.
- **Seed-level paired t-test** is primary; the within-run bootstrap is reported beside it as
  evaluation-set uncertainty. They disagreed once, and [ADR-013](docs/DECISIONS.md) explains why.
- **A stated resolution floor.** Dev cannot resolve anything below **~0.027 macro-F1** in a single
  run. Any UIT-VSFC result claiming a +0.01 improvement from one run is reporting noise.
- **Anything fitted on the evaluation set must be cross-fitted** before its benefit is reported
  ([ADR-015](docs/DECISIONS.md)). Optimism bias measured at +0.012 to +0.036.
- **Test evaluated at gates only**, every touch appended to `results/test_evaluations.log`.
- **Every number traceable** to a `run_id` in `results/registry.csv`.

Things the published literature on this dataset does not report: topic macro-F1, seed variance,
per-class confidence intervals, train↔test leakage, label noise, or any latency figure.

---

## Reference machine

All latency figures come from one documented machine; a number from a cloud VM is not comparable and
does not enter the registry.

| | |
|---|---|
| CPU | AMD Ryzen 5 6600H · 6C/12T · AVX2 · **no AVX512-VNNI** |
| GPU | RTX 3050 Laptop, 4.29 GB — 69 s/epoch for PhoBERT-base |
| Runs to date | **69**, all local. Zero external GPU used |

`max_length` 96 (from the Gate G0 subword profile) plus dynamic padding already cut FP32 CPU p95
from **177.5 ms to 50.8 ms — 3.49×, before any quantization**.

---

## Status · 7/10

- [x] Problem definition, data card, split — *G0*
- [x] TF-IDF baseline — *G1*
- [x] PhoBERT fine-tuning & reproduction — *G2*
- [x] Macro / per-class F1 + confusion matrix — *G1*
- [x] Word-segmentation ablation — *G3*
- [x] API + Docker + CI — *G7*
- [x] Reproducible from a clean clone — *G7*
- [ ] ≥30 error cases coded by linguistic feature — *G5*
- [ ] ONNX / quantization benchmark — *G6*
- [ ] HF model card — *G7*

Open problems and the plan for closing them: **[docs/STATUS.md](docs/STATUS.md)**.

---

## CV snippet

> Fine-tuned PhoBERT for Vietnamese sentiment/topic classification on 16k+ labeled sentences;
> raised test macro-F1 from **0.745** (tuned TF-IDF) to **0.837 ± 0.003** over 5 seeds, and showed
> that a published "segmentation is unnecessary" result is metric-dependent — it holds on accuracy
> (+0.96 pp) and fails on macro-F1 (+2.34 pp) and minority-class F1 (+5.42 pp).

Every figure above is traceable to a `run_id`. Rules for quoting them honestly:
[ROADMAP § 9](docs/ROADMAP.md#9-cv-snippet-and-claim-discipline).

---

## License and citation

MIT — see [LICENSE](LICENSE). **The UIT-VSFC corpus is not covered by it** and no corpus data is
included here; see [DATA_CARD § 11](docs/DATA_CARD.md#11-licensing-and-citation).

Cite Nguyen et al. (KSE 2018) for the corpus and Nguyen & Nguyen (EMNLP Findings 2020) for PhoBERT.
