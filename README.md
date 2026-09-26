# ViFeedback NLP Service

Vietnamese feedback classification — **sentiment** (3-class) and **topic** (4-class) on
[UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) — benchmarking
TF-IDF against fine-tuned PhoBERT, with a CPU latency budget and a measurement protocol strict
enough to have overturned several of this project's own claims — two of them retracted after
external review (ADR-015, ADR-018).

[![CI](https://github.com/ThanhDatVN/ViFeedback-NLP-Service/actions/workflows/ci.yml/badge.svg)](https://github.com/ThanhDatVN/ViFeedback-NLP-Service/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Result

**Test split, 5 seeds, PhoBERT-base + word segmentation.** The test set was untouched until Gate G4
and every evaluation is logged.

Two pipelines are reported separately, because the best *research* score and the *deployed*
configuration are not the same artifact.

| Sentiment, test, 5 seeds | Macro-F1 | Weighted F1 | Neutral F1 |
|---|---|---|---|
| PhoBERT-base + **VnCoreNLP** (best measured) | **0.8373 ± 0.0031** | 0.9391 ± 0.0020 | 0.5955 ± 0.0070 |
| PhoBERT-base + **pyvi** (what the service runs) | **0.8288 ± 0.0108** | 0.9369 | 0.5714 |
| TF-IDF B3 + dev-fitted priors | 0.7450 | 0.8817 | 0.3530 |

| Topic, test, 5 seeds | Macro-F1 | Weighted F1 | Others F1 |
|---|---|---|---|
| PhoBERT-base + pyvi | **0.8038 ± 0.0045** | 0.8907 | 0.5533 |
| TF-IDF B4 LinearSVC | 0.7423 | 0.8596 | 0.4790 |

For reference, [BamiBERT (2026)](https://arxiv.org/html/2607.02259v1) Table 2 reports UIT-VSFC
sentiment F1 **83.41** and topic F1 **79.90**. The averaging convention, seed protocol and model
selection behind those numbers are not stated in enough detail to assume they match ours, so this is
context rather than a ranking. A like-for-like comparison would mean running that model in this
harness.

**No state-of-the-art claim is made.** These numbers sit near published figures, but a seed
standard deviation from this harness cannot establish significance against someone else's point
estimate, and the protocols behind those figures are not documented in comparable detail.

What the locked test split *did* establish is worth more. On **dev** the same model scored
**0.8670**, a gap that reads like a comfortable win. The dev-to-test drop is **-0.0298**, and the
TF-IDF baseline dropped **-0.026** in the same direction, so the gap belongs to the split rather
than the model. Reporting the dev number would have been wrong, and only running the locked split
revealed it.

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
| **H2** | Word segmentation is unnecessary and dominates p95 latency | **Both halves wrong, for different reasons.** *Latency*: falsified by measurement — 0.31 ms p95, 0.6% of the model's cost. *Accuracy*: segmentation is worth **+0.0234 macro-F1** (t = 8.58, p = 0.0010, 5/5 seeds), which **agrees with** the source paper. My reading of that paper was the error (ADR-018) |
| **H3** | INT8 may be slower than FP32 without AVX512-VNNI | **Open.** Reference CPU measured: AMD Ryzen 5 6600H, `avx2=true`, **`avx512_vnni=false`** |

### Segmentation: an independent replication, not a refutation

An earlier version of this README claimed to have refuted
[arXiv:2301.00418](https://arxiv.org/abs/2301.00418). **That was a misreading and is retracted**
(ADR-018). The paper's conclusion is conditional: segmentation may be unnecessary for *traditional
classifiers*, and **is necessary** for deep-learning models that use BPE. PhoBERT is the latter, so
the measurement below agrees with the paper rather than contradicting it.

| Metric | raw to segmented (VnCoreNLP) |
|---|---|
| Accuracy | +0.96 pp |
| Weighted F1 | +1.02 pp |
| **Macro-F1** | **+2.34 pp** |
| **Neutral F1** | **+5.42 pp** |

What remains is a **quantified independent replication**: an effect size with seed variance
(t = 8.58, p = 0.0010, non-overlapping seed ranges), a per-class breakdown showing the effect is
roughly five times larger on the minority class than on the aggregate, and a per-segmenter latency
cost the original does not report. The paper's finding that RDRsegmenter is the most stable toolkit
also reproduces here: VnCoreNLP 0.8670 > pyvi 0.8643 > underthesea 0.8618 on dev.

---

## What was got wrong, and corrected

The decision log records four occasions where measurement overturned a plan — including one
retraction of a result already written up as a success.

| ADR | What was claimed | What measurement showed |
|---|---|---|
| [007](docs/DECISIONS.md) | Teencode and missing diacritics are the error-analysis targets | They are **0.16%** and **0.14%** of the corpus. Reframed as robustness targets measured by induced perturbation |
| [008](docs/DECISIONS.md) | The TF-IDF baseline would reach ~0.70 macro-F1 | It reached **0.78**. Three pre-registered ranges falsified, all low; success criteria revised against the measured baseline |
| [012](docs/DECISIONS.md) | Removing segmentation would be the biggest latency win | Segmentation **helps accuracy** and costs 0.6% of p95. The headline hypothesis was wrong |
| [018](docs/DECISIONS.md) | We refuted a published paper's conclusion | **We did not.** That conclusion is conditional and our result *agrees* with it. Caught by external review. Three documents repeated the claim because none of them carried the quote that would have refuted it |
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
- **Interval width, stated honestly.** A *single model's* dev macro-F1 carries a bootstrap
  half-width of about **±0.027**, driven by 73 neutral examples. That is not a threshold below which
  differences are noise: a **paired** difference cancels the shared evaluation-sample variation and
  can be resolved far more tightly. Paired differences are therefore estimated by resampling the
  same examples for both systems, never inferred from one model's interval (corrected after
  review — R7).
- **Anything fitted on the evaluation set must be cross-fitted** before its benefit is reported
  ([ADR-015](docs/DECISIONS.md)). Optimism bias measured at +0.012 to +0.036.
- **Test evaluated at gates only**, every touch appended to `results/test_evaluations.log`.
- **Every number traceable** to a `run_id` in `results/registry.csv`.

Not found in the published work surveyed here: seed variance, per-class confidence intervals,
train-test leakage, label noise, or any latency figure. (Topic F1 *is* reported — BamiBERT Table 2
gives 79.90 — an earlier version of this README wrongly said otherwise.)

---

## Reference machine

All latency figures come from one documented machine; a number from a cloud VM is not comparable and
does not enter the registry.

| | |
|---|---|
| CPU | AMD Ryzen 5 6600H · 6C/12T · AVX2 · **no AVX512-VNNI** |
| GPU | RTX 3050 Laptop, 4.29 GB — 69 s/epoch for PhoBERT-base |
| Runs to date | 94 registry rows (73 dev, 21 test). Ten rows come from two models run on a Kaggle T4; the rest are local |

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

Open problems: **[docs/STATUS.md](docs/STATUS.md)**.
External code review and the next research cycle:
**[docs/REVIEW_AND_RESEARCH_PLAN.md](docs/REVIEW_AND_RESEARCH_PLAN.md)**.

Known open issues from that review, tracked in `docs/STATUS.md`: export quality contract (R3),
API inference tests with a real artifact (R4), readiness semantics (R5), unbounded metrics
buffer (R6), dependency/revision pinning (R11).

---

## CV snippet

> Built a reproducible Vietnamese feedback classification benchmark on UIT-VSFC; PhoBERT +
> VnCoreNLP reached sentiment test macro-F1 **0.837 ± 0.003** across five seeds against **0.745**
> for a tuned TF-IDF baseline, with minority-class analysis, preprocessing ablations and
> per-segmenter latency measured on documented hardware.

The deployed pipeline uses pyvi and scores **0.829 ± 0.011** — quote that figure when describing the
service, not the research best.

Every figure above is traceable to a `run_id`. Rules for quoting them honestly:
[ROADMAP § 9](docs/ROADMAP.md#9-cv-snippet-and-claim-discipline).

---

## License and citation

MIT — see [LICENSE](LICENSE). **The UIT-VSFC corpus is not covered by it** and no corpus data is
included here; see [DATA_CARD § 11](docs/DATA_CARD.md#11-licensing-and-citation).

Cite Nguyen et al. (KSE 2018) for the corpus and Nguyen & Nguyen (EMNLP Findings 2020) for PhoBERT.
