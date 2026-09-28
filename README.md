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
| PhoBERT-base + **pyvi** | **0.8288 ± 0.0108** | 0.9369 | 0.5714 |
| PhoBERT-base + pyvi + diacritic/teencode augmentation (**what the service runs**, [ADR-027](docs/DECISIONS.md)) | **0.8296 ± 0.0071** | 0.9365 | 0.5757 |
| TF-IDF B3 + dev-fitted priors | 0.7450 | 0.8817 | 0.4207 |

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

The augmented model matches the plain one on clean test text and keeps working when accents are
missing: no-diacritic test macro-F1 **0.64 vs 0.27**, and on constructed unaccented sentences that the
augmentation code did not produce (a frozen challenge set), accuracy **0.74 vs 0.32**. That is why the service switched to it (Cycle 2).

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

## Three hypotheses, all resolved

| | Hypothesis | Outcome |
|---|---|---|
| **H1** | Macro-F1 is the binding constraint | **Supported.** On test, PhoBERT's advantage over TF-IDF is +0.057 weighted F1 but **+0.175 neutral F1** (+0.092 macro-F1) |
| **H2** | Word segmentation is unnecessary and dominates p95 latency | **Both halves wrong, for different reasons.** *Latency*: falsified by measurement — 0.31 ms p95, 0.6% of the model's cost. *Accuracy*: segmentation is worth **+0.0234 macro-F1** (t = 8.58, p = 0.0010, 5/5 seeds), which **agrees with** the source paper. My reading of that paper was the error (ADR-018) |
| **H3** | INT8 may be slower than FP32 without AVX512-VNNI | **Falsified.** On the Ryzen 5 6600H (AVX2, no VNNI) dynamic INT8 is **1.7× faster** than FP32 ONNX. It is still not shipped: it costs 0.088 neutral F1 (ADR-022) |

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

The decision log records every occasion where measurement or review overturned a plan, including
one retraction of a result already written up as a success.

| ADR | What was claimed | What measurement showed |
|---|---|---|
| [007](docs/DECISIONS.md) | Teencode and missing diacritics are the error-analysis targets | They are **0.16%** and **0.14%** of the corpus. Reframed as robustness targets measured by induced perturbation |
| [008](docs/DECISIONS.md) | The TF-IDF baseline would reach ~0.70 macro-F1 | It reached **0.78**. Three pre-registered ranges falsified, all low; success criteria revised against the measured baseline |
| [012](docs/DECISIONS.md) | Removing segmentation would be the biggest latency win | Segmentation **helps accuracy** and costs 0.6% of p95. The headline hypothesis was wrong |
| [018](docs/DECISIONS.md) | We refuted a published paper's conclusion | **We did not.** That conclusion is conditional and our result *agrees* with it. Caught by external review. Three documents repeated the claim because none of them carried the quote that would have refuted it |
| [019](docs/DECISIONS.md) | Scaling the encoder buys nothing; multilingual loses to monolingual | Only *among the configurations tested*: XLM-R was only ever given pyvi-segmented text it never saw in pretraining. Narrowed after review; the missing control is Cycle 1 H3 |
| [020](docs/DECISIONS.md) | The export verified its artifact | It calibrated and checked parity on **raw** text while the service feeds segmented text, and ignored its own logit check. Replaced by a staged release with a manifest |
| [015](docs/DECISIONS.md) | Threshold tuning is "the single largest lever", +0.035 | Cross-fitted, the gain on PhoBERT is **zero**. Retracted — and the correction *raised* the reported lift, because the inflated baseline had been understating the model |

ADR-015 is the one worth reading. The bug was not convenient, and it surfaced because a result that
had already been written up got re-tested.

---

## Research cycles after external review

An external review ([REVIEW_AND_RESEARCH_PLAN](docs/REVIEW_AND_RESEARCH_PLAN.md)) set the next
workflow: *observe an error → propose competing explanations → design a controlled intervention →
evaluate independently → record the decision and its limits.*

**Cycle 0 — what the deployed model gets wrong** ([results/studies](results/studies/README.md)).
Validation plus out-of-fold predictions over train; the test split was not touched.

| Question | Measured | So |
|---|---|---|
| Is neutral's gap a threshold problem? | 66–72% of neutral errors are at p ≥ 0.9; a tuned neutral bias gains +0.004 | No. The errors are confident, so the question is labels vs representation, and a human audit is next |
| Are probabilities trustworthy? | Overconfident, T ≈ 1.5–1.6 in four independent fits; NLL −17% after scaling | Ship temperature scaling; it changes no prediction |
| Does abstention help? | 90% coverage halves the error rate **and keeps only 48% of neutral** | Report coverage per class, or it hides the minority class |
| Robust to real typing? | No diacritics: macro-F1 0.863 → **0.268**, with 64% of predictions becoming neutral | A deployment risk the 99.86%-diacritized benchmark cannot show |
| Is training reproducible? | 10 re-run seeds and Cycle 1's controls reproduce earlier runs **exactly** | Seed variance is the only run-to-run variance |

**Cycle 1 — four hypotheses, declared before running**
([configs/experiments/cycle1.yaml](configs/experiments/cycle1.yaml), committed before the first run;
decisions computed by code from the declared rules):

| Hypothesis | Decision |
|---|---|
| H1 Neutral is fixable in the classifier (logit adjustment, balanced head retraining) | **No.** Both +0.003 macro-F1, not advanced; each trades neutral precision for recall |
| H2 Diacritic/teencode augmentation buys robustness without clean cost | **Supported at 5 seeds, confirmed on test**: no-diacritic test macro-F1 0.27 → 0.64, clean test accuracy unchanged (+0.0008). Only in-family: character noise barely moves |
| H3 XLM-R's deficit was a preprocessing confound | **Partly.** Raw text +0.010 over pyvi input (4/5 seeds, p = 0.03), but XLM-R still trails PhoBERT-base by 0.014 |
| H4 A shared sentiment/topic encoder helps | **No gain** at λ = 0.3 (one model for both tasks, no loss detected); sentiment cost at λ = 1 |

**Cycle 2 — confirmation on new data**
([configs/experiments/cycle2.yaml](configs/experiments/cycle2.yaml)). The test split had been used 28
times, so Cycle 2 decides on a **305-sentence challenge set** written for it and frozen by SHA-256
before any model saw it ([data/challenge](data/challenge/README.md)).

| Hypothesis | Decision |
|---|---|
| H5 Stacking TF-IDF with PhoBERT helps topic | **No.** −0.010 macro-F1; TF-IDF's apparent edge on `facility` was noise |
| H6 Serve the augmented model | **Yes, by the declared rule.** Constructed noisy text 0.61 → 0.80 accuracy, other rows unchanged. Possible costs, not significant at one seed: teencode 0.975 → 0.875 (p = 0.13), short factual sentences 0.77 → 0.63 (p = 0.22) |
| H7 An instruction LLM does better on the hard cases | **gpt-4o-mini on real posts from another university (NEU-ESC): yes**, macro-F1 0.60 vs 0.49 / 0.46, mostly on neutral (USD 0.21 for 6,613 posts; a reference, not the serving path). **Pilot (Qwen3-1.7B): no.** Neutral F1 0.26–0.29 against the encoder's 0.66. It rejects the corpus's label policy (requests as negative), yet gets factual sentences right where the encoder fails. Declared Qwen3-4B and gpt-4o-mini runs pending |

**Cycle 3 — the served model on real input**
([configs/experiments/cycle3.yaml](configs/experiments/cycle3.yaml),
[docs/EVALUATION_DATA.md](docs/EVALUATION_DATA.md)). The Cycle 2 challenge set was constructed, so it
became development data; real text came from ViLexNorm (human-normalized social-media comments) and
NEU-ESC (forum posts by students of another university, human labels), which is the confirmation set.

| Question | Result |
|---|---|
| Were Cycle 2's drops real? (5 seeds) | Teencode: no. Contrast sentences (*nhưng*): yes, 0.875 → 0.800 |
| Does capitalization matter? | Yes: 1% of labels flipped (the corpus has no uppercase letter). **The service now lowercases input** |
| Is the model stable on real typing? | No: 17% of labels change between a real comment and its human normalization |
| Diacritic restoration before the model | Stripped validation macro-F1 **0.686 → 0.857**, clean predictions unchanged; confirmed on NEU-ESC (unaccented posts 0.270 → 0.374). **Now in the service** |
| Out-of-scope detection | Mahalanobis AUROC 0.949 (development), **0.977 on real off-topic posts** vs 0.920 for max-probability. **The API now returns `in_scope`** |
| Real student text from another university (NEU-ESC) | Macro-F1 drops to 0.46 (CE) and 0.43 (augmented): a domain and label-policy shift; the augmentation hurts here |
| Plain CE + restoration instead of the augmented model | Its contrast-sentence advantage (0.875 vs 0.800) did not replicate on real posts (0.440 vs 0.437): the augmented model stays |
| Real-typing lexicon, careful INT8 | Both failed their declared rules (negative results, kept) |

Details: [STATUS § 3–4](docs/STATUS.md#3-cycle-1--declared-hypotheses-and-their-outcome) ·
report: [RESEARCH_REPORT](docs/RESEARCH_REPORT.md) · open items: [NEXT_PLAN § 1](docs/NEXT_PLAN.md).

---

## Quick start

```bash
git clone https://github.com/ThanhDatVN/ViFeedback-NLP-Service.git
cd ViFeedback-NLP-Service
python scripts/setup_venv.py   # .venv with the pinned versions; CUDA torch if an NVIDIA GPU is found
.venv\Scripts\Activate.ps1      # Windows  (Linux/macOS: source .venv/bin/activate)
cp .env.example .env           # optional: API keys, read by the CLI; .env is git-ignored
make data             # fetch UIT-VSFC + run the integrity suite
make test             # 379 fast tests
make report           # Phase 0 profiling + EDA figures
make baseline         # TF-IDF ladder
make train            # fine-tune PhoBERT (needs a GPU; ~5 min/seed on an RTX 3050)
```

`make` targets use `.venv` automatically when it exists. Keys go in `.env`, never in a tracked file
or a chat: `OPENAI_API_KEY` only for the LLM reference measurements (NEXT_PLAN § 2), `HF_TOKEN`
only for publishing the model.

Research studies are CLI calls too, each writing to `results/studies/`:

```bash
vifeedback study tables              # result tables + paired comparisons from the registry
vifeedback study neutral-audit       # dev + out-of-fold predictions, stratified audit sheet
vifeedback study calibration         # cross-fitted temperature scaling, per-class coverage
vifeedback study robustness          # perturbation suites, slices, negation probe
vifeedback study cycle1              # apply Cycle 1's declared decision rules
vifeedback study challenge           # Cycle 2 H6 on the frozen challenge set
vifeedback study llm-reference       # Cycle 2 H7: an LLM scored by label likelihood
vifeedback study h7-decide           # the H7 rule once both arms exist (Holm, cycle3.yaml v6)
vifeedback study audit-report        # analyse the filled neutral-audit sheet
```

`make help` lists every target. Models too large for a 4.29 GB GPU go to Kaggle —
see [KAGGLE_GUIDE](docs/KAGGLE_GUIDE.md).

### Serving

```bash
make export           # checkpoint -> ONNX in staging -> verify on full dev -> release + manifest,
                      # then the diacritic restorer and the out-of-scope score (each hash-checked)
make docker-e2e       # the image with the released model: /readyz, SHA-256, golden labels
make docker && make docker-run
curl -s localhost:8000/v1/classify \
  -H 'content-type: application/json' \
  -d '{"texts":["giảng viên nhiệt tình với sinh viên ."]}'
```

Each prediction carries `label`, `confidence`, `probabilities` and, when the release has the out-of-scope score,
`in_scope` and `scope_score`. The service lowercases input and restores diacritics on unaccented
text before segmenting it; it never refuses an off-topic input, so the caller decides what
`in_scope: false` means for them.

---

## Repository

```
├── src/vifeedback/
│   ├── data/          loading · integrity · leakage · profiling · lexical analysis
│   ├── preprocess/    normalizers · 4 segmentation backends · materialized variants
│   ├── models/        TF-IDF ladder (B0–B5) · cross-fitted prior tuning
│   ├── training/      fine-tuning loop · losses (focal, logit-adjust, R-Drop, FGM) · seeding
│   ├── evaluation/    metrics · bootstrap · registry · tables from the registry
│   │                  error analysis · calibration · robustness suites
│   ├── inference/     ONNX export · quantization · parity checks · latency harness
│   ├── serving/       FastAPI app · request/response contracts
│   └── cli.py         every experiment is a CLI call
├── notebooks/         EDA and results, both executed with outputs
├── configs/           one YAML per experiment · experiments/ = declared cycles + run ledger
├── docs/              16 documents — see below
├── tests/             unit · data · contract · integration · packaging guards
├── results/           registry.csv (append-only) · per-run metrics · studies/ (generated)
└── Dockerfile · docker-compose.yml · Makefile · .github/workflows/ci.yml
```

| Document | Contents |
|---|---|
| **[STATUS](docs/STATUS.md)** | **Progress, open problems, next experiments, compute plan** |
| [ROADMAP](docs/ROADMAP.md) | Objectives, 8 phases, exit gates, risk register |
| [DECISIONS](docs/DECISIONS.md) | 31 ADRs — every plan correction forced by measurement or review |
| [DATA_CARD](docs/DATA_CARD.md) | Provenance, splits, distributions, 11 measured limitations |
| [EVALUATION_PROTOCOL](docs/EVALUATION_PROTOCOL.md) | Metrics, seeds, significance, latency harness, error taxonomy, perturbation suites, calibration |
| [ANNOTATION_GUIDE](docs/ANNOTATION_GUIDE.md) | Neutral-label audit: taxonomy, ambiguity vs incorrect gold, agreement and adjudication |
| [REVIEW_AND_RESEARCH_PLAN](docs/REVIEW_AND_RESEARCH_PLAN.md) | External review (R1–R12) and the research plan this cycle follows |
| [REVIEW_COMPLIANCE](docs/REVIEW_COMPLIANCE.md) | Every review recommendation → status and evidence |
| [RESEARCH_REPORT](docs/RESEARCH_REPORT.md) | Technical report: questions, method, findings, negative results, limitations |
| [NEXT_PLAN](docs/NEXT_PLAN.md) | Next experiments and improvements, step by step, each tied to a measured result |
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
| Runs to date | 153 registry rows, 118 distinct results after removing duplicate ids and same-condition re-runs; 119 weight-updating runs, ~9.6 GPU-hours, all in [the ledger](configs/experiments/ledger.csv) |

Single-sentence latency, model-only, two steady passes (`vifeedback study latency`):

| Configuration | p50 | p95 | vs L0, p50 / p95 | texts/s (b=32) | Quality |
|---|---:|---:|---|---:|---|
| L0 PyTorch FP32, pad to 96 | 117.4 ms | 120.5 ms | 1× | 12.0 | reference |
| L1 PyTorch FP32, dynamic padding | 46.7 ms | 64.1 ms | 2.5× / 1.9× | 24.1 | identical |
| **L3 ONNX FP32 (served)** | **15.5 ms** | **33.6 ms** | **7.6× / 3.6×** | 24.9 | identical (parity 8.2e-5) |
| L4 ONNX INT8 dynamic (blocked) | 9.0 ms | 20.4 ms | 13.1× / 5.9× | 40.9 | neutral F1 −0.088: not released |

Absolute times shift between laptop sessions with the CPU's power state; the speed-up ratios
reproduce, so they are the claim.

---

## Status · 8/10

- [x] Problem definition, data card, split — *G0*
- [x] TF-IDF baseline — *G1*
- [x] PhoBERT fine-tuning & reproduction — *G2*
- [x] Macro / per-class F1 + confusion matrix — *G1*
- [x] Word-segmentation ablation — *G3*
- [x] API + Docker + CI — *G7*
- [x] Reproducible from a clean clone — *G7*
- [ ] ≥30 error cases coded by linguistic feature — *G5: tooling, robustness suite, a 160-row audit sheet and its analysis done; the human coding is not*
- [x] ONNX / quantization benchmark — *G6: FP32 ONNX served, 7.6× faster than padded PyTorch; INT8 1.7× faster again but blocked by the quality gate*
- [ ] HF model card — *built with checksums (`vifeedback serve publish`, dry run); the upload needs the owner's account*

Open problems: **[docs/STATUS.md](docs/STATUS.md)**.
External code review and the next research cycle:
**[docs/REVIEW_AND_RESEARCH_PLAN.md](docs/REVIEW_AND_RESEARCH_PLAN.md)**.

Every review item (R1–R12) is closed, tracked in [STATUS § 6](docs/STATUS.md#6-external-review--item-status).
What remains needs the owner: the human audit, the declared LLM runs, publishing, natural data. Full audit:
[REVIEW_COMPLIANCE](docs/REVIEW_COMPLIANCE.md). Technical report: [RESEARCH_REPORT](docs/RESEARCH_REPORT.md).

---

## CV snippet

> Built a reproducible Vietnamese feedback classification benchmark on UIT-VSFC; PhoBERT +
> VnCoreNLP reached sentiment test macro-F1 **0.837 ± 0.003** across five seeds against **0.745**
> for a tuned TF-IDF baseline, with minority-class analysis, preprocessing ablations and
> per-segmenter latency measured on documented hardware.

The deployed pipeline uses pyvi with diacritic/teencode augmentation and scores **0.830 ± 0.007**
(5 seeds) — quote that figure when describing the service, not the research best.

Every figure above is traceable to a `run_id`. Rules for quoting them honestly:
[ROADMAP § 9](docs/ROADMAP.md#9-cv-snippet-and-claim-discipline).

---

## License and citation

MIT — see [LICENSE](LICENSE). **The UIT-VSFC corpus is not covered by it** and no corpus data is
included here; see [DATA_CARD § 11](docs/DATA_CARD.md#11-licensing-and-citation).

Cite Nguyen et al. (KSE 2018) for the corpus and Nguyen & Nguyen (EMNLP Findings 2020) for PhoBERT.
