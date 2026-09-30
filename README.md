# ViFeedback NLP Service

Vietnamese student-feedback classification: **sentiment** (3 classes) and **topic** (4 classes) on
[UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback), and a CPU service
for the sentiment model.
- **The benchmark.** It compares TF-IDF with fine-tuned PhoBERT under a measurement protocol strict
  enough to have overturned several of this project's own claims (ADR-015, ADR-018).
- **The research after it.** Pre-registered cycles asked what limits the minority class and what
  survives real input: typing without diacritics, informal text, and other universities' posts.
- **The published model.**
  [Datk4/vifeedback-sentiment-phobert](https://huggingface.co/Datk4/vifeedback-sentiment-phobert)
  on the Hugging Face Hub, with a verified release manifest.

[![CI](https://github.com/ThanhDatVN/ViFeedback-NLP-Service/actions/workflows/ci.yml/badge.svg)](https://github.com/ThanhDatVN/ViFeedback-NLP-Service/actions/workflows/ci.yml)
[![Reproduce from the Hub](https://github.com/ThanhDatVN/ViFeedback-NLP-Service/actions/workflows/reproduce.yml/badge.svg)](https://github.com/ThanhDatVN/ViFeedback-NLP-Service/actions/workflows/reproduce.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Code license](https://img.shields.io/badge/code-MIT-green)
![Weights license](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-lightgrey)

---

## At a glance

| | |
|---|---|
| **Served model** | A 6-layer PhoBERT student distilled from five seeds of the 12-layer model (ADR-039, ADR-040); weights stored in FP16, computed in FP32: **185 MB** (the 12-layer model: 540 MB) |
| **Sentiment, test, 5 seeds** | Served student **0.817**; the 12-layer model 0.830 ± 0.007 (−0.013 [−0.020, −0.005]: the price of the size and speed); research best 0.837 with VnCoreNLP |
| **Robustness** | Diacritics restored before the model: stripped validation 0.686 → **0.857** (ADR-031). Out-of-scope input flagged by a topic detector: AUROC **0.922** within another university's forum (ADR-034) |
| **Latency** | Raw text in, served pipeline p95 **12.3 ms** on a laptop CPU (the 12-layer model 22.1 ms in the same sessions; target 30 ms) |
| **Reproducible** | A clean clone reproduces the published model's validation macro-F1 (0.8644) in **2.3 min** in CI, every file checked against `SHA256SUMS` |
| **Research** | Cycles 0–4 closed, Cycle 5 run; 40 ADRs, 157 weight-updating runs in [the ledger](configs/experiments/ledger.csv); Cycle 5 declared ([cycle5.yaml](configs/experiments/cycle5.yaml)) |

---

## Result

**Test split, 5 seeds, PhoBERT-base + word segmentation.** The test set was untouched until Gate G4,
and every evaluation since is logged (28 in all, [test_evaluations.log](results/test_evaluations.log)).

Two pipelines are reported separately, because the best *research* score and the *deployed*
configuration are not the same artifact.

| Sentiment, test, 5 seeds | Macro-F1 | Weighted F1 | Neutral F1 |
|---|---|---|---|
| PhoBERT-base + **VnCoreNLP** (best measured) | **0.8373 ± 0.0031** | 0.9391 ± 0.0020 | 0.5955 ± 0.0070 |
| PhoBERT-base + **pyvi** | **0.8288 ± 0.0108** | 0.9369 | 0.5714 |
| PhoBERT-base + pyvi + diacritic/teencode augmentation (the 12-layer release, [ADR-027](docs/DECISIONS.md)) | **0.8296 ± 0.0071** | 0.9365 | 0.5757 |
| 6-layer student distilled from it (**what the service runs**, [ADR-040](docs/DECISIONS.md)) | **0.8168** (5 seeds) | 0.9320 (seed 42) | 0.5455 (seed 42) |
| TF-IDF B3 + dev-fitted priors | 0.7450 | 0.8817 | 0.4207 |

| Topic, test, 5 seeds | Macro-F1 | Weighted F1 | Others F1 |
|---|---|---|---|
| PhoBERT-base + pyvi | **0.8038 ± 0.0045** | 0.8907 | 0.5533 |
| TF-IDF B4 LinearSVC | 0.7423 | 0.8596 | 0.4790 |

**No state-of-the-art claim is made.**
[BamiBERT (2026)](https://arxiv.org/html/2607.02259v1) reports UIT-VSFC sentiment F1 83.41 and topic
F1 79.90. Its averaging convention, seed protocol and model selection are not stated in enough
detail to assume they match these, so it is context, not a ranking.

What the locked test split *did* establish is worth more:
- On **dev** the same model scored **0.8670**, which reads like a comfortable win.
- The dev-to-test drop is **−0.0298**, and the TF-IDF baseline dropped **−0.026** in the same
  direction. The gap belongs to the split, not the model.

### Why macro-F1

UIT-VSFC sentiment is **4.32% neutral**. On the real test split, a classifier that never predicts
`neutral` but is otherwise perfect scores accuracy **0.947**, weighted F1 **0.922**, macro-F1
**0.649**. Only macro-F1 notices that an entire class is missing, and most published UIT-VSFC results
report the other two. *(Pinned by `tests/unit/test_metrics.py`.)*

---

## What the service does

```text
raw text ─► lowercase, NFC ─► diacritic restorer* ─► pyvi segmentation ─► 6-layer PhoBERT (ONNX) ─► label, probabilities
                                                                    └──► TF-IDF topic detector ─► in_scope, scope_score
* only when the input is essentially unaccented (ADR-031)
```

```bash
curl -s localhost:8000/v1/classify -H 'content-type: application/json' \
  -d '{"texts":["giang vien nhiet tinh voi sinh vien", "tuyển dụng thực tập sinh marketing"]}'
```

Each prediction carries `label`, `confidence`, `probabilities`, `in_scope` and `scope_score`.
- The service never refuses an input: `in_scope: false` is information for the caller.
- The restorer and the scope detector are refused unless their SHA-256 matches the release
  manifest. `/version` reports the model file's hash, and `make docker-e2e` checks it inside the
  image.
- `/healthz`, `/readyz`, `/version` and `/metrics` cover operations.

The limits are stated in the [model card](https://huggingface.co/Datk4/vifeedback-sentiment-phobert):
- The labels follow UIT-VSFC's policy, where a suggestion counts as negative.
- The detector judges topic from words.
- Other universities' posts score far lower (0.44 macro-F1 on NEU-ESC validation).
- The served student is 0.013 macro-F1 below the 12-layer model on test; that model stays published
  for uses where accuracy matters more than size and speed.

---

## Research cycles

An external review ([REVIEW_AND_RESEARCH_PLAN](docs/REVIEW_AND_RESEARCH_PLAN.md)) set the workflow:
*observe an error → propose competing explanations → design a controlled intervention → evaluate
independently → record the decision and its limits.* Each cycle's hypotheses, controls and decision
rules are committed before its first run ([configs/experiments/](configs/experiments/README.md)).

| Cycle | Question | What it settled |
|---|---|---|
| **0** | What does the deployed model get wrong? | Neutral errors are confident (66–72% at p ≥ 0.9), so this is not a threshold problem. Probabilities are overconfident (T ≈ 1.5). No diacritics collapses macro-F1 from 0.863 to 0.268 |
| **1** | Four declared hypotheses | Neutral is not fixable in the classifier head. Augmentation doubles no-diacritic robustness at no clean cost, **confirmed on test**. XLM-R's deficit is partly preprocessing. A shared encoder brings no gain |
| **2** | Confirmation on a frozen, SHA-256-pinned challenge set | Topic stacking fails. The service switched to the augmented model. An instruction LLM beats the encoder on neutral, but only on constructed text (H7) |
| **3** | The served model on real input (NEU-ESC forum posts, ViLexNorm comments) | **The service lowercases and restores diacritics** (NEU-ESC unaccented posts 0.270 → 0.374). 17% of labels flip on real typing. A lexicon and careful INT8 fail their rules |
| **4** | Student text from other institutions | In-domain data helps only with one head per label policy (+0.111 on NEU-ESC), but it failed two declared conditions, so it was not released. **`in_scope` now comes from a topic detector.** INT8 passed non-inferiority but not fidelity (91.4% label agreement, batch-dependent), so it was not released |
| **5** | Real typing, then a smaller model (the owner's order) | **H10 not passed**: consistency training on ViLexNorm pairs removed label flips only by calling informal text negative, caught by the NEU-ESC guard (ADR-038). **H11 passed and served**: a 6-layer student in FP16 storage, 185 MB and 1.8x faster, equal on validation but 0.013 below on test (ADR-039, ADR-040). H12 (other institutions) waits for new labelled data |

Per-cycle tables: [STATUS](docs/STATUS.md) and [EXPERIMENT_MATRIX § 5](docs/EXPERIMENT_MATRIX.md).
Technical report: [RESEARCH_REPORT](docs/RESEARCH_REPORT.md).

### What was got wrong, and corrected

The decision log records every occasion where measurement or review overturned a plan, including
retractions of results already written up as successes.

| ADR | What was claimed | What measurement showed |
|---|---|---|
| [008](docs/DECISIONS.md) | The TF-IDF baseline would reach ~0.70 macro-F1 | It reached **0.78**. Three pre-registered ranges were falsified, all low, and the success criteria were revised against the measured baseline |
| [012](docs/DECISIONS.md) | Removing segmentation would be the biggest latency win | Segmentation costs 0.6% of p95 and **helps accuracy** (+0.0234 macro-F1, 5/5 seeds) |
| [015](docs/DECISIONS.md) | Threshold tuning is "the single largest lever", +0.035 | Cross-fitted, the gain is **zero**. Retracted |
| [018](docs/DECISIONS.md) | We refuted a published paper's conclusion on segmentation | **We did not.** Its conclusion is conditional, and our result agrees with it. Caught by external review |
| [020](docs/DECISIONS.md) | The export verified its artifact | It checked parity on raw text while the service feeds segmented text. It was replaced by a staged release gate with a manifest |
| [032](docs/DECISIONS.md) | The out-of-scope score (AUROC 0.977) detects off-topic input | It detected *another institution*: within one forum its AUROC was 0.573. Replaced by a topic detector (ADR-034) |
| [036](docs/DECISIONS.md) | INT8 passed non-inferiority, so it can ship | It changed 8.6% of labels and depended on the batch. The release gate's fidelity rule held it back |
| [038](docs/DECISIONS.md) | Consistency training cut real-typing label flips from 16.6% to 0.3% | By calling 99.4% of informal comments negative. A guard on real student posts caught it |
| [040](docs/DECISIONS.md) | The distilled student matches the 12-layer model (validation) | On test it is 0.013 lower in every seed; 73 neutral validation sentences could not resolve it. Stated, not hidden |

---

## Quick start

```bash
git clone https://github.com/ThanhDatVN/ViFeedback-NLP-Service.git
cd ViFeedback-NLP-Service
python scripts/setup_venv.py   # .venv with the pinned versions; CUDA torch if an NVIDIA GPU is found
.venv\Scripts\Activate.ps1      # Windows  (Linux/macOS: source .venv/bin/activate)
cp .env.example .env           # optional keys (HF_TOKEN, OPENAI_API_KEY); .env is git-ignored
make data             # fetch UIT-VSFC at its pinned revision + run the integrity suite
make test             # about 400 fast tests
make baseline         # TF-IDF ladder
make train            # fine-tune PhoBERT (needs a GPU; about 5 min per seed on an RTX 3050)
```

`make help` lists every target, and targets use `.venv` automatically. Keys live in `.env` only:
- `HF_TOKEN` is used only for publishing;
- `OPENAI_API_KEY` is used only for the LLM reference measurements (H7), and UIT-VSFC text is never
  sent to an API.

### Reproduce the published model (CPU only, no checkpoint)

```bash
pip install -e ".[serve]"
vifeedback data fetch
vifeedback serve reproduce     # Hub download, SHA-256 checks, validation macro-F1 0.8644
```

[reproduce.yml](.github/workflows/reproduce.yml) runs this from a clean clone in CI (2.3 min,
install included).

### Serve

```bash
make export           # checkpoint -> ONNX in staging -> verified on full validation -> released,
                      # then the diacritic restorer and the scope detector attached (each hash-checked)
make docker-e2e       # build the image, mount the release, check /readyz, SHA-256s and golden labels
make docker && make docker-run
```

The runtime image is 519 MB. It carries no scikit-learn, SciPy or `transformers`: a small PhoBERT
BPE tokenizer and pyvi's CRF were verified identical on all 49,141 corpus texts.

### Research studies

Every study is one CLI call, writing to `results/studies/`:

```bash
vifeedback study tables              # result tables + paired comparisons from the registry
vifeedback study robustness          # perturbation suites, slices, negation probe
vifeedback study challenge           # Cycle 2 H6 on the frozen challenge set
vifeedback study neu-esc-confirm     # Cycle 3 confirmation on real student posts
vifeedback study b4prime             # Cycle 4 topic-aware scope detector
vifeedback study audit-report        # analyse the filled neutral-audit sheet
```

The Kaggle notebooks are for models that exceed a 4.29 GB GPU ([KAGGLE_GUIDE](docs/KAGGLE_GUIDE.md)).

---

## Repository

```text
├── src/vifeedback/
│   ├── cli/           one module per command group; `study` split by research cycle (study_cycle1…4)
│   ├── data/          loading · integrity · leakage · profiling
│   ├── preprocess/    normalizers · segmentation backends · diacritic restorer · variants
│   ├── models/        TF-IDF ladder (B0–B5) · cross-fitted prior tuning
│   ├── training/      fine-tuning loop · losses · augmentation · multi-task · domain (H8)
│   ├── evaluation/    metrics · bootstrap · decisions · calibration · robustness · audit · scope
│   │                  external data · LLM reference · INT8 power
│   ├── inference/     ONNX export · release gate · PhoBERT BPE tokenizer · publish · reproduce
│   └── serving/       FastAPI app · pipeline · scope detector (numpy)
├── configs/           model configs · experiments/ = declared cycles (cycle1–5.yaml) + run ledger
├── data/              challenge set and probes (the project's own); corpora fetched, never committed
├── docs/              see docs/README.md
├── results/           registry, per-run metrics, studies (see results/README.md)
├── notebooks/         EDA and results (executed) · two Kaggle notebooks
├── scripts/           venv setup · Docker end-to-end · tokenizer and segmenter equivalence checks
├── tests/             unit · data · contract · integration · packaging guards
└── models/            local weights, gitignored (see models/README.md)
```

**Documentation map:** [docs/README.md](docs/README.md). The ones to start with:

| Document | Contents |
|---|---|
| **[STATUS](docs/STATUS.md)** | Progress by gate and cycle, open problems, what runs next |
| **[NEXT_PLAN](docs/NEXT_PLAN.md)** | Cycle 5 and the open targets, with research sources |
| [RESEARCH_REPORT](docs/RESEARCH_REPORT.md) | Technical report: questions, method, findings, negative results, limitations |
| [DECISIONS](docs/DECISIONS.md) | 40 ADRs: every plan correction forced by measurement or review |
| [EXPERIMENT_MATRIX](docs/EXPERIMENT_MATRIX.md) | Run-ID scheme and every result table |
| [EVALUATION_PROTOCOL](docs/EVALUATION_PROTOCOL.md) · [EVALUATION_DATA](docs/EVALUATION_DATA.md) | What every number is held to; what counts as evidence |
| [DATA_CARD](docs/DATA_CARD.md) | UIT-VSFC measured from the files, and what is never committed |
| [HUONG_DAN_THU_CONG](docs/HUONG_DAN_THU_CONG.md) | *(Tiếng Việt)* The owner's manual tasks, step by step |

---

## Measurement protocol

What every number in this repository is held to ([EVALUATION_PROTOCOL](docs/EVALUATION_PROTOCOL.md)):

- **Five fixed seeds**, mean ± std. A single-seed number is never a headline.
- **Paired comparisons.** Seed-paired runs and paired bootstrap over the same examples. A single
  model's dev macro-F1 carries about ±0.027, because 73 neutral examples carry a third of the macro
  average. A paired difference resolves far more tightly (corrected after review, R7).
- **Declared before run.** Hypotheses, controls, metrics and decision rules are committed before the
  first run of each cycle, and the rules are code (`evaluation/decisions.py`).
- **Cross-fitting.** Anything fitted on the evaluation set is cross-fitted before its benefit is
  reported (ADR-015; optimism bias measured at +0.012 to +0.036).
- **Test discipline.** The test set is used at gates only, and every use is logged. Later cycles
  confirm on new data (a frozen challenge set, then NEU-ESC test, each use logged).
- **Traceability.** Every number is traceable to a `run_id` in `results/registry.csv`.
- **Negative results stay.** Failed hypotheses are committed with the same detail as passed ones.

### Reference machine

All latency figures come from one documented machine; a number from a cloud VM does not enter the
registry.

| | |
|---|---|
| CPU | AMD Ryzen 5 6600H · 6C/12T · AVX2 · **no AVX512-VNNI** |
| GPU | RTX 3050 Laptop, 4.29 GB · about 65 s per epoch for PhoBERT-base |
| Compute to date | 173 registry rows; 144 weight-updating runs, about 12.9 GPU-hours ([ledger](configs/experiments/ledger.csv)) |

Single-sentence latency, model only, two steady passes (`vifeedback study latency`):

| Configuration | p50 | p95 | vs L0, p50 / p95 | Quality |
|---|---:|---:|---|---|
| L0 PyTorch FP32, pad to 96 | 117.4 ms | 120.5 ms | 1× | reference |
| L1 PyTorch FP32, dynamic padding | 46.7 ms | 64.1 ms | 2.5× / 1.9× | identical |
| **L3 ONNX FP32 (served)** | **15.5 ms** | **33.6 ms** | **7.6× / 3.6×** | identical (parity 8.2e-5) |
| L4 ONNX INT8 dynamic | 9.0 ms | 20.4 ms | 13.1× / 5.9× | neutral F1 −0.088: not released |

Absolute times shift with the laptop's power state, and the speed-up ratios reproduce, so the ratios
are the claim. The 33.6 ms session was a slow machine state: later sessions measured the served
model at 19–20 ms p95, and the full served pipeline at 26.9 ms (ADR-032). Three sessions on an idle
machine will re-measure the pipeline with the scope detector.

---

## Status

- [x] Problem definition, data card, split — *G0*
- [x] TF-IDF baseline — *G1*
- [x] PhoBERT fine-tuning and reproduction — *G2*
- [x] Macro and per-class F1, confusion matrix — *G1*
- [x] Word-segmentation ablation — *G3*
- [x] API, Docker, CI — *G7* · image 519 MB (S8)
- [x] Reproducible from a clean clone — 2.3 min in CI (S10)
- [x] ONNX benchmark — *G6*: FP32 ONNX served, 7.6× faster than padded PyTorch. INT8 is faster
  still, but not released (ADR-022, ADR-036)
- [x] Model card and Hub release — [Datk4/vifeedback-sentiment-phobert](https://huggingface.co/Datk4/vifeedback-sentiment-phobert), every file verified against `SHA256SUMS`
- [ ] ≥ 30 error cases coded by linguistic feature — *G5 / S9*: tooling, a 160-row audit sheet and
  its analysis are done; the human audit is not
- [x] Serving artifact ≤ 200 MB — *S7*: a 6-layer student in FP16 storage, 185 MB (ADR-039, ADR-040)

Every external-review item (R1–R12) is closed ([STATUS § 8](docs/STATUS.md#8-external-review--item-status)).

---

## CV snippet

> Built a reproducible Vietnamese feedback classification benchmark on UIT-VSFC; PhoBERT +
> VnCoreNLP reached sentiment test macro-F1 **0.837 ± 0.003** across five seeds against **0.745**
> for a tuned TF-IDF baseline, with minority-class analysis, preprocessing ablations and
> per-segmenter latency measured on documented hardware.

The deployed pipeline scores **0.830 ± 0.007** (5 seeds); quote that figure when describing the
service, not the research best. Rules for quoting: [ROADMAP § 9](docs/ROADMAP.md#9-cv-snippet-and-claim-discipline).

---

## License and citation

- **Code:** MIT, see [LICENSE](LICENSE).
- **Published weights:** CC BY-NC 4.0, a non-commercial research artifact, in line with UIT-VSFC's
  research-purpose release.
- **Data:** no corpus text is included here. UIT-VSFC, NEU-ESC and ViLexNorm are fetched under their
  own terms ([DATA_CARD § 11](docs/DATA_CARD.md#11-licensing-and-citation)).

Cite Nguyen et al. (KSE 2018) for the corpus and Nguyen & Nguyen (Findings of EMNLP 2020) for
PhoBERT.
