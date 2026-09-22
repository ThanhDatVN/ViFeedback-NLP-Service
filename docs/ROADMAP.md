# ViFeedback NLP Service — Development Roadmap

**Owner:** [@ThanhDatVN](https://github.com/ThanhDatVN)
**Planned:** 2026-09-22
**Horizon:** 8 core weeks (2026-09-22 → 2026-11-16) + 2 stretch weeks (→ 2026-11-30)
**Budget assumption:** 10–12 focused hours/week. Every phase has an explicit time box; when a box is
blown, the *scope* is cut, not the gate.

---

## Table of contents

1. [Objectives](#1-objectives)
2. [Scope boundaries](#2-scope-boundaries)
3. [Success criteria](#3-success-criteria)
4. [Repository layout](#4-repository-layout)
5. [Phase plan](#5-phase-plan)
6. [Schedule](#6-schedule)
7. [Checklist-to-gate mapping](#7-checklist-to-gate-mapping)
8. [Risk register](#8-risk-register)
9. [CV snippet and claim discipline](#9-cv-snippet-and-claim-discipline)
10. [Definition of done](#10-definition-of-done)

---

## 1. Objectives

Objectives are stated so that each is *falsifiable* and has a named owning gate. Anything that cannot
be measured is not an objective; it is a nice-to-have and lives in [Phase 8](#phase-8--stretch-optional).

### O1 — Modeling quality (owner: G4)
Produce a Vietnamese feedback classifier whose **test macro-F1** beats a tuned TF-IDF baseline by a
margin that is statistically significant under a paired bootstrap (α = 0.05), on **both** the sentiment
and topic tasks, using the official UIT-VSFC split without modification.

### O2 — Honest measurement (owner: G1)
Every reported number carries: macro-F1, weighted F1, accuracy, per-class P/R/F1, support, a confusion
matrix, mean ± std over 5 seeds, and a 95% bootstrap CI. No single-seed number is ever reported as a
headline result. The test split is evaluated only at gates, and every evaluation is logged.

### O3 — Preprocessing evidence (owner: G3)
Answer, with numbers rather than convention, whether VnCoreNLP word segmentation is required for this
task — measuring both its **accuracy effect** and its **latency cost**, and treating the two jointly.

### O4 — CPU inference efficiency (owner: G6)
Reduce p95 single-request latency on the reference laptop CPU against the PyTorch FP32 baseline, subject
to a **pre-registered accuracy budget of ≤ 0.5 pp macro-F1 loss**. Report model size, p50/p95/p99, and
throughput. If the optimization does not pay off, report that instead — the negative result is the deliverable.

### O5 — Linguistic error understanding (owner: G5)
Produce a coded taxonomy over ≥ 30 (target: 60) test-set errors, with per-category counts, at least one
mitigation attempt per top-3 category, and a re-measurement showing whether the mitigation worked.
Supplement with controlled perturbation suites (de-diacritization, teencode injection) that quantify
robustness as a macro-F1 *drop*, not an anecdote.

### O6 — Shippable service (owner: G7)
A FastAPI service in a Docker image, with health/readiness endpoints, structured logging, Prometheus
metrics, a pinned model artifact, a green CI pipeline, and an end-to-end reproduction from a clean clone
in one documented command.

### O7 — Communicable result (owner: G7)
A README, an HF model card, and a results table that a reviewer can read in 90 seconds and a hiring
manager can interrogate for 20 minutes without finding an unsupported claim.

---

## 2. Scope boundaries

**In scope:** sentence-level sentiment (3-class) and topic (4-class) classification on UIT-VSFC;
classical and transformer models; preprocessing ablation; CPU inference optimization; error analysis;
a single-model-per-task serving API.

**Explicitly out of scope** — recorded so that scope creep is a decision, not a drift:

| Excluded | Reason |
|---|---|
| Aspect-based sentiment (ABSA) | Different task and dataset (UIT-ViSFD); would double the project |
| Pretraining a Vietnamese LM from scratch | No compute budget; no marginal value over fine-tuning |
| Multi-dataset joint training as the headline | Breaks comparability with published UIT-VSFC numbers. Allowed only as a Phase 8 stretch, reported separately |
| LLM API classification as the main system | Cost and latency make it a poor fit for the deployment objective. Kept as one *reference row* only (Phase 8) |
| GPU serving | The stated constraint is CPU inference |
| Web UI | The OpenAPI `/docs` page is sufficient evidence of a usable API |

---

## 3. Success criteria

These are **pre-registered targets**, set before any experiment runs, so a miss stays visible instead of
being retro-fitted. The reasoning behind each range is in
[EXPERIMENT_MATRIX.md § Expected ranges](EXPERIMENT_MATRIX.md#expected-ranges-pre-registered).

> **Revised at Gate G1 (ADR-008).** S1–S4 below are the revised thresholds. The originals were set
> against an assumed baseline of ~0.70 macro-F1; the measured baseline is **0.782**, which made the
> original S3 (+0.08) require beating the published state of the art, and made S1/S4 passable by the
> TF-IDF baseline alone. Originals are preserved in ADR-008.

| ID | Criterion | Minimum | Target |
|---|---|---|---|
| S1 | Sentiment test macro-F1 (best model) | ≥ 0.80 | ≥ 0.84 |
| S2 | Topic test macro-F1 (best model) | ≥ 0.79 | ≥ 0.83 |
| S3 | Macro-F1 lift over the **tuned** TF-IDF baseline (sentiment) | ≥ +0.025 abs, p < 0.05 | ≥ +0.05 |
| S4 | Neutral-class F1 (sentiment) | ≥ 0.55 | ≥ 0.65 |
| S5 | p95 latency, batch = 1, reference CPU | ≤ 60 ms | ≤ 30 ms |
| S6 | Accuracy cost of the shipped optimization | ≤ 0.5 pp macro-F1 | ≤ 0.2 pp |
| S7 | Served artifact size on disk | ≤ 200 MB | ≤ 120 MB |
| S8 | Docker image size | ≤ 1.2 GB | ≤ 700 MB |
| S9 | Coded error cases | ≥ 30 | ≥ 60 |
| S10 | Clean-clone reproduction | Works | Works, < 15 min on CPU for the eval path |

**S4 is the criterion most likely to fail and the one most worth fighting for.** With 458 neutral
training examples out of 11,426, neutral F1 is precisely what separates this project from every other
UIT-VSFC notebook on GitHub. The tuned TF-IDF baseline already reaches **0.503**, so the revised bar of
0.55 asks the transformer to earn its place on exactly the class that matters.

---

## 4. Repository layout

```
ViFeedback-NLP-Service/
├── README.md
├── pyproject.toml              # ruff + mypy + pytest config, pinned deps
├── Makefile                    # one target per reproducible step
├── .github/workflows/ci.yml
├── Dockerfile                  # multi-stage, non-root, ORT-only runtime
├── docker-compose.yml
├── configs/                    # one YAML per run; the run IS the config
│   ├── data/*.yaml
│   ├── baseline/*.yaml
│   ├── phobert/*.yaml
│   └── serve/*.yaml
├── data/
│   ├── raw/                    # gitignored; fetched by `make data`
│   ├── interim/                # segmented / normalized variants, cached
│   └── processed/              # model-ready parquet, one per preprocessing variant
├── src/vifeedback/
│   ├── data/                   # loading, split integrity, leakage checks
│   ├── preprocess/             # normalizers, teencode map, segmenters (pluggable)
│   ├── models/                 # baseline_tfidf.py, phobert.py, multitask.py
│   ├── training/               # trainer, losses (focal, r-drop), callbacks, seeding
│   ├── evaluation/             # metrics, bootstrap, confusion, error export, challenge sets
│   ├── inference/              # onnx export, quantization, runtime wrappers
│   ├── serving/                # FastAPI app, schemas, middleware
│   └── cli.py                  # typer entrypoint: every experiment is a CLI call
├── notebooks/
│   ├── 01_eda.ipynb
│   └── colab_train.ipynb       # thin wrapper: clones repo, calls the SAME cli.py
├── results/
│   ├── runs/<run_id>/          # config.yaml, metrics.json, preds.csv, env.json, logs
│   ├── registry.csv            # append-only index of every run — committed
│   └── figures/
├── tests/
│   ├── unit/  integration/  data/  contract/  perf/
└── docs/                       # this folder
```

**Two rules that carry the reproducibility of the whole project:**

1. **Notebooks contain no logic.** `colab_train.ipynb` clones the repo and calls
   `python -m vifeedback.cli train --config configs/phobert/xxx.yaml`. The laptop and Colab therefore
   execute identical code paths, and a Colab-only bug becomes impossible.
2. **`results/registry.csv` is append-only and committed.** One row per run:
   `run_id, date, phase, task, model, preprocessing, seed, split, macro_f1, weighted_f1, accuracy, p95_ms, notes, git_sha`.
   This file *is* the experiment log, which makes every table in
   [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) a query rather than a retyping exercise — and makes
   every claim in the README traceable to a `run_id`.

---

## 5. Phase plan

Each phase lists **objective → tasks → artifacts → exit gate**. A gate is binary. If a gate fails, the
phase is extended *or* descoped by an explicit written decision recorded in `docs/DECISIONS.md`
(one ADR-style entry per decision) — never silently skipped.

---

### Phase 0 — Foundations and data integrity
**Week 1 · laptop · ~10 h · owns checklist item 1**

The phase most portfolio projects skip, and the reason most of them are not trusted.

**Tasks**

1. Scaffold the repo per §4. Pin dependencies (`torch`, `transformers`, `scikit-learn`, `onnxruntime`,
   `fastapi`) to exact versions in `pyproject.toml`. Set up ruff, mypy, pytest, pre-commit.
2. `make data`: fetch UIT-VSFC from the HF Hub (`uitnlp/vietnamese_students_feedback`), cross-check the
   counts against the official UIT NLP release, persist to `data/raw/` with a SHA256 manifest.
3. **Split integrity suite** (`tests/data/`):
   - split sizes are exactly **11,426 / 1,583 / 3,166**;
   - label ids are in range and map to the documented names (sentiment `0=negative, 1=neutral, 2=positive`;
     topic `0=lecturer, 1=training program, 2=facility, 3=others`);
   - **train↔test leakage check** — exact-duplicate detection, then near-duplicate detection on a
     normalized form (lowercased, diacritics stripped, whitespace collapsed) via MinHash/LSH. Publish the
     overlap count *whatever it turns out to be*. UIT-VSFC is short, formulaic student feedback, so
     non-trivial overlap is plausible and it materially changes how the headline number should be read;
   - a class-distribution snapshot test, which fails loudly if the upstream data ever changes under you.
4. **EDA notebook**: per-split class distribution for both tasks; sentence-length distribution in
   characters, syllables and **PhoBERT subword tokens**; the sentiment × topic joint contingency table;
   top n-grams per class; and a manual read of 50 random sentences to build intuition before modeling.
5. Write [DATA_CARD.md](DATA_CARD.md) from measured values, not from the paper's abstract.
6. Choose `max_length` from the observed token-length p99 — not by copying 256 from a tutorial. This one
   decision propagates directly into Phase 6's latency numbers.

**Artifacts** — `data/raw` + manifest, `data/processed/base`, `notebooks/01_eda.ipynb`,
`results/figures/dist_*.png`, a completed `DATA_CARD.md`, green `tests/data/`.

**Gate G0** — all integrity tests pass; the data card has zero `TBD`s in its measured sections;
`max_length` chosen with a written justification; the leakage number published.

---

### Phase 1 — Baselines and the evaluation harness
**Week 2 · laptop · ~11 h · owns checklist items 2 and 4**

The harness built here is reused *unchanged* by every later phase. Building it against cheap models means
its bugs surface in seconds rather than after a 40-minute GPU run.

**Tasks**

1. **Evaluation harness** (`src/vifeedback/evaluation/`) — the single most reused component in the repo:
   - `evaluate(y_true, y_pred, y_prob, labels) → metrics dict`: macro / weighted / micro F1, per-class
     P/R/F1/support, accuracy, balanced accuracy, MCC, confusion matrix (counts and row-normalized);
   - 95% bootstrap CI on macro-F1 (10k resamples) and a **paired bootstrap** for model-vs-model tests;
   - a standardized artifact writer producing
     `results/runs/<run_id>/{config.yaml,metrics.json,preds.csv,confusion.png,env.json}` plus one
     `registry.csv` row;
   - unit-tested against hand-computed values and against `sklearn` on synthetic cases.
2. **Baseline ladder** — each rung exists to make the next number interpretable:

   | ID | Model | Purpose |
   |---|---|---|
   | B0 | Majority class | The macro-F1 *floor* (~0.22 for 3 classes) |
   | B0b | Stratified random | Chance under the true prior |
   | B1 | TF-IDF word 1–2 gram + Logistic Regression | The conventional baseline |
   | B2 | TF-IDF **char_wb 3–5 gram** + LR | Expected stronger on Vietnamese user text — char n-grams survive teencode, typos and missing diacritics that destroy word features |
   | B3 | `FeatureUnion(word, char)` + LR | Combined |
   | B4 | LinearSVC on the best feature set | Classifier axis |
   | B5 | Best features + `class_weight='balanced'` | Isolates the *imbalance* effect from the *model* effect |

3. Tune B1–B5 with `RandomizedSearchCV` fit on train and **selected on the official dev split**, never on
   test: `min_df`, `max_features`, `sublinear_tf`, `C`, loss.
4. Run every baseline for both tasks; produce confusion matrices; fill the first results table.
5. **Decision-threshold tuning on dev.** For sentiment, tune per-class thresholds / class priors to
   maximize dev macro-F1. This is a cheap and legitimate macro-F1 lever, and it establishes whether the
   baseline's neutral failure is a *representation* problem or a *threshold* problem — a distinction
   almost no public UIT-VSFC notebook draws, and a good interview answer on its own.

**Artifacts** — a tested harness, 10+ registry rows, `results/figures/cm_baseline_*.png`, the baseline
table in [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) filled.

**Gate G1** — harness unit tests green; all baselines reported with the full metric suite; the B2-vs-B1
comparison recorded; the best baseline's neutral-class F1 documented as the number to beat.

---

### Phase 2 — PhoBERT fine-tuning and reproduction
**Week 3 · Colab GPU (train) + laptop (eval) · ~12 h · owns checklist item 3**

**Tasks**

1. **Colab workflow**: `colab_train.ipynb` clones the repo, installs pinned deps, mounts Drive for
   checkpoints, calls `cli.py train`. Keep any single run **under 30 minutes** and make it resumable —
   free-tier sessions get interrupted, and a training loop that cannot resume will cost you a week.
2. Fine-tune `vinai/phobert-base` for sentiment and for topic on canonically segmented input
   (RDRSegmenter) as the **reference configuration**:
   - AdamW, lr 2e-5, batch 32, 4 epochs, warmup 10%, weight decay 0.01, grad clip 1.0, fp16;
   - checkpoint selection on **dev macro-F1** — not dev loss, not accuracy. With a 4% minority class,
     selecting on loss or accuracy quietly selects the model that ignores neutral;
   - `max_length` from Phase 0.
3. **5 seeds** (42, 1337, 2024, 7, 31337) per configuration; report mean ± std. BERT fine-tuning is
   famously unstable on small datasets, and neutral has ~458 training examples — a single-seed number
   here is noise dressed as a result.
4. Evaluate on dev throughout; **one** locked test evaluation at the gate; log it.
5. Compare against published UIT-VSFC results and **reconcile the metric definitions explicitly**. The
   literature's ~92–94% figures are weighted F1 / accuracy, not macro-F1 (see
   [RESEARCH_NOTES.md § Prior results](RESEARCH_NOTES.md#prior-results-on-uit-vsfc)). Put both columns in
   the table so the comparison misleads in neither direction.

**Artifacts** — 10 training runs in the registry, a seed-variance table, the first macro-vs-weighted F1
side-by-side, an HF-format checkpoint on Drive.

**Gate G2** — PhoBERT beats the best baseline on dev macro-F1 for both tasks; seed std reported; the
weighted-F1 reproduction lands within ~2 pp of published work (or the divergence is explained); test
evaluated exactly once and logged.

---

### Phase 3 — Preprocessing and word-segmentation ablation
**Week 4 · Colab + laptop · ~11 h · owns checklist item 5**

Deliberately designed as a **joint accuracy-and-latency** experiment, which is what makes it more than a
box to tick. PhoBERT's model card states input *must* be word-segmented; published work
([arXiv:2301.00418](https://arxiv.org/abs/2301.00418)) finds segmentation is not necessary for Vietnamese
sentiment classification. Both cannot be operationally true. Resolve it on this task, on this hardware.

**Conditions** — one axis at a time, 5 seeds each, all selected on dev:

| ID | Preprocessing |
|---|---|
| P0 | Raw text straight into the PhoBERT tokenizer (violates the model card) |
| P1 | **VnCoreNLP RDRSegmenter** — canonical; the Phase 2 reference |
| P2 | `underthesea` word_tokenize — a segmenter swap, so the ablation measures sensitivity to *tool choice*, not just to segmentation |
| P3 | P1 + Unicode NFC normalization + whitespace/punctuation cleanup |
| P4 | P3 + teencode / abbreviation dictionary normalization |
| P5 | P3 + lowercasing |
| P6 | Best of the above + emoji → sentiment-token mapping |

**Latency measurement, run inside this phase, not deferred to Phase 6.** Time the segmentation step in
isolation on the laptop (p50/p95 per sentence). `py_vncorenlp` starts a JVM and calls into it, so the
plausible outcome is that **segmentation dominates end-to-end p95**. If it does, then P0 performing within
noise of P1 is not a curiosity — it is the largest single latency win available, and it surfaces *before*
the quantization work rather than after it.

**Analysis** — paired bootstrap P0 vs P1; mean ± std across seeds; and a 2-D plot of dev macro-F1 against
end-to-end p95 latency with the Pareto frontier marked. That figure is the phase's headline output and
probably the best single slide in the whole project.

**Artifacts** — ~35 runs, the ablation table, `results/figures/pareto_preprocessing.png`, and a written
conclusion carrying an explicit recommendation for the served pipeline.

**Gate G3** — all conditions run at 5 seeds; segmentation latency measured in isolation; a documented
serving-pipeline decision justified on both axes.

---

### Phase 4 — Improvement ladder
**Week 5 · Colab · ~12 h · owns O1**

Methods ordered basic → advanced. **Stop climbing when the dev gain stops exceeding the seed std** — that
stopping rule is itself worth stating in the write-up. Everything is selected on dev; test stays locked.

**Tier A — imbalance handling** *(highest expected value, given S4)*
- class-weighted cross-entropy; focal loss (γ ∈ {1, 2}); logit adjustment / balanced softmax;
- per-class threshold tuning on dev, carried over from Phase 1;
- oversampling neutral *versus* weighting it — compare, do not assume.

**Tier B — training stability and schedule**
- **layer-wise learning-rate decay** (top-layer lr, multiplicative decay ≈ 0.9 downward);
- lr ∈ {1e-5, 2e-5, 3e-5, 5e-5} × epochs ∈ {3, 5, 10} with early stopping; warmup ratio; weight decay 0.1;
- re-initializing the top *k* encoder layers;
- longer training with early stopping in place of a few fixed epochs.

**Tier C — regularization and robustness**
- label smoothing (0.05–0.1);
- **R-Drop** — KL between two dropout passes; strong on small datasets;
- **FGM adversarial training** in embedding space — cheap, well-suited to noisy user text, and it targets
  the teencode/typo failure modes Phase 5 will quantify;
- weight EMA / checkpoint averaging.

**Tier D — data-centric**
- augmentation aimed at the neutral class: back-translation (vi → en → vi), synonym swap, and
  **synthetic teencode + diacritic-stripping augmentation** — the last both adds data and hardens the
  model against a known failure mode, so it pays twice;
- a label-noise audit on train using confident-learning-style scoring; inspect the top-100 suspected
  labels by hand. **Do not clean the test set** — report what you find instead.

**Tier E — model axis** *(one variable at a time, against the same recipe)*
- `vinai/phobert-base-v2` — same size, much larger pretraining corpus (+120 GB OSCAR-2301);
- `uitnlp/visobert` — social-media-pretrained, SentencePiece, needs **no** segmentation, so it pairs
  naturally with Phase 3's P0 condition;
- `xlm-roberta-base` — multilingual control;
- `vinai/phobert-large` — if Colab allows;
- **multi-task**: one shared encoder, two heads (sentiment + topic), joint loss. The tasks share an input
  distribution, so this is both an accuracy bet *and* a halving of the number of served models — an
  operational argument worth making explicitly in the write-up.

**Tier F — ensembling**
- seed ensemble (logit averaging) and **model soup** (weight averaging across seeds/recipes). A soup costs
  *nothing* at inference, which is the right trade under a latency budget; a logit ensemble costs N× and
  will probably lose to quantization on the Pareto plot. Show both on that plot and let the figure make
  the argument.

**Artifacts** — ~40 runs; a per-tier gain table with the seed std printed beside every delta; one selected
champion config per task.

**Gate G4** — champion selected on **dev**; a single locked test evaluation; paired bootstrap vs. baseline
significant (O1); every tier's outcome recorded, including the ones that did not help.

---

### Phase 5 — Error analysis and robustness
**Week 6 · laptop · ~11 h · owns checklist item 6**

**Tasks**

1. Export the champion's misclassifications. **Sample stratified by confusion-matrix cell**, over-sampling
   the interesting ones (neutral→positive, neutral→negative, others→lecturer) rather than sampling errors
   uniformly — uniform sampling just re-discovers that the big classes have the most errors.
2. Code 60 cases against the taxonomy in
   [EVALUATION_PROTOCOL.md § Error taxonomy](EVALUATION_PROTOCOL.md#error-taxonomy): negation and its
   scope, contrastive discourse (`nhưng`, `tuy nhiên`), teencode, missing diacritics, typos and
   elongation, sarcasm, mixed polarity, suggestion-as-neutral, implicit sentiment, code-switching,
   arguable gold label, multi-topic. Do **two coding passes separated by ≥ 24 h** and record the
   disagreement rate between your own passes as a self-consistency figure — that single number signals
   annotation literacy more clearly than the taxonomy itself.
3. **Controlled perturbation suites** — the step that converts anecdotes into measurement. Derive
   perturbed copies of the *entire* test set programmatically and report the macro-F1 drop for each:
   - de-diacritization (`không tốt` → `khong tot`);
   - teencode injection from the dictionary (`không` → `k` / `ko` / `hok`);
   - character-level noise (swap / drop / duplicate at 5% and 10%);
   - a hand-built negation probe set (~50 minimal pairs with and without `không`).

   The resulting robustness table is something no other public UIT-VSFC project will have.
4. Attempt one mitigation for each of the top-3 categories — most likely the adversarial training or
   augmentation from Tiers C/D — and **re-measure both clean and perturbed macro-F1**. Report honestly
   wherever the mitigation bought robustness by spending clean accuracy.

**Artifacts** — `results/error_analysis.csv` with 60 coded rows, a category tally with examples, the
robustness table, and a before/after mitigation comparison.

**Gate G5** — ≥ 30 (target 60) cases coded; perturbation suites run; ≥ 1 mitigation measured; findings
written in prose that names specific Vietnamese linguistic phenomena.

---

### Phase 6 — CPU inference optimization
**Week 7 · laptop only · ~12 h · owns checklist item 7 and O4**

Reference-machine discipline: record CPU model, core/thread count, **AVX2 / AVX512-VNNI support**, RAM,
OS, power plan, and whether the machine was on AC. Without VNNI, INT8 dynamic quantization of a transformer
encoder can be *slower* than FP32 — which is exactly why H3 is measured rather than assumed.

**Optimization ladder, measured cumulatively**

| Step | Change | Expectation |
|---|---|---|
| L0 | PyTorch FP32, padding to `max_length` | Baseline (deliberately naive) |
| L1 | **Dynamic padding to the longest sequence in the batch** | Often the biggest single win; UIT-VSFC sentences are short |
| L2 | `torch.no_grad`, eval mode, `intra_op_num_threads` sweep 1…N | Free |
| L3 | ONNX export (opset 17) + ORT `ORT_ENABLE_ALL` graph optimization | Moderate |
| L4 | ORT **dynamic** INT8 quantization | Uncertain — may regress; this is H3 |
| L5 | ORT **static** INT8 with calibration on dev | Usually the real win |
| L6 | OpenVINO INT8 via `optimum-intel` / NNCF | Frequently best on Intel CPUs |
| L7 | Drop segmentation from the pipeline, if Phase 3 permits | Potentially larger than L3–L6 combined |
| L8 | *(stretch)* distil a 4–6 layer student from the champion | Best size and latency, costs accuracy |

**Benchmark protocol** — full spec in
[EVALUATION_PROTOCOL.md § Latency harness](EVALUATION_PROTOCOL.md#latency-harness): 200 warmup +
1000 timed iterations; inputs sampled from the **real test-length distribution**, never a fixed-length
dummy; 5 repetitions with the median reported; p50/p95/p99 at batch = 1; throughput at batch ∈ {1, 8, 32};
artifact size on disk; and macro-F1 re-evaluated after *every* step against the ≤ 0.5 pp budget.

**Artifacts** — a latency × accuracy × size table for L0–L7, `results/figures/pareto_inference.png`, a
documented serving choice, and an `env.json` capturing the reference machine.

**Gate G6** — every ladder step measured, regressions included and reported as such; one configuration
selected with latency *and* accuracy inside budget; **the CV snippet's latency number is now fixed and quotable**.

---

### Phase 7 — Service, CI, and release
**Week 8 · laptop · ~12 h · owns checklist items 8, 9, 10**

**Tasks**

1. **FastAPI service** — `POST /v1/classify` (single and batch; returns label, probabilities, model
   version), `GET /healthz`, `GET /readyz` (model actually loaded), `GET /metrics` (Prometheus: request
   count, latency histogram, per-class prediction counts for drift-watching), `GET /version` (model SHA +
   git SHA). Pydantic v2 schemas, a request-size cap, a timeout, and structured JSON logging with a
   request id.
2. **Docker** — multi-stage build on `python:3.11-slim`; **ONNX Runtime only in the runtime layer, no
   torch** — this is precisely where the S8 image-size target is met or missed; non-root user;
   `HEALTHCHECK`; thread count configurable by env var.
3. **CI (GitHub Actions)** — ruff → mypy → pytest (unit / data / contract / api) → docker build →
   container smoke test → a perf smoke test with a loose CI-runner threshold. Full strategy in
   [EVALUATION_PROTOCOL.md § Software testing](EVALUATION_PROTOCOL.md#software-testing-strategy).
4. **Model contract test** — a golden file of ~20 fixed inputs with expected labels, asserted after every
   export and quantization step, plus a requirement of ≥ 99.5% label agreement between FP32 and the
   shipped artifact across the full test set. This is the test that catches a silently broken quantized
   model before it ships, and it is the one engineers will notice.
5. **Hugging Face Hub** — push model, tokenizer and ONNX artifact; write a model card covering intended
   use, training data, the **full metric table including per-class and macro F1**, limitations (neutral-class
   weakness, domain = university feedback 2013–2016, the robustness deltas from Phase 5), and the
   evaluation protocol.
6. **README final pass**, then fill the CV snippet under the §9 rules.
7. **Clean-clone rehearsal** — a fresh directory, `git clone`, follow the README *literally*, and time it.
   Everything that breaks gets fixed. This is the step most portfolio repos fail, and the cheapest one to pass.

**Gate G7** — CI green on a clean clone; the container serves correct predictions; model card published;
README complete; checklist at 10/10; clean-clone reproduction timed and documented.

---

### Phase 8 — Stretch (optional)
**Weeks 9–10 · owns nothing; purely upside**

Ranked by CV value per hour:

1. **Knowledge distillation** into a 4–6 layer student — the strongest latency story, and a genuinely
   advanced technique to be able to discuss.
2. **Cross-domain generalization** — evaluate the UIT-VSFC champion zero-shot on another Vietnamese
   feedback corpus (UIT-ViSFD, NEU-ESC). A domain-shift number costs one evaluation run and signals
   maturity better than almost anything else here.
3. **LLM reference point** — zero/few-shot classification with a modern LLM as a single comparison row,
   with cost and latency alongside. Positions the fine-tuned encoder correctly instead of ignoring the
   elephant in the room.
4. **LLM-as-teacher pseudo-labeling** for the neutral class, then retrain.
5. **Calibration analysis** — ECE, reliability diagram, temperature scaling. Matters for any service that
   returns probabilities.
6. **Monitoring** — log prediction distributions and add a PSI-based drift check.

---

## 6. Schedule

| Week | Dates | Phase | Where | Primary deliverable | Gate |
|---|---|---|---|---|---|
| 1 | Sep 22 – Sep 28 | P0 Foundations | Laptop | Data card + integrity suite | **G0** |
| 2 | Sep 29 – Oct 05 | P1 Baselines | Laptop | Eval harness + baseline ladder | **G1** |
| 3 | Oct 06 – Oct 12 | P2 PhoBERT | Colab | 5-seed reproduction | **G2** |
| 4 | Oct 13 – Oct 19 | P3 Ablation | Both | Segmentation accuracy × latency Pareto | **G3** |
| 5 | Oct 20 – Oct 26 | P4 Improvement | Colab | Champion model per task | **G4** |
| 6 | Oct 27 – Nov 02 | P5 Error analysis | Laptop | 60 coded errors + robustness suites | **G5** |
| 7 | Nov 03 – Nov 09 | P6 Inference | Laptop | ONNX / quantization benchmark | **G6** |
| 8 | Nov 10 – Nov 16 | P7 Release | Laptop | API + Docker + CI + model card | **G7** |
| 9–10 | Nov 17 – Nov 30 | P8 Stretch | Both | Distillation / cross-domain / LLM row | — |

**Weekly cadence** — Mon: plan (pick runs from the matrix) · Tue–Thu: execute · Fri: evaluate and append
to `registry.csv` · Sat: write *this week's* section of the report · Sun: 30-minute gate review, honestly.

**If you fall behind,** cut in this order: Phase 8 → Tiers E/F of Phase 4 → conditions P2/P5/P6 of Phase 3
→ error cases from 60 down to 30. **Never cut:** the evaluation harness (P1), the 5-seed policy, or the
clean-clone rehearsal (P7). Those three are what make everything else credible.

---

## 7. Checklist-to-gate mapping

| # | Checklist item | Gate | Evidence |
|---|---|---|---|
| 1 | Problem definition + data card + split | G0 | `docs/DATA_CARD.md`, green `tests/data/` |
| 2 | TF-IDF baseline | G1 | Baseline table, registry rows B0–B5 |
| 3 | PhoBERT fine-tuning & reproduction | G2 | 5-seed table, literature reconciliation |
| 4 | Macro/per-class F1 + confusion matrix | G1 → G2 | `metrics.json`, `results/figures/cm_*.png` |
| 5 | Word segmentation / preprocessing ablation | G3 | Ablation table + Pareto figure |
| 6 | ≥ 30 error cases by linguistic feature | G5 | `results/error_analysis.csv` + robustness table |
| 7 | ONNX / quantization benchmark | G6 | L0–L7 table + inference Pareto |
| 8 | API + Docker + CI | G7 | Green workflow badge, running container |
| 9 | HF model card + README | G7 | Hub URL, README |
| 10 | Reproducible end-to-end run | G7 | Clean-clone rehearsal, timed |

---

## 8. Risk register

| ID | Risk | L | I | Mitigation | Trigger point |
|---|---|---|---|---|---|
| R1 | `py_vncorenlp` needs a JVM → breaks the slim Docker image or blows past S8 | High | High | Segment **offline** during data prep so runtime never needs a JVM; if Phase 3 shows segmentation is unnecessary, drop it from serving entirely; fallback is pure-Python `pyvi`/`underthesea` in the service | Phase 0 install; Phase 7 image build |
| R2 | Dynamic INT8 slower than FP32 (no AVX512-VNNI) | Med-High | Med | Pre-registered: static INT8 and OpenVINO are already in the ladder, not bolted on after a failure. Report the regression as a finding — it is H3 | Phase 6, step L4 |
| R3 | Neutral class too small → macro-F1 swings across seeds | High | High | 5 seeds mandatory; report std and bootstrap CI; never quote a single seed; Tier A of Phase 4 targets it directly | Phase 2 onward |
| R4 | Colab session limits / GPU unavailable | Med | Med | Keep runs < 30 min, checkpoint to Drive every epoch, make training resumable, queue runs so a lost session costs one run not one week | Phase 2 onward |
| R5 | Test-set overfitting through repeated evaluation | Med | High | Test evaluated **only at gates**; every test evaluation logged with date and reason; all selection on dev | Every gate |
| R6 | Train/test near-duplicate leakage inflates every number | Med | High | Measured in Phase 0 and published in the data card; if material, report both raw and deduplicated-test numbers | Gate G0 |
| R7 | Scope creep (extra datasets, extra models) | High | Med | The §2 exclusion table; anything new requires an ADR in `docs/DECISIONS.md` | Weekly gate review |
| R8 | Results land but the write-up is late and thin | Med | High | Each phase's report section is written on the **Saturday of its own week**, not at the end of the project | Weekly cadence |
| R9 | Published baselines don't reproduce (different metric definitions) | Med | Low | Reconcile weighted vs macro F1 explicitly in Phase 2; treat any mismatch as a finding, not a failure | Gate G2 |
| R10 | Laptop thermal throttling corrupts latency numbers | Med | Med | 5 repetitions, median reported; AC power; record the power plan; re-run any benchmark whose repetitions disagree by > 10% | Phase 6 |

---

## 9. CV snippet and claim discipline

**Template**

> Fine-tuned PhoBERT for Vietnamese sentiment/topic classification on 16k+ labeled sentences;
> raised macro-F1 from `[baseline]` to `[result]` and reduced CPU p95 latency by `[x]`% with `[method]`.

**Rules for filling it**

1. `[baseline]` is the **best tuned** TF-IDF baseline's test macro-F1, never the weakest one. Quoting a
   deliberately weak baseline to inflate a delta is the most common failure in portfolio projects, and an
   interviewer finds it in one question.
2. `[result]` is the champion's test macro-F1, **mean over 5 seeds**, quoted to 3 decimals with the std
   available on request.
3. `[x]` is the p95 reduction on the reference CPU, with both configurations measured under the identical
   harness on the identical machine.
4. `[method]` names whatever actually won. If dropping segmentation beat quantization, say so — it is the
   more interesting sentence anyway.
5. Every number in the snippet must be traceable to a `run_id` in `results/registry.csv`.

**Pre-computed variants** — pick the one your results actually support:

- *Efficiency-led:* "…cut CPU p95 latency 3.1× (ONNX + static INT8) at a 0.2 pp macro-F1 cost, and shipped
  it behind a FastAPI/Docker service with CI."
- *Ablation-led:* "…showed VnCoreNLP word segmentation gave no significant macro-F1 gain while accounting
  for 60% of end-to-end p95, and removed it — a larger latency win than quantization."
- *Robustness-led:* "…quantified robustness to teencode and missing diacritics with controlled perturbation
  suites (−7.4 pp macro-F1) and recovered 4.1 pp via adversarial training."

**Never claim** a number from a single seed, a dev number described as test, an accuracy figure presented
as macro-F1, or a latency figure from a machine other than the one documented in `env.json`.

---

## 10. Definition of done

The project is done when a reviewer can:

1. `git clone`, follow the README, and reproduce the headline evaluation on CPU in under 15 minutes;
2. read **one** table giving macro-F1, weighted F1, per-class F1, latency and size for every model considered;
3. find, for every claim in the README, a `run_id` in `results/registry.csv` that produced it;
4. see at least one clearly reported negative result;
5. `docker run` the image and get a correct prediction from `curl` within a minute;
6. read the error analysis and learn something specific about Vietnamese text they did not previously know.

Items 4 and 6 are what make this a portfolio project rather than a tutorial.
