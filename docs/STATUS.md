# Status, Open Problems and Next Experiments

**Updated:** 2026-09-27, during research Cycle 1. Living document. Results live in
[EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) and [results/studies/](../results/studies/README.md);
decisions in [DECISIONS.md](DECISIONS.md); the external review this cycle follows is
[REVIEW_AND_RESEARCH_PLAN.md](REVIEW_AND_RESEARCH_PLAN.md). This file says **where the project
stands, what is wrong with it, and what runs next.**

---

## 1. Progress

### Gates (the original plan)

| Gate | Phase | Status | Evidence |
|---|---|---|---|
| **G0** | Foundations & data integrity | ✅ | `results/data_report.json`, data tests |
| **G1** | Baselines & evaluation harness | ✅ | 22 runs, harness pinned against sklearn |
| **G2** | PhoBERT reproduction | ✅ | 10 runs, 5 seeds × 2 tasks |
| **G3** | Word-segmentation ablation | ✅ | 15 runs + latency benchmark |
| **G4** | Locked test evaluation | ✅ | 21 test rows, every touch in `results/test_evaluations.log` |
| G5 | Error analysis & robustness | 🔶 | Tooling, robustness suite and audit sheet done ([Cycle 0](#2-cycle-0--what-the-current-model-gets-wrong)); the human audit is not |
| G6 | CPU inference optimization | 🔶 | FP32 release verified on the reference machine (ADR-020); INT8 release on Kaggle and a steady-state benchmark still to run |
| **G7** | Service, Docker, CI | ✅ | FastAPI, multi-stage image, CI; API now also tested against a real artifact (R4) |

### Research cycles (after external review)

| Cycle | Question | State |
|---|---|---|
| **0** | Fix validity problems; what does the current model get wrong? | ✅ [results/studies/](../results/studies/README.md) |
| **1** | Three hypotheses chosen from Cycle 0, declared before running ([cycle1.yaml](../configs/experiments/cycle1.yaml)) | 🔶 laptop runs (H1, H2) in progress; H3 on Kaggle |

**Compute to date** ([ledger](../configs/experiments/ledger.csv)): 55 weight-updating runs before
Cycle 1 (40 laptop, 10 Kaggle T4, 5 OOF diagnostics), about 4.5 GPU-hours. The registry holds 97
rows. After removing 4 duplicated ids and 10 same-seed re-runs, 83 distinct results remain.

---

## 2. Cycle 0 — what the current model gets wrong

Full write-up: [results/studies/README.md](../results/studies/README.md). All on validation or
out-of-fold train predictions; the test split was not touched.

| Finding | Number | Consequence |
|---|---|---|
| Neutral recall is the gap | 0.56 dev / 0.53 OOF; precision 0.80 / 0.70 | The model under-predicts neutral |
| Neutral errors are **confident** | 66–72% at p ≥ 0.9; 33% of train neutral confidently assigned elsewhere | No threshold reaches them; this is a label/representation question |
| The boundary is already nearly right | Tuned neutral bias +0.10 → +0.004 dev macro-F1 | Weak support for the decision-boundary explanation |
| Probabilities are overconfident | T ≈ 1.5–1.6 in four independent fits; NLL −17%, ECE 0.034 → 0.014 | Temperature scaling is worth shipping; it changes no F1 |
| Abstention hides neutral | 90% coverage keeps 48% of neutral | Any abstention rule must report per-class coverage |
| No diacritics breaks the pipeline | macro-F1 0.863 → 0.268; 64% of predictions become neutral | Deployment risk invisible in the 99.86%-diacritized benchmark |
| Negation is learned one-directionally | 36/36 positive→negated pairs; 2/8 negative→negated | *không* acts as a negative cue more than as an operator (small probe) |
| Training is deterministic | 10 re-run seeds agree to 4 decimals; stage 1 of Cycle 1 reproduces seed 42 exactly | Seed variance is the only run-to-run variance |
| fp16 evaluation noise | One tied neutral example: 0.8634 vs 0.8672 | Single-seed deltas under ~0.004 are uninterpretable |

---

## 3. Cycle 1 — declared hypotheses and their outcome

Declared in [configs/experiments/cycle1.yaml](../configs/experiments/cycle1.yaml) and committed
(0f3055d) before the first run. Decisions are computed by `vifeedback study cycle1` using exactly the
declared rules (`evaluation/decisions.py`); output in `results/studies/cycle1/decisions.json`.

<!-- CYCLE1_RESULTS -->

---

## 4. Open problems

Ordered by how much they threaten the conclusions.

### P1 — The dev set cannot resolve small effects
Dev holds **73 neutral examples** carrying one third of the macro average. One neutral example moves
dev macro-F1 by ~0.004, and fp16 rounding alone moved one (ADR-020). A single model's dev macro-F1
carries a bootstrap half-width of about ±0.027. That describes one model's uncertainty; a paired
comparison resolves differences more tightly (R7). *Mitigations in use:* seed-paired tests,
same-session controls (Cycle 1), OOF predictions over train for anything label-related.
*Open:* repeated stratified CV on train for the finalist selection.

### P2 — The neutral gap is not a threshold problem
Cycle 0 shows the errors are confident and the boundary nearly optimal. What remains is whether the
confident errors are **label policy / ambiguity** or **representation**. Only the human audit can
separate them ([ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md); 160-row sheet ready). *Blocking for:*
E04 (training on corrected labels) and any further imbalance work.

### P3 — Measured gold-label noise bounds the ceiling
~4.8% disagreement on textually identical train/test pairs. The corpus annotates suggestions as
negative (91.1%), but implicit suggestions without a cue word appear among gold-neutral examples.
The audit measures this within strata; test labels are never changed.

### P4 — Topic is the weaker task and has had no intervention
Topic PhoBERT + pyvi: 0.8111 dev / 0.8038 test, against sentiment's 0.864 / 0.829. Segmentation helps
topic too (+0.0141, p = 0.004). `facility` is the one class where TF-IDF beats PhoBERT (0.921 vs
0.905). *Candidates:* TF-IDF × PhoBERT stacking (E08), multi-task (E07).

### P5 — Robustness to informal orthography is a deployment risk
Cycle 0 measured it; H2 tests the first mitigation. The no-diacritic failure is a whole-pipeline
failure: pyvi also mis-segments unaccented text, and the suite does not separate the two. A diacritic
restoration front-end is the untested alternative.

### P6 — Latency numbers need re-measuring
The Phase 6 torch benchmark file was lost with the `results/` deletion, and its last run showed p95
spreads of 22–48% across repeats: an unstable measurement that must not be reported (R10). The
harness now times model-only on preprocessed input and names throughput texts/s. *Next:* benchmark
the released FP32 artifact, and the Kaggle INT8 artifacts, on an idle, cool reference machine.

### P7 — Reproducibility is partly pinned
Run identity includes a config hash, and run directories cannot be overwritten by a different
configuration (R11). Per-run `metrics.json`/`config.yaml` are committed. *Open:* pretrained model and
dataset revisions are not pinned yet (revisions recorded below), and Kaggle rows carry no git SHA.

| Artifact | Revision to pin |
|---|---|
| `vinai/phobert-base` | `01daacda68afe13d83023d16ec647239e344a1e6` |
| `vinai/phobert-large` | `70e2cfcd3cce29c970aee4954ea34a32bb30afdc` |
| `FacebookAI/xlm-roberta-base` | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` |
| UIT-VSFC `refs/convert/parquet` | `2d76906157232f87e9883fa37ebfc91ffdee1f83` |

---

## 5. External review — item status

Full traceability, including the study designs, catalog and backlog:
[REVIEW_COMPLIANCE.md](REVIEW_COMPLIANCE.md).

| Item | Severity | Status |
|---|---|---|
| R1 segmentation paper misread | High | ✅ retracted (ADR-018) |
| R2 FGM + AMP gradient bug | High | ✅ fixed, 6 regression tests |
| R3 export quality contract | High | ✅ release step with staging, manifest, full-validation acceptance (ADR-020) |
| R4 API never tested with a model | High | ✅ real-artifact API tests (run when a release exists) |
| R5 readiness semantics | Medium | ✅ 503 unless required tasks + segmenter loaded; fallback cost measured (−0.052) |
| R6 unbounded metrics buffer | Medium | ✅ bounded buffer + counter |
| R7 over-strong statistical claims | High | ✅ interval wording corrected |
| R8 model comparison closed too early | Medium | 🔶 narrowed (ADR-019); tokenizer profiles measured; XLM-R raw control running (Cycle 1 H3) |
| R9 headline vs deployed config | Medium | ✅ reported separately in README |
| R10 docs and benchmark out of sync | Medium | 🔶 docs synced; benchmark re-run pending (P6) |
| R11 reproducibility pinning | Medium | ✅ run ids, overwrite guard, revisions pinned, source hash, lock file, data content reference, CI integration job |
| R12 benchmark claims | Medium | ✅ no SOTA claim; BamiBERT reported as context |

---

## 6. Next — Cycle 2 options

Chosen after Cycle 1's decisions, one specialization at a time (review § 10):

1. **Finish Cycle 1**: H3 on Kaggle; finalist seeds for any method that met its advance rule; the
   closing gate (one logged test evaluation per finalist, calibrated vs uncalibrated).
2. **The human audit** (P2/P3). Cheapest remaining information, and it gates E04.
3. **Data science track:** label-efficiency curves (E11), OOF-ranked label correction (E04).
4. **ML engineering track:** INT8 on the reference CPU, distillation into a 4–6 layer student (E15).
5. **AI engineering track:** an LLM reference row with frozen prompts, cost and latency (E19).

---

## 7. Compute plan

| Work | Where | Why |
|---|---|---|
| PhoBERT-base training, OOF, robustness, calibration, FP32 export | **Laptop RTX 3050** | ~65 s/epoch, no session limits |
| Full-embedding XLM-R, PhoBERT-large, INT8 quantization | **Kaggle** | > 4.29 GB, or needs the blocked `onnx` package (ADR-017/020) |
| **All latency benchmarking** | **Laptop only, idle** | the reference machine is recorded in `env.json`; no benchmark while training runs |

Notebook: [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb) (§ 4e for Cycle 1) and
[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md).
