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
| **1** | Four hypotheses, declared before running ([cycle1.yaml](../configs/experiments/cycle1.yaml) v1 + v2) | ✅ all four decided; closing gate (one logged test evaluation per finalist) next |

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

| Hypothesis | Declared prediction | Result (validation, seed-paired vs same-seed control) | Decision by the declared rule |
|---|---|---|---|
| **H1a** logit adjustment τ=1 | < 0.01 macro-F1; neutral recall ↑, precision ↓ | +0.0029 mean [−0.047, +0.053], 2/3 seeds; neutral P 0.79→0.61–0.70, R 0.56→0.66–0.67 | **Not advanced** (needs ≥ +0.005). Prediction confirmed |
| **H1b** balanced head retraining (cRT) | same | +0.0028 mean [−0.025, +0.031], 2/3 seeds; neutral P → 0.59–0.69, R → 0.64–0.77 | **Not advanced.** Seed spread 0.0012 vs 0.011 for CE (observation) |
| **H2** diacritic/teencode augmentation, 30% | ≥ 20% less `nodiacritic-50` degradation, ≤ 0.005 clean loss | **5 seeds** (finalist): degradation 0.217 → 0.124 (**−43%**); full no-diacritic 0.28 → 0.65; teencode −54%; clean mean +0.004, but per seed −0.006 … +0.030 (median −0.002); character noise unchanged (−4%) | **Supported at 5 seeds.** Targeted, in-family robustness at no material clean cost |
| **H3** XLM-R on raw text | raw − pyvi > 0 | Kaggle, same session, 5 seeds: **+0.0097** [+0.0014, +0.0179], 4/5 seeds, p = 0.032. Raw XLM-R 0.8499 vs PhoBERT-base 0.8643 | **Supported**: segmented input cost XLM-R ~0.01. Raw mean below the 0.8523 withdrawal threshold, so ADR-016's narrowed conclusion stands: ~40% of the gap was preprocessing, ~60% remains |
| **H4** shared encoder, λ=0.3 | helps / harms / no difference | sentiment +0.0012, topic +0.0005 (3 seeds) | **No material difference**: one model serves both tasks at half the inference cost, no loss detected |
| **H4** shared encoder, λ=1 | same | sentiment −0.0056 (0/3 seeds), topic +0.0031 (2/3) | **Negative transfer on sentiment** by the rule (interval still spans 0) |

**Controls.** Every H1/H2 control is stage 1 of a cRT run at the same seed. All five reproduce the
registry's CE rows **exactly**, so treatment and control differ only in the treatment. H4's controls
are the single-task P4 runs at the same seeds.

**What Cycle 1 establishes.**

1. **The neutral gap is not in the classifier.** Two independent ways of re-placing the decision
   boundary (training-time logit adjustment, post-hoc balanced head retraining) and Study A's post-hoc
   bias all move macro-F1 by noise-sized amounts. Each one trades neutral precision for recall. The
   remaining explanations, label ambiguity and representation, need the human audit.
2. **Robustness to missing diacritics is cheaply trainable, but only in-family.** Augmentation repairs
   the perturbations it was trained on at no material clean cost, and leaves character noise it never
   saw unchanged (at 3 seeds it looked slightly harmful; at 5 it is −4%, i.e. nothing). It is a
   targeted fix, not general robustness.
3. **XLM-R's deficit was partly a preprocessing artifact.** Fed raw text it gains +0.010 over pyvi
   input in the same session, but still trails PhoBERT-base by 0.014. Its robustness is not better
   where it matters: without diacritics 0.35 (PhoBERT 0.27, PhoBERT + augmentation 0.65), with half
   the diacritics 0.66 (PhoBERT 0.65). Augmentation buys more robustness than the architecture swap.
4. **Sharing the encoder costs nothing at λ = 0.3 and something at λ = 1.** The engineering reading is
   one model instead of two for serving. The scientific reading is that the tasks' measured
   dependence (Cramér's V 0.344) does not translate into a gain at this data size.

**Budget.** 19 laptop runs (H1 6, H2 3 + 4 finalist, H4 6; about 85 GPU-minutes) plus H3's 10 on
Kaggle: 29 of the 30 declared. Cycle 0's 5 OOF folds are counted separately in the ledger. The Kaggle
session also re-ran Tier E (10 runs, outside the budget) and reproduced every original row exactly;
the 19 laptop runs were regenerated after `results/` was lost (ADR-023).


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

### P6 — INT8 is blocked by the quality gate; latency is being re-measured
FP32 ONNX is released and verified (logit parity 8.2e-5, 100% label agreement on 1,583 sentences).
**INT8 does not pass** (ADR-022): dynamic INT8 keeps both majority classes and drops neutral F1
0.672 → 0.584 (macro −0.030, upper bound 0.054 against a 0.005 margin); static per-tensor INT8
breaks the model. The first Kaggle builds had failed for a different reason (quantizing a fused
graph), now fixed. The steady-state latency ladder, with the blocked INT8 graph timed only to answer
H3, is being re-measured after the `results/` loss (ADR-023). *Next if INT8 matters:* per-channel or
partial quantization selected on a held-out subset, or a distilled FP32 student (E15).

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
