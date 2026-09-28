# Status, Open Problems and Next Experiments

**Updated:** 2026-09-27, during research Cycle 2. Living document. Results live in
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
| G5 | Error analysis & robustness | 🔶 | Tooling, robustness suite, audit sheet and its analysis (`study audit-report`) done ([Cycle 0](#2-cycle-0--what-the-current-model-gets-wrong)); the human annotation is not |
| **G6** | CPU inference optimization | ✅ | FP32 ONNX released through the gate (ADR-020) and served (augmented model, ADR-027); latency ladder measured; INT8 blocked by the quality gate (ADR-022) |
| **G7** | Service, Docker, CI | ✅ | FastAPI, multi-stage image, CI; API now also tested against a real artifact (R4) |

### Research cycles (after external review)

| Cycle | Question | State |
|---|---|---|
| **0** | Fix validity problems; what does the current model get wrong? | ✅ [results/studies/](../results/studies/README.md) |
| **1** | Four hypotheses, declared before running ([cycle1.yaml](../configs/experiments/cycle1.yaml) v1 + v2) | ✅ all four decided; closing gate on test done (H2 and calibration confirmed) |
| **2** | Track A: confirmation on a frozen challenge set, the serving model, topic stacking, an LLM reference ([cycle2.yaml](../configs/experiments/cycle2.yaml) v2) | ✅ H5 not supported; H6 switched the served model; H7 decided: both LLM arms better on neutral on challenge v1 (construction caveat); on real posts only gpt-4o-mini is ahead |
| **3** | Reliability of the served model on real input ([cycle3.yaml](../configs/experiments/cycle3.yaml) v5) | 🔶 confirmed on NEU-ESC real student text (ADR-030): the service now lowercases, restores diacritics and reports an out-of-scope score (ADR-031); CE + restoration, the real-typing lexicon and careful INT8 fail their rules. ✅ H7 decided |
| **4** | Student text from other institutions ([cycle4.yaml](../configs/experiments/cycle4.yaml) v2) | ✅ H8 **not passed** (ADR-033): two heads (one per label policy) lift NEU-ESC test by +0.111 [+0.102, +0.119] at 5 seeds with UIT-VSFC within 0.005, but stripped text −0.0107 and the refitted out-of-scope score fail; the served model stays. ADR-032: the Mahalanobis `in_scope` tracked the institution; ADR-034 replaced it with a TF-IDF topic detector (NEU-ESC test AUROC 0.922, 7% of in-scope posts flagged). S8 (519 MB) and S10 (2.3 min) met. Careful INT8 passed S5′ but the release gate held it back (91% agreement, batch-dependent; ADR-036): S7 goes to distillation. Hub release updated to ADR-034 (2026-09-29, reproduced 0.8644 from the Hub). Owner item: neutral audit (👤) |

**Compute to date** ([ledger](../configs/experiments/ledger.csv)): 124 weight-updating runs, about 9.9 GPU-hours, of which 19 regenerated runs after the `results/` loss (ADR-023), plus 12 GPU-minutes of LLM inference. The registry holds 153 rows.

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

### Closing gate — the finalists on test, once (ADR-024)

| | Deployed CE | H2 augmented |
|---|---:|---:|
| Test macro-F1, 5 seeds (mean) | 0.8288 | 0.8296 |
| Paired, augmented − CE | — | **+0.0008** [−0.012, +0.013], 2/5 seeds |
| Seed 42: test macro-F1 / without the 55 train-overlapping rows | 0.8437 / 0.8431 | 0.8371 / 0.8364 |
| Seed 42, test: no diacritics | 0.271 | **0.635** |
| Seed 42, test: half the diacritics | 0.649 | **0.731** |
| Seed 42, test: teencode / char noise | 0.803 / 0.783 | 0.825 / 0.797 |
| Temperature fitted on validation → test NLL | 0.259 → **0.205** | 0.262 → **0.207** |
| → test ECE (15 bins) | 0.042 → **0.015** | 0.043 → **0.015** |

**H2 is confirmed on test**: clean accuracy unchanged, missing-diacritic robustness more than doubled.
**Calibration is confirmed on test**: a temperature fitted only on validation transfers. Removing the
train-overlapping test rows moves macro-F1 by at most 0.0007, so overlap does not inflate the headline.
Seven test evaluations were logged; none changes a Cycle 1 decision.

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

## 4. Cycle 2 — confirmation on new data

Declared in [cycle2.yaml](../configs/experiments/cycle2.yaml) before any Cycle 2 run (ADR-025). Version 2
(ADR-026) froze a 305-row **challenge set** by SHA-256 before any model saw it. No Cycle 2 decision
uses the official test.

| Hypothesis | Result | Decision by the declared rule |
|---|---|---|
| **H5** topic stacking (TF-IDF B4 × PhoBERT, out-of-fold meta-model) | Stacked − PhoBERT 5-fold ensemble −0.0102 [−0.0228, +0.0013]; `facility` and `others` both lower | **Not supported.** The "TF-IDF wins facility" gap reverses under a paired interval: −0.021 [−0.060, +0.015] |
| **H6** serve the H2-augmented model? | Constructed typed-style noise (90 rows): accuracy 0.611 → 0.800, +0.189 [+0.067, +0.311]; other rows unchanged (0.867) | **Switched** (ADR-027), released through the FP32 gate. Possible regressions, not significant at one seed (ADR-028): teencode 0.975 → 0.875 (p = 0.13); short factual sentences 0.767 → 0.633 (p = 0.22) |
| **H7 pilot** Qwen3-1.7B, label likelihood, frozen prompt | Validation macro-F1 0.68 / 0.65 (0- / 6-shot) vs 0.86; neutral F1 0.26–0.29 vs 0.66; over-calls neutral (precision 0.16–0.18) | Pilot only; the declared arms are below (H7 decision) |

**What Cycle 2 establishes so far.**

1. **H2's robustness transfers to unaccented text the augmentation code did not produce** (0.32 → 0.74,
   p < 0.001), but **not demonstrably to teencode** beyond its 14-entry map (0.975 → 0.875, p = 0.13 at one
   seed). Neither set was typed by real users (ADR-028).
2. **The serving decision rests on new data, and its costs are named.** The pooled rule passed. The
   per-category view shows a possible neutral-recall cost on short factual text (p = 0.22 at one seed,
   to be checked at 5 seeds), and the next rule will require per-category non-inferiority.
3. **Stacking does not help topic**, and the observation that motivated it was noise.
4. **A small LLM fails where the encoder succeeds, and the reverse.** It rejects the corpus's policy
   (suggestions and contrasts as polar), yet labels factual sentences correctly where the encoder
   drifts. The two error sets are the audit's two hypotheses, label policy and representation.
5. **Off-topic input is not flagged.** Both encoders label it with mean confidence 0.86–0.90.

### Cycle 3 — reliability on real input (in progress)

Declared in [cycle3.yaml](../configs/experiments/cycle3.yaml) (v1–v5, ADR-029, ADR-030). Evidence
ranking and the evaluation matrix: [EVALUATION_DATA.md](EVALUATION_DATA.md). Challenge v1 is
development data now; NEU-ESC test (real forum posts, human labels) replaces the human-typed challenge
v2 as the confirmation set (ADR-030), with rules committed before the cells were computed.

| Step | Result | Decision |
|---|---|---|
| V1: Cycle 2 drops at 5 seeds | Teencode 0.935 → 0.900 (3/5, p = 0.09); short factual 0.747 → 0.660 (3/5, p = 0.02); contrast 0.875 → 0.800 (4/5, p = 0.004); unaccented 0.33 → 0.66 (5/5 better) | Teencode: noise. Contrast: confirmed drop (exploratory). Short factual: unconfirmed |
| Case invariance | Capitalized input flipped 1.07% of labels (the corpus has no uppercase letter) | **Service lowercases input** (rule met; 0 flips; no validation change) |
| ViLexNorm invariance | 17% of labels flip between real comments and their human normalization; augmentation no help (p = 0.93) | Measured |
| S2a: lexicon from real typing | ViLexNorm flips 0.169 → 0.177 (p = 0.13); validation +0.003 | **Not passed**: not a candidate |
| S2b: diacritic restoration | Development: stripped validation 0.686 → 0.857; clean predictions unchanged; +0.2 ms. **NEU-ESC**: unaccented posts 0.270 → 0.374 (+0.104 [+0.091, +0.118]); posts as written −0.0006 | **Confirmed → served** (ADR-031) |
| S2b′: CE + restoration (v4) | Development, 5 seeds: unaccented 0.904 vs 0.880; contrast 0.875 vs 0.800 (p = 0.004). **NEU-ESC**: contrast 0.440 vs 0.437 (p = 0.68); overall +0.028, unaccented +0.036, short neutral +0.033 | **Not passed**: the contrast advantage did not replicate; the augmented model stays |
| S3: out-of-scope score | Development: Mahalanobis AUROC 0.949 vs max-probability 0.862. **NEU-ESC** off-topic posts (563): AUROC 0.977 vs 0.920; 86% flagged at 5% of validation | Served as `in_scope` (ADR-031); **ADR-032: it tracks the institution, not the topic** (within NEU-ESC AUROC 0.573; 77% of in-scope posts flagged) |
| S5: careful INT8 | 178.5 MB, macro-F1 drop +0.0004, neutral 0.661 → 0.667, upper bound 0.0095; p50 about 6.4 ms vs 11.1 ms (indicative: no steady pass) | **Not passed** (0.005 margin not demonstrable with 73 neutral examples) |
| **H7 decision** (Holm over two zero-shot arms, `cycle3.yaml` v6) | Challenge v1 neutral F1 minus CE: Qwen3-4B +0.187 [+0.106, +0.277], gpt-4o-mini +0.241 [+0.164, +0.329], Holm p 0.0004 each. Qwen3-4B elsewhere: UIT-VSFC validation 0.816 (encoder 0.86); NEU-ESC 0.475, served model +0.014 [−0.001, +0.028], gpt-4o-mini −0.128; calls 3,923 posts negative (gold 1,221) | **llm_better_on_neutral**, on constructed text only. A local 4B model does not close the real-text gap: the router is not worth declaring (NEXT_PLAN v5) |
| H7 API arm | gpt-4o-mini zero-shot. Challenge v1: neutral F1 0.955 vs 0.713 (construction confound). **NEU-ESC** (6,613 real posts): macro-F1 **0.604** vs CE 0.494 and augmented 0.462 (seed 42; 5-seed means 0.463 / 0.434); neutral F1 0.769 vs 0.526 / 0.474; over-calls negative (precision 0.43). USD 0.21, about 1.1 s per call | Recorded: the LLM transfers across the domain shift better; not in the serving path |
| NEU-ESC (real forum posts, Tier B) | macro-F1 CE 0.463 · augmented 0.434 · S2a 0.433 (5 seeds each); predicted neutral 30% vs gold 69%; augmented − CE at s42 −0.033 [−0.042, −0.024] | Domain and label-policy shift is large; augmentation hurts on real text: evidence for CE + restoration |

---

## 5. Open problems

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

### P4 — Topic is the weaker task, and two interventions did not help
Topic PhoBERT + pyvi: 0.8111 dev / 0.8038 test, against sentiment's 0.864 / 0.829. Segmentation helps
topic too (+0.0141, p = 0.004). Multi-task (H4) gave no material gain, and TF-IDF × PhoBERT stacking
(H5) lost 0.010. The apparent TF-IDF advantage on `facility` was noise. `others` (F1 about 0.6) is
the weak class. Cycle 3's review (`results/studies/topic_others.json`): recall 0.53, precision 0.76;
missed `others` rows go to `lecturer` (27) and `training_program` (17), the neutral pattern again: a
residual minority class absorbed by the majority. *Open:* include `others` in the audit design.

### P5 — Robustness to informal orthography is a deployment risk
Cycle 0 measured it; H2's augmentation is now served (ADR-027). On constructed typed-style text it lifts
unaccented accuracy from 0.32 to 0.74 (p < 0.001); teencode moves 0.975 → 0.875, not significant at one seed. The no-diacritic failure is a
whole-pipeline failure: pyvi also mis-segments unaccented text. A diacritic-restoration front-end
remains the untested alternative (NEXT_PLAN P5, conditional).

### P6 — Latency measured; INT8 is faster but blocked by the quality gate
Reference CPU (Ryzen 5 6600H, AVX2, no AVX512-VNNI), single sentence, model-only on pre-segmented
input, two steady passes in rotated order agreeing within 1% (`results/studies/latency/reference_cpu.json`):

| Configuration | p50 | p95 | vs L0, p50 / p95 | texts/s (b=32) | Quality |
|---|---:|---:|---|---:|---|
| L0 PyTorch FP32, pad to 96 | 117.4 ms | 120.5 ms | 1× | 12.0 | reference |
| L1 PyTorch FP32, dynamic padding | 46.7 ms | 64.1 ms | 2.5× / 1.9× | 24.1 | identical |
| **L3 ONNX FP32 (served)** | **15.5 ms** | **33.6 ms** | **7.6× / 3.6×** | 24.9 | identical (parity 8.2e-5) |
| L4 ONNX INT8 dynamic (blocked) | 9.0 ms | 20.4 ms | 13.1× / 5.9× | 40.9 | neutral F1 −0.088: not released |

- **Serve FP32 ONNX**: 7.6× faster than padded PyTorch at the median, identical predictions.
  S5 (p95 ≤ 60 ms minimum, ≤ 30 ms target): **both met.** The one miss (33.6 ms, the table above) was
  a slow machine state: PyTorch L0 was 117 ms there against 70–75 ms in three later sessions on AC
  power, where the served model's p95 was 19.2 and 20.0 ms in the two reportable sessions (one
  session had no steady pass; its passes read 18.7 and 20.7 ms). Files:
  `results/studies/latency/sessions/`. Absolute latency depends on the laptop's state, and so does the
  p50 ratio (6.3× in the later sessions vs 7.6× here), so claims quote a range.
- **H3 is falsified.** INT8 was expected to be possibly *slower* without VNNI. It is 1.7× faster than
  FP32 ONNX on AVX2. It is still not shipped: it costs 0.088 neutral F1 (ADR-022). The review's
  efficiency stretch target (≥ 1.5× with neutral loss ≤ 0.02) is therefore not met.
- Peak process memory with all four models loaded: 2.5 GB.

### P7 — Reproducibility is pinned
Run identity includes a config hash, and run directories cannot be overwritten by a different
configuration (R11). Per-run `metrics.json`/`config.yaml`/`env.json` are committed. Model and dataset
revisions are pinned in code (`constants.MODEL_REVISIONS`, `loader.PARQUET_REVISION`). Kaggle runs,
which carry no git SHA, are resolved to their commit through their source hash
(`results/provenance.json`): the 21 Kaggle runs executed 5b2858c. Two cRT runs executed an
uncommitted tree; that is recorded, not hidden.

| Artifact | Pinned revision |
|---|---|
| `vinai/phobert-base` | `01daacda68afe13d83023d16ec647239e344a1e6` |
| `vinai/phobert-large` | `70e2cfcd3cce29c970aee4954ea34a32bb30afdc` |
| `FacebookAI/xlm-roberta-base` | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` |
| UIT-VSFC `refs/convert/parquet` | `2d76906157232f87e9883fa37ebfc91ffdee1f83` |

---

## 6. External review — item status

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
| R8 model comparison closed too early | Medium | ✅ narrowed (ADR-019); tokenizer profiles measured; XLM-R raw control decided (H3: ~40% of the gap was preprocessing) |
| R9 headline vs deployed config | Medium | ✅ reported separately in README |
| R10 docs and benchmark out of sync | Medium | ✅ docs synced; latency ladder re-measured with steady passes (P6) |
| R11 reproducibility pinning | Medium | ✅ run ids, overwrite guard, revisions pinned, source hash, lock file, data content reference, CI integration job |
| R12 benchmark claims | Medium | ✅ no SOTA claim; BamiBERT reported as context |

---

## 7. Next

Every unmet target, hole, unrun experiment and unconfirmed result, with where it is handled:
**[NEXT_PLAN.md](NEXT_PLAN.md) v5 § 2**, one prioritized work list. Cycle 4 (student text from other institutions) is its Track B.
Needs the owner:

1. **H7 is done**: Qwen3-4B ran on Kaggle (both GPUs, prefix cache, 19–28 s per 1k) and
   `study h7-decide` applied the rule. UIT-VSFC text was never sent to the API.
2. **The human audit** (P2/P3), which gates E04. `study audit-report` analyses the filled sheet.
3. **The model is published** (2026-09-28): <https://huggingface.co/Datk4/vifeedback-sentiment-phobert>, CC BY-NC 4.0 weights, ONNX + PyTorch,
   restorer and out-of-scope parameters, every file checked against `SHA256SUMS`.

---

## 8. Compute plan

| Work | Where | Why |
|---|---|---|
| PhoBERT-base training, OOF, robustness, calibration, FP32 export | **Laptop RTX 3050** | ~65 s/epoch, no session limits |
| Full-embedding XLM-R, PhoBERT-large, INT8 quantization | **Kaggle** | > 4.29 GB, or needs the blocked `onnx` package (ADR-017/020) |
| **All latency benchmarking** | **Laptop only, idle** | the reference machine is recorded in `env.json`; no benchmark while training runs |
| LLM pilot (Qwen3-1.7B, fp16) | Laptop | fits 4 GB; about 70 s per 1k sentences zero-shot |
| Declared LLM (Qwen3-4B) | **Kaggle** (cell 4f) | 8 GB in fp16 |

Notebook: [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb) (§ 4e for Cycle 1, § 4f for Cycle 2 H7) and
[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md).
