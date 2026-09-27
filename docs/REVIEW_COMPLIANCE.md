# Review Compliance — every recommendation, its status, and its evidence

Traceability from [REVIEW_AND_RESEARCH_PLAN.md](REVIEW_AND_RESEARCH_PLAN.md) to what exists in the
repository. **Audited 2026-09-27, during Cycle 1; updated the same day after the gaps it found were worked.** Each status points to a file, commit or result,
not to intent.

| Mark | Meaning |
|---|---|
| ✅ | Done; evidence linked |
| 🔶 | Partly done, or running; the missing part is named |
| ⬜ | Not done, but inside the current scope |
| ⏸ | Not done, deliberately: the review defers it, or it belongs to the Cycle 2 choice |
| 👤 | Needs the owner: human annotation, new data, or an account-level action |

## Scoreboard

| Part of the review | ✅ | 🔶 | ⬜ | ⏸ | 👤 |
|---|---:|---:|---:|---:|---:|
| § 2 Issues R1–R12 | 10 | 2 | 0 | 0 | 0 |
| First five concrete tasks | 5 | 0 | 0 | 0 | 0 |
| Recommended starting point (4 items) | 2 | 2 | 0 | 0 | 0 |
| Implementation backlog (12 items) | 8 | 1 | 0 | 3 | 0 |

**In one sentence:** every validity problem the review raised is fixed or has a named remaining step,
Cycle 0 is complete, and Cycle 1 is running. What is *not* done falls into three groups: one item from
the review's recommended starting point (the shared sentiment/topic encoder), the human annotation
work, and the Cycle 2 specialization, which the review says to choose after Cycle 1, not run in
parallel.

---

## § 2 — Issues to resolve (R1–R12)

| Item | Sev. | Status | Evidence | Remaining |
|---|---|---|---|---|
| R1 segmentation paper misread | High | ✅ | ADR-018; README "an independent replication, not a refutation" | — |
| R2 FGM + AMP gradient bug | High | ✅ | `trainer.py` unscales once per step; 6 tests in `tests/integration/test_training_loop.py` | FGM sweep itself not run (E06, ⏸) |
| R3 export quality contract | High | ✅ | `inference/release.py`, ADR-020: staging, manifest + SHA-256, preprocessing-aware calibration/acceptance, full-dev acceptance, FP32 logit parity 8.2e-5 measured | INT8 release runs on Kaggle (§4d); INT8 latency on the reference CPU |
| R4 API never tested with a model | High | ✅ | `tests/contract/test_api_with_model.py`: real artifact, golden cases, manifest file check | CI has no artifact, so it skips there; a tiny-artifact smoke test for CI is not built |
| R5 readiness / fallback semantics | Med | ✅ | `/readyz` is HTTP 503 unless every `REQUIRED_TASKS` model and its manifest's segmenter are loaded; no silent raw-text fallback. The fallback cost was **measured**: −0.052 macro-F1, not the documented −0.023 (`results/studies/robustness/raw_input_fallback.json`) | — |
| R6 unbounded metrics buffer | Med | ✅ | Bounded `deque(maxlen=10_000)` plus a separate request counter; test pins the bound | Multi-worker aggregation design (single worker today) |
| R7 over-strong statistical claims | High | ✅ | README/STATUS interval wording; seed-level and bootstrap both reported; no equivalence claimed from p > 0.05 | Cross-architecture seed pairing is labelled "pairs data order only" in the comparison family (below) |
| R8 architecture search closed too early | Med | 🔶 | ADR-019 narrows ADR-016; tokenizer length profiles measured (`results/studies/tables/tokenizer_length_profiles.json`: XLM-R produces 36% more subwords on pyvi input) | XLM-R raw control running on Kaggle (Cycle 1 H3); comparable tuning budgets (LR sweep) not run |
| R9 headline vs deployed config | Med | ✅ | README reports VnCoreNLP 0.8373 and pyvi 0.8288 separately; CV snippet quotes each correctly | — |
| R10 docs and benchmark out of sync | Med | 🔶 | STATUS rewritten; run counts from the registry; texts/s naming; model-only on preprocessed input; instability no longer called "throttling" | Steady-state benchmark re-run on an idle machine (after Cycle 1's GPU runs) |
| R11 reproducibility pinning | Med | ✅ | Config hash + overwrite guard; per-run metrics/config/env and validation predictions committed; model and dataset commits pinned; `source_sha256` in env.json; `requirements-lock.txt`; committed content-hash data reference verified on fetch; CPU training-integration tests in CI | Published checkpoint with checksum (👤 HF account); mypy still advisory |
| R12 benchmark comparison claims | Med | ✅ | "No state-of-the-art claim is made"; BamiBERT reported as context incl. topic F1 79.90 | — |

## First five concrete tasks (§ 12)

| # | Task | Status | Evidence |
|---|---|---|---|
| 1 | Correct the paper interpretation; separate VnCoreNLP/pyvi | ✅ | ADR-018, README Result table |
| 2 | Validate or disable FGM; protect run artifacts | ✅ | R2 tests; config hash + `_assert_not_clobbering` |
| 3 | Stratified train/dev error sample with predictions and uncertainty; audit guide | ✅ | `results/studies/study_a/` (160-row sheet, OOF over train), [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) |
| 4 | Freeze a baseline, the calibration protocol, the robustness slices | ✅ | `configs/experiments/cycle1.yaml` `common`; EVALUATION_PROTOCOL § 8; `SLICES_VERSION = "1"` |
| 5 | Three hypotheses, only the needed controls, within budget | ✅ | `cycle1.yaml` committed before the first run (0f3055d); ledger |

## Recommended starting point (header of the review)

| Item | Status | Evidence / remaining |
|---|---|---|
| Complete the validity fixes | ✅ | R-table above |
| Investigate neutral errors | 🔶 | Study A diagnosis done (confident errors, boundary test, OOF ranking); **the human audit is not** (👤) |
| Evaluate calibration / robustness | ✅ | `results/studies/calibration/`, `results/studies/robustness/` |
| **Test one shared sentiment/topic encoder** | 🔶 | Was missing from Cycle 1 v1. Now declared as H4 in `cycle1.yaml` v2 (ADR-021), before any H4 run; `training/multitask.py` built and tested |

## Implementation backlog (§ 12)

| Module / artifact | Priority | Status |
|---|---|---|
| `evaluation/compare_runs.py` | P0 | ✅ tables + seed-paired comparisons + BH, duplicate/re-run hygiene |
| `configs/experiments/` | P0 | ✅ `cycle1.yaml`, `ledger.csv` |
| `evaluation/error_analysis.py` | P1 | ✅ stratified sampling, OOF, uncertainty, audit export, boundary test |
| `evaluation/calibration.py` | P1 | ✅ cross-fitted temperature, NLL/Brier/ECE (two binnings), class-wise ECE, risk–coverage by class |
| `evaluation/robustness.py` | P1 | ✅ versioned suites, grouped bootstrap, slices, negation probe |
| `results/studies/` | P1 | ✅ Cycle 0 outputs + README |
| `docs/ANNOTATION_GUIDE.md` | P1 | ✅ |
| `training/multitask.py` | P1 if Q4 | ✅ built and unit-tested; runs are Cycle 1 H4 |
| `training/low_resource.py` | P2 | ⏸ Cycle 2 (data-efficiency branch) |
| `training/distillation.py` | P2 | ⏸ Cycle 2 (efficient-modelling branch) |
| `evaluation/llm_reference.py` | P2 | ⏸ Cycle 2 (AI-engineering branch) |
| `docs/RESEARCH_REPORT.md` | Final | 🔶 [drafted](RESEARCH_REPORT.md) from settled results; H3, H4 and the H2 5-seed confirmation fill in as they land |

## § 3 Research questions

| Q | Status |
|---|---|
| Q1 What limits neutral? | 🔶 Boundary explanation weakened (Study A); classifier-head explanation under test (H1: logit adjustment, cRT); label-ambiguity explanation needs the audit (👤) |
| Q2 What transfers beyond clean text? | 🔶 Synthetic shift measured; augmentation under test (H2); no natural external set (👤 data) |
| Q3 Label efficiency | ⏸ Cycle 2 option |
| Q4 Shared learning | 🔶 H4 declared; running next |
| Q5 Uncertainty | ✅ Calibration, class-wise coverage, AURC for three signals; OOD detection ⏸ (no OOD set) |
| Q6 Compute budget | 🔶 FP32 release verified; INT8 on Kaggle; latency re-run pending; distillation/LoRA ⏸ |

## § 4 Dataset extensions

| Recommendation | Status |
|---|---|
| Keep the official split; add a predefined slice **excluding the 55 train-overlapping test rows** | ✅ frozen in `configs/data/eval_slices_v1.json`; evaluated once at the closing gate |
| Audit 100–200 train/dev examples, OOF-ranked, with a random component | 🔶 sheet + guide ready; annotation 👤 |
| Neutral taxonomy (facts / no opinion / requests / mixed / insufficient context) | 🔶 defined in the guide; counts need the audit 👤 |
| Challenge set of 300–500 sentences | 🔶 only the 44-pair negation probe exists |
| Independent naturally sampled set (500–1,000) | 👤 needs new data |
| Multi-aspect education set; UIT-ViSFD ABSA | ⏸ Stage C / Cycle 2 options |
| Separate input-variation / domain-transfer / out-of-scope shift | 🔶 input variation done; the other two ⏸ |

## § 5–6 Model ladder and experiment catalog

| ID | Pri. | Status |
|---|---|---|
| E01 ranking depends on preprocessing/tuning | P0 | 🔶 XLM-R raw control running on Kaggle (H3); LR candidates not run |
| E02 imbalance (logit adjustment) | P1 | 🔶 τ = 1 at 3 seeds done; decision pending the declared rule |
| E03 balanced head retraining (cRT) | P1 | ✅ 3 seeds: +0.0028 mean, 2/3 wins → does not advance under the declared rule |
| E04 label noise / audit | P1 | 👤 |
| E05 augmentation | P1 | 🔶 running (H2) |
| E06 regularization (LLRD, R-Drop, FGM) | P2 | ⏸ |
| E07 multi-task | P1 | 🔶 H4 declared (v2), running next |
| E08 sparse + dense stacking | P2 | ⏸ |
| E09 calibration | P1 | ✅ on validation/OOF; test confirmation at the closing gate |
| E10 selective prediction | P1 | ✅ risk–coverage by class; AURC for max-prob / margin / entropy (equal, ~77% of the random-to-oracle gap); OOD false acceptance ⏸ |
| E11–E22 | P2–P3 | ⏸ Cycle 2 choice |
| Models: TF-IDF, PhoBERT(VnCoreNLP/pyvi), PhoBERT-large | — | ✅ |
| Models: XLM-R raw | Required | 🔶 H3 |
| Models: BamiBERT, ViSoBERT, e5-small, SetFit, Qwen3-4B, student | High/Opt. | ⏸ The review says to start with two new encoder controls, not all of them |

## § 7 Study designs

| Study | Status |
|---|---|
| A — neutral | Steps 1, 3 ✅; step 2 guide ✅, annotation 👤; step 4 ✅ (CE vs logit adjustment vs cRT); step 5 depends on the audit |
| B — robustness | Synthetic layer ✅ with grouped bootstrap and CheckList-style invariance/directional split; PhoBERT control ✅, augmentation 🔶 (H2), alternative encoder 🔶 (XLM-R in-run on Kaggle); natural external layer 👤 |
| C — learning efficiency | ⏸ |
| D — multi-task | 🔶 H4 declared (v2), running next |
| E — encoder vs LLM | ⏸ |

## § 8 Evaluation protocol

| Rule | Status |
|---|---|
| Test is "already inspected", never called untouched | ✅ wording says "untouched *until* G4" |
| Separate training / selection / calibration / confirmation | 🔶 calibration is cross-fitted, but on the same validation set used for epoch selection (disclosed); the OOF view is fully independent |
| Register hypotheses, controls, metrics, budget before a sweep | ✅ `cycle1.yaml` |
| Save config, source hash, versions, model revision, data hash, seed, predictions, metrics | ✅ except checkpoint hash per run (release manifests have one) |
| Log failed / discarded runs | ✅ ledger (none failed so far) |
| Paired example-level intervals for model comparisons | 🔶 validation predictions are now committed and cRT runs save their stage-1 control predictions, so this is possible from here on; Cycle 1's first runs predate it |
| Cross-architecture pairing caveat | ✅ every comparison carries a `pairing` label |
| Quantization judged by a non-inferiority margin with uncertainty | ✅ release requires the one-sided 95% upper bound of the paired macro-F1 drop ≤ 0.005 |
| Latency: RSS, rotation of configuration order | ⬜ |
| Track decisions influenced by test results | ✅ EVALUATION_PROTOCOL § 4 records it: none so far |

## Found by this audit, outside the review's list

| Finding | Status |
|---|---|
| **CI had failed on every push**: the quality job installed only `.[dev]`, `serving/` imports fastapi, so the import test failed and the unit tests after it never ran; the Docker job, which depends on it, was skipped | ✅ fixed in f5a45b4; all four jobs green, Docker smoke included |
| The runtime image would crash at start-up if `SEGMENTER` were unset (an import pulled in pandas) | ✅ import-free mapping, pinned by a test |
| README listed the TF-IDF baseline's neutral F1 as 0.3530; the artifact says 0.4207 (0.35 is the un-tuned run), and the H1 claim was computed from it | ✅ corrected; H1 now states test-split figures |

## § 9 Success criteria (current evidence)

| Area | Minimum defensible | Status |
|---|---|---|
| Reproducibility | Recreate baseline from pinned config; regenerate tables | ✅ stage 1 of every cRT run reproduces the registry exactly; tables regenerate from the registry |
| Scientific contribution | ≥ 3 questions answered with controls | 🔶 segmentation, calibration, boundary test, cRT answered; H2/H3 pending |
| Minority quality | Explain dominant neutral errors; evaluate a targeted intervention | 🔶 explained up to the label question; cRT evaluated (neutral recall ↑, precision ↓, macro-F1 ≈) |
| Robustness | Clean and challenging-slice results with support | ✅ |
| Uncertainty | Calibrated vs uncalibrated on independent data | ✅ |
| Efficiency | Reproducible quality/latency/memory comparison | 🔶 quality parity ✅; latency/memory pending |

## § 11 Portfolio checklist

| Item | Status |
|---|---|
| Problem statement with 3–5 research questions | 🔶 in README "Research cycles"; an explicit RQ list belongs in the report |
| Strong sparse and neural baselines | ✅ |
| ≥ 3 controlled investigations incl. a negative/inconclusive one | ✅ segmentation (+), Tier E (negative, narrowed), cRT (inconclusive), boundary test (negative) |
| Manually inspected error taxonomy of 60–100 cases | 👤 |
| One robustness evaluation beyond the aggregate | ✅ |
| Seed variation vs evaluation uncertainty kept apart | ✅ |
| One extension completed deeply | ⏸ Cycle 2 |
| Executed notebooks, figures, artifact checksums | ✅ `02_results.ipynb` rebuilt and executed; release manifest has SHA-256 |
| Short technical report | ⏸ at Cycle 1 close |

---

## What happens next

**Now, on the laptop, while Kaggle runs H3:**

1. Cycle 1 H4, shared sentiment/topic encoder: declared as `cycle1.yaml` v2 *before* its runs,
   `training/multitask.py`, λ ∈ {0.3, 1} × 3 seeds, fitted into the remaining budget.
2. Cheap protocol gaps: the overlap-excluded test slice definition; AURC and a margin/entropy
   comparison for E10; CI-based INT8 non-inferiority; the cross-architecture pairing label; saving
   stage-1 predictions; R6 bounded buffer and R5 readiness semantics (small, though deferred by scope).
3. Steady-state latency benchmark once the GPU is idle.

**Needs you (👤):**

- The neutral-label audit: `results/studies/study_a/local/audit_sheet.csv` following
  [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md), ideally with a second annotator on ≥ 50 rows.
- Kaggle §4e (H3) and, optionally, §4d (INT8 releases).
- Any new, naturally sampled evaluation data, if available.
- Publishing a checkpoint (HF Hub) needs your account; the release manifest is already checksummed.

**Then:** the Cycle 1 closing gate (one logged test evaluation per finalist, calibrated vs
uncalibrated), `docs/RESEARCH_REPORT.md`, and the choice of one Cycle 2 specialization.
