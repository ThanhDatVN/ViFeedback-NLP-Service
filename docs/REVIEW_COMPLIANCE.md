# Review Compliance — every recommendation, its status, and its evidence

Traceability from [REVIEW_AND_RESEARCH_PLAN.md](REVIEW_AND_RESEARCH_PLAN.md) to what exists in the
repository. **Audited 2026-09-27 during Cycle 1; re-audited the same day during Cycle 2** (NEXT_PLAN v2, item I3). Each status points to a file, commit or result,
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
| § 2 Issues R1–R12 | 12 | 0 | 0 | 0 | 0 |
| First five concrete tasks | 5 | 0 | 0 | 0 | 0 |
| Recommended starting point (4 items) | 3 | 1 | 0 | 0 | 0 |
| Implementation backlog (12 items) | 10 | 0 | 0 | 2 | 0 |

**In one sentence:** every validity problem the review raised is fixed, Cycles 0 and 1 are closed, and
Cycle 2 (track A) has decided two of its three hypotheses on a frozen challenge set. What is *not*
done needs the owner: the human annotation, the declared LLM runs (Kaggle, API key), new natural data,
and the Hugging Face upload.

---

## § 2 — Issues to resolve (R1–R12)

| Item | Sev. | Status | Evidence | Remaining |
|---|---|---|---|---|
| R1 segmentation paper misread | High | ✅ | ADR-018; README "an independent replication, not a refutation" | — |
| R2 FGM + AMP gradient bug | High | ✅ | `trainer.py` unscales once per step; 6 tests in `tests/integration/test_training_loop.py` | FGM sweep itself not run (E06, ⏸) |
| R3 export quality contract | High | ✅ | `inference/release.py`, ADR-020: staging, manifest + SHA-256, preprocessing-aware calibration/acceptance, full-dev acceptance, FP32 logit parity 8.2e-5 measured; the served augmented model passed the same gate (1.7e-5, ADR-027) | — (INT8 built and blocked by the gate, ADR-022) |
| R4 API never tested with a model | High | ✅ | `tests/contract/test_api_with_model.py`: real artifact, golden cases, manifest file check; `make docker-e2e` runs the image with the released artifact mounted (`/readyz` 200, SHA-256 inside the container, golden labels) | CI has no artifact, so both run locally only |
| R5 readiness / fallback semantics | Med | ✅ | `/readyz` is HTTP 503 unless every `REQUIRED_TASKS` model and its manifest's segmenter are loaded; no silent raw-text fallback. The fallback cost was **measured**: −0.052 macro-F1, not the documented −0.023 (`results/studies/robustness/raw_input_fallback.json`) | — |
| R6 unbounded metrics buffer | Med | ✅ | Bounded `deque(maxlen=10_000)` plus a separate request counter; test pins the bound | Multi-worker aggregation design (single worker today) |
| R7 over-strong statistical claims | High | ✅ | README/STATUS interval wording; seed-level and bootstrap both reported; no equivalence claimed from p > 0.05 | Cross-architecture seed pairing is labelled "pairs data order only" in the comparison family (below) |
| R8 architecture search closed too early | Med | ✅ | ADR-019 narrows ADR-016; tokenizer profiles (+36% subwords on pyvi input); XLM-R raw control run (H3): +0.0097 over pyvi, still −0.014 vs PhoBERT-base | Per-model LR tuning not run (Cycle 2 option) |
| R9 headline vs deployed config | Med | ✅ | README reports VnCoreNLP 0.8373 and pyvi 0.8288 separately; CV snippet quotes each correctly | — |
| R10 docs and benchmark out of sync | Med | ✅ | STATUS rewritten; texts/s; model-only on pre-segmented input; rotated two-pass benchmark reporting steady passes only (`results/studies/latency/reference_cpu.json`); cross-session variation disclosed, ratios claimed | — |
| R11 reproducibility pinning | Med | ✅ | Config hash + overwrite guard; per-run metrics/config/env and validation predictions committed; model and dataset commits pinned; `source_sha256` in env.json; `requirements-lock.txt`; committed content-hash data reference verified on fetch; CPU training-integration tests in CI | Publishing: bundle, SHA256SUMS and a model card built from committed results (`serve publish`, dry run); the upload needs the owner's HF account 👤. mypy is now blocking in CI; Kaggle runs resolved to commits (`results/provenance.json`) |
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
| **Test one shared sentiment/topic encoder** | ✅ | Declared as H4 (`cycle1.yaml` v2, ADR-021) before its runs; λ = 0.3: no material difference, λ = 1: negative transfer on sentiment (`results/studies/cycle1/decisions.json`) |

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
| `evaluation/llm_reference.py` | P2 | ✅ label likelihood, HF and OpenAI backends, prompt frozen on a train subset, data-egress rule; pilot run (Cycle 2 H7) |
| `docs/RESEARCH_REPORT.md` | Final | ✅ [written](RESEARCH_REPORT.md) from settled results, Cycles 0–2, with an explicit research-question list |

## § 3 Research questions

| Q | Status |
|---|---|
| Q1 What limits neutral? | 🔶 Boundary and classifier-head explanations weakened (Study A; H1: logit adjustment and cRT both +0.003, not advanced). The H7 pilot splits the errors along the audit's two hypotheses (the LLM rejects the label policy; the encoder misses factual text). Deciding between them needs the audit (👤; analysis automated) |
| Q2 What transfers beyond clean text? | 🔶 Synthetic shift measured; augmentation supported at 5 seeds (H2) and confirmed on constructed unaccented text the augmentation code did not produce (0.32 → 0.74, H6); teencode not demonstrated (0.975 → 0.875, p = 0.13 at one seed); no real user typing yet; no natural external sample (👤 data) |
| Q3 Label efficiency | ⏸ Cycle 2 option |
| Q4 Shared learning | ✅ H4 decided at 3 seeds: no gain at λ = 0.3, sentiment cost at λ = 1 |
| Q5 Uncertainty | ✅ Calibration, class-wise coverage, AURC for three signals. Out-of-scope input measured on 20 challenge rows: max-probability does not flag it (mean 0.86–0.90); OOD detection ⏸ |
| Q6 Compute budget | ✅ quality/latency/memory ladder: FP32 ONNX 7.6× faster than padded PyTorch at the median, identical output; INT8 1.7× faster again but blocked (neutral −0.088); distillation/LoRA ⏸ |
| Q7 Sparse + dense topic (added in Cycle 2) | ✅ H5 not supported: stacking −0.010; the facility gap was noise |
| Q8 Encoder vs LLM (added in Cycle 2) | 🔶 H7 pilot: Qwen3-1.7B neutral F1 0.26–0.29 vs 0.66; declared Qwen3-4B and gpt-4o-mini runs 👤 |

## § 4 Dataset extensions

| Recommendation | Status |
|---|---|
| Keep the official split; add a predefined slice **excluding the 55 train-overlapping test rows** | ✅ frozen in `configs/data/eval_slices_v1.json`; evaluated once at the closing gate |
| Audit 100–200 train/dev examples, OOF-ranked, with a random component | 🔶 sheet + guide ready; annotation 👤 |
| Neutral taxonomy (facts / no opinion / requests / mixed / insufficient context) | 🔶 defined in the guide; counts need the audit 👤 |
| Challenge set of 300–500 sentences | ✅ 305 rows in 10 categories, checked against the corpus, frozen by SHA-256 before evaluation (`data/challenge/`, ADR-026) |
| Independent naturally sampled set (500–1,000) | 👤 needs new data |
| Multi-aspect education set; UIT-ViSFD ABSA | ⏸ Stage C / Cycle 2 options |
| Separate input-variation / domain-transfer / out-of-scope shift | 🔶 input variation done (synthetic and constructed typed-style); out-of-scope measured descriptively (20 rows); domain transfer ⏸ |

## § 5–6 Model ladder and experiment catalog

| ID | Pri. | Status |
|---|---|---|
| E01 ranking depends on preprocessing/tuning | P0 | ✅ preprocessing part answered (H3: yes, ~40% of XLM-R's gap); LR candidates not run |
| E02 imbalance (logit adjustment) | P1 | ✅ τ = 1, 3 seeds: +0.0029, not advanced; neutral recall ↑ precision ↓ |
| E03 balanced head retraining (cRT) | P1 | ✅ not advanced at 3 seeds (+0.0028); +0.0006 at 5 seeds (the extra seeds ran as H2 controls) |
| E04 label noise / audit | P1 | 👤 |
| E05 augmentation | P1 | ✅ supported at 5 seeds: `nodiacritic-50` degradation −43%, no material clean cost; out-of-family noise unchanged |
| E06 regularization (LLRD, R-Drop, FGM) | P2 | ⏸ |
| E07 multi-task | P1 | ✅ H4: λ = 0.3 no material difference; λ = 1 negative transfer on sentiment |
| E08 sparse + dense stacking | P2 | ✅ H5: not supported (−0.0102 [−0.0228, +0.0013]) |
| E09 calibration | P1 | ✅ validation/OOF, and confirmed on test at the closing gate (NLL −21%, ECE 0.042 → 0.015) |
| E10 selective prediction | P1 | ✅ risk–coverage by class; AURC for max-prob / margin / entropy (equal, ~77% of the random-to-oracle gap); OOD false acceptance ⏸ |
| E16 quantization/runtime | P2 | ✅ PyTorch → ORT FP32 → dynamic → static INT8 with parity, per-class loss, size, p50/p95, texts/s, RSS; INT8 blocked (ADR-022) |
| E11–E15, E17–E22 | P2–P3 | ⏸ Cycle 2 choice |
| Models: TF-IDF, PhoBERT(VnCoreNLP/pyvi), PhoBERT-large | — | ✅ |
| Models: XLM-R raw | Required | ✅ H3 |
| Models: Qwen3 (LLM reference) | Opt. | 🔶 Qwen3-1.7B pilot done; Qwen3-4B declared, Kaggle cell 4f 👤 |
| Models: BamiBERT, ViSoBERT, e5-small, SetFit, student | High/Opt. | ⏸ The review says to start with two new encoder controls, not all of them |

## § 7 Study designs

| Study | Status |
|---|---|
| A — neutral | Steps 1, 3 ✅; step 2 guide ✅, annotation 👤; step 4 ✅ decided (neither intervention advances); step 5 depends on the audit |
| B — robustness | Synthetic layer ✅ with grouped bootstrap and CheckList-style invariance/directional split; PhoBERT control ✅, augmentation ✅ (H2, 5 seeds), alternative encoder ✅ (XLM-R raw: no-diacritic 0.35 vs PhoBERT 0.27, augmented PhoBERT 0.65); constructed typed-style layer ✅ (challenge set, H6); real user typing 👤 (challenge v2); natural external layer 👤 |
| C — learning efficiency | ⏸ |
| D — multi-task | ✅ 3 seeds, both λ; joint exact match reported per run |
| E — encoder vs LLM | 🔶 pipeline, frozen prompt and pilot done; per-class, latency and cost reported; declared Qwen3-4B and API runs 👤 |

## § 8 Evaluation protocol

| Rule | Status |
|---|---|
| Test is "already inspected", never called untouched | ✅ wording says "untouched *until* G4" |
| Separate training / selection / calibration / confirmation | ✅ confirmation now on a frozen challenge set no model was selected on (Cycle 2). Calibration is still cross-fitted on the validation set used for epoch selection (disclosed); the OOF view is fully independent |
| Register hypotheses, controls, metrics, budget before a sweep | ✅ `cycle1.yaml`, `cycle2.yaml` (v2 before any evaluation) |
| Save config, source hash, versions, model revision, data hash, seed, predictions, metrics | ✅ except checkpoint hash per run (release manifests have one) |
| Log failed / discarded runs | ✅ ledger (none failed so far) |
| Paired example-level intervals for model comparisons | 🔶 validation predictions are now committed and cRT runs save their stage-1 control predictions, so this is possible from here on; Cycle 1's first runs predate it |
| Cross-architecture pairing caveat | ✅ every comparison carries a `pairing` label |
| Quantization judged by a non-inferiority margin with uncertainty | ✅ release requires the one-sided 95% upper bound of the paired macro-F1 drop ≤ 0.005 |
| Latency: RSS, rotation of configuration order | ✅ two passes in rotated order; peak RSS recorded |
| Track decisions influenced by test results | ✅ EVALUATION_PROTOCOL § 4: one so far, the serving switch (ADR-027), decided on the challenge set but counted because the test gate was known |

## Found by this audit, outside the review's list

| Finding | Status |
|---|---|
| **CI had failed on every push**: the quality job installed only `.[dev]`, `serving/` imports fastapi, so the import test failed and the unit tests after it never ran; the Docker job, which depends on it, was skipped | ✅ fixed in f5a45b4; all four jobs green, Docker smoke included |
| The runtime image would crash at start-up if `SEGMENTER` were unset (an import pulled in pandas) | ✅ import-free mapping, pinned by a test |
| README listed the TF-IDF baseline's neutral F1 as 0.3530; the artifact says 0.4207 (0.35 is the un-tuned run), and the H1 claim was computed from it | ✅ corrected; H1 now states test-split figures |
| mypy was advisory in CI (35 errors) | ✅ fixed; blocking since cde2174 |
| The challenge set's first recorded hash came from a CRLF working copy; a Linux checkout would not match | ✅ LF hash recorded before any evaluation; `.gitattributes` pins `eol=lf`; loader normalizes line endings |
| Few-shot result files stored the train demonstration sentences (corpus text) | ✅ caught before push: stored as train indices, the draw re-verified identical, the local commit amended; a test guards it |

## § 9 Success criteria (current evidence)

| Area | Minimum defensible | Status |
|---|---|---|
| Reproducibility | Recreate baseline from pinned config; regenerate tables | ✅ stage 1 of every cRT run reproduces the registry exactly; tables regenerate from the registry |
| Scientific contribution | ≥ 3 questions answered with controls | ✅ Q1 (partly, pending audit), Q2, Q4, Q5, Q7 answered with declared controls; R8's XLM-R control decided (H3); Q8 at pilot stage |
| Minority quality | Explain dominant neutral errors; evaluate a targeted intervention | ✅ minimum met: errors characterized, two targeted interventions evaluated (recall ↑, precision ↓, macro-F1 ≈); the +0.03 neutral-F1 stretch target is not met |
| Robustness | Clean and challenging-slice results with support | ✅ minimum met; **stretch target met**: a predefined slice's degradation reduced 43% (≥ 20%) with no material clean loss, and confirmed on constructed unaccented text |
| Uncertainty | Calibrated vs uncalibrated on independent data | ✅ including test, with a validation-fitted temperature |
| Efficiency | Reproducible quality/latency/memory comparison | ✅ minimum met; stretch (≥ 1.5× with neutral loss ≤ 0.02) not met: INT8 is 1.7× faster but loses 0.088 neutral F1 |

## § 11 Portfolio checklist

| Item | Status |
|---|---|
| Problem statement with 3–5 research questions | ✅ [RESEARCH_REPORT § 2](RESEARCH_REPORT.md): Q1–Q8, each with the study that answers it (Q3 deferred) |
| Strong sparse and neural baselines | ✅ |
| ≥ 3 controlled investigations incl. a negative/inconclusive one | ✅ segmentation (+), Tier E (negative, narrowed), cRT (inconclusive), boundary test (negative), topic stacking (negative) |
| Manually inspected error taxonomy of 60–100 cases | 👤 |
| One robustness evaluation beyond the aggregate | ✅ |
| Seed variation vs evaluation uncertainty kept apart | ✅ |
| One extension completed deeply | 🔶 Cycle 2 track A (encoder vs LLM): pipeline, prompt freeze, pilot; the declared runs need Kaggle and the API key 👤 |
| Executed notebooks, figures, artifact checksums | ✅ `02_results.ipynb` rebuilt and executed; release manifest has SHA-256 |
| Short technical report | ✅ [RESEARCH_REPORT.md](RESEARCH_REPORT.md) |

---

## What happens next

The register of every open item, with what closes it: [NEXT_PLAN.md § 1](NEXT_PLAN.md). Done in this
round: Cycle 2 declared and its evaluation data frozen; H5 and H6 decided; the served model switched
through the gate; the H7 pipeline, prompt freeze and pilot; mypy blocking; Docker end-to-end script;
provenance of Kaggle runs; publish bundle and model card; audit analysis tooling.

**Needs you (👤):**

- **Neutral audit**: annotate `results/studies/study_a/local/audit_sheet.csv` under
  [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md); a second pass over ≥ 50 rows; then
  `vifeedback study audit-report --second <file> --kind inter|intra`.
- **H7 declared runs**: Kaggle cell 4f (Qwen3-4B); set `OPENAI_API_KEY` for gpt-4o-mini; confirm
  whether UIT-VSFC may be sent to the API.
- **Publishing**: review `models/publish/<name>/README.md`, then `serve publish --repo-id … --upload`.
- **Natural data**: any lawfully usable sample of real feedback.
