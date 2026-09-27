# Next Plan — v2: close every open item, then Cycle 2

**v1** was written 2026-09-27 when Cycle 1 closed. **v2** (same day) was rewritten at the owner's
request. It starts from a register of everything still open: outstanding issues, experiments not
yet run, and results that rest on weaker evidence than their claim needs. Each item names what closes
it, and the steps below close them in order. Cycle 2 is declared in
[`configs/experiments/cycle2.yaml`](../configs/experiments/cycle2.yaml) (v2, ADR-025/026).

Status key: ✅ done · 🔄 running · ⏳ next, no owner input needed · 👤 needs the owner ·
⏸ deferred, with the reason.

**Progress, 2026-09-27 (same day).** Everything that needed no owner input is done: the register below
carries the evidence for each item. What remains is marked 👤. The official test was not touched.

---

## 1. Open-items register

### A. Outstanding issues (engineering, documentation)

| ID | Issue | What closes it | Who | Status |
|---|---|---|---|---|
| I1 | `mypy` is advisory in CI (35 errors in 13 files), so type errors can merge | Fix them; make `mypy` a blocking CI step | me | ✅ cde2174 |
| I2 | The Docker image has never served the real model. CI smoke-tests it without one | `make docker-e2e`: build, mount the released artifact, `/readyz` 200, golden `/v1/classify` cases | me | ✅ passes; it found `/version` reporting "unversioned", now fixed (f7b456c) |
| I3 | Status lines went stale: STATUS R8 ("XLM-R control running"), R10 ("benchmark pending"), P7 ("revisions not pinned"); REVIEW_COMPLIANCE ("H3 pending", an outdated "what happens next") | One consistency sweep, then the compliance audit re-run at Cycle 2 close | me | ✅ swept; compliance re-audited |
| I4 | Kaggle runs carry no git SHA: the code is uploaded as a zip. They do carry `source_sha256` | Map each recorded source hash to the commit whose `src/` produces it; record it | me | ✅ `results/provenance.json`: Kaggle runs → 5b2858c; 2 cRT runs from an uncommitted tree |
| I5 | The served model is not downloadable (R11: artifacts another researcher can fetch) | A publish command with a dry run, and a model card (data, per-class metrics, limits, manifest hash). The upload stays with the owner | me → 👤 | ✅ `serve publish` (dry run) · 👤 `--upload` |
| I6 | Local leftovers: blocked INT8 staging directories, `kaggle_results/` | Delete them. Keep `.previous-sentiment` (rollback) and the p9 checkpoint (H6) | me | ✅ 270 MB removed after a dry-run merge showed nothing unmerged |
| I7 | No explicit research-question list (review § 11, 🔶) | RQ1–RQ5 in RESEARCH_REPORT, each linked to its evidence | me | ✅ the report had Q1, Q2, Q4–Q6; Q3 (deferred), Q7, Q8 added |

### B. Experiments not yet run or not yet decided

| ID | Experiment | Status now | What closes it | Who | Status |
|---|---|---|---|---|---|
| H5 | Topic stacking, TF-IDF × PhoBERT (E08) | Declared; 5 OOF folds on the laptop | Declared rule applied; code, features and summary committed | me | ✅ not supported: −0.0102 [−0.0228, +0.0013] |
| H6 | Which model the service runs | Challenge set frozen | Rule applied on the challenge set; if it switches, release through the gate, then ADR plus EVALUATION_PROTOCOL § 4 | me | ✅ switched to the augmented model (ADR-027) |
| H7 | LLM reference on the hard cases (Study E) | Code, frozen prompt, pilot done | `evaluation/llm_reference.py`; pilot Qwen3-1.7B (laptop); declared Qwen3-4B (Kaggle cell); **gpt-4o-mini** (API, owner's key) | me, 👤 (Kaggle, key) | ✅ pilot · 👤 Kaggle cell 4f · 👤 `OPENAI_API_KEY` |
| A1 | Neutral audit (Step 1 of v1) | Sheet exported; decision tree frozen; analysis tool built | `study audit-report` (κ, per-stratum rates, Wilson intervals on the random stratum, applies the frozen tree), with tests. The owner annotates | me → 👤 | ✅ tooling · 👤 annotation |
| E04 | Label correction of train | Gated by A1's outcome | Runs only if A1 finds ≥ 30% incorrect gold | — | ⏸ gated |
| P5 | Diacritic-restoration front-end (alternative to H2) | Untested | Becomes the next robustness question **only if** H6 shows the augmented model failing on typed unaccented text | — | ⏸ not triggered: typed unaccented 0.74 |
| B | Distillation, careful INT8 (track B) | Designed in v1 | A later cycle. The review's rule is one specialization per cycle | — | ⏸ |
| R8′ | Per-model learning-rate tuning for XLM-R and PhoBERT-large; gradient balancing for multi-task | Proposed in RESEARCH_REPORT § 6 | Only with a named question (see "What not to do") | — | ⏸ |

### C. Results not yet confirmed

| ID | Claim | Why it is not confirmed | What confirms or refutes it | Status |
|---|---|---|---|---|
| C1 | H2 augmentation makes the model robust to missing diacritics | Shown only on *synthetic* noise (`strip_diacritics`, scripted teencode) | H6: `unaccented_typed` and `teencode_typed` rows, written by hand | ✅ confirmed for unaccented (0.32 → 0.74); **refuted for teencode** (0.975 → 0.875) |
| C2 | TF-IDF beats PhoBERT on `facility` (0.921 vs 0.905) | One model each on dev, no interval | H5 descriptive: paired bootstrap on the same validation rows, plus the 5-fold out-of-fold view | ✅ refuted: −0.021 [−0.060, +0.015] |
| C3 | The model handles negation | 44 templated pairs, simple *không* forms | `negation_pair` rows: *đâu có*, *chẳng … chút nào*, *chưa bao giờ*, negated negatives, both models | ✅ measured: both pair members right in 60% (CE) / 67% (augmented) of 15 pairs; harder negation is a real weakness |
| C4 | Served p95 latency meets the 30 ms target | Met in one session (23.9 ms), missed in the latest (33.6 ms) | Three sessions on AC power in the same Windows power mode; report the median and range. The claim stays the ratio | ✅ met: served p95 19.2 and 20.0 ms in the two reportable sessions (all six passes 18.7–20.7 ms; `latency/sessions/`). The 33.6 ms session had PyTorch 1.6× slower too: machine state |
| C5 | Out-of-scope input behaves sensibly | Never measured (RESEARCH_REPORT § 7) | Confidence on the 20 `out_of_scope` rows vs `objective_neutral`, descriptive | ✅ measured: it does not. Mean confidence 0.86–0.90 on off-topic text |
| C6 | "The neutral gap is not in the classifier" (H1) | Rules out the head, but cannot tell label ambiguity from representation | A1 (human audit) | 👤 |
| C7 | LLM results on UIT-VSFC are fair | Pretraining contamination unknown | The challenge set is the primary LLM comparison; validation is secondary | by design |

---

## 2. Where the evidence leaves us

| What Cycle 0–1 established | What it means now |
|---|---|
| Neutral errors are confident; boundary shifts, logit adjustment and cRT all move macro-F1 by noise (H1) | The gap is not in the classifier. Only the audit (A1) can split label ambiguity from representation |
| Augmentation repairs missing diacritics (0.27 → 0.64 on test), clean accuracy unchanged (H2) | A deployable gain, shown on synthetic noise. C1 tests real typing |
| XLM-R lost ~40% of its gap to preprocessing but still trails PhoBERT (H3) | New encoders only with a named question |
| Shared encoder: no gain at λ 0.3, sentiment cost at λ 1 (H4) | Multi-task is a serving-cost option, not a quality lever |
| INT8: 1.7× faster, −0.088 neutral F1 | The efficiency stretch target is unmet by the minority class (track B) |
| The test split has been evaluated 28 times | **New claims need new data**: the challenge set (frozen) |

---

## 3. Steps

### Step 0 — Engineering and documentation *(no GPU; runs while the GPU works)*

I1 → I3 → I4 → I6 → I7, then I2 (Docker needs the machine quiet). I5's publish command and model
card are written after H6, because H6 decides which model the card describes.

### Step 1 — Neutral audit *(owner: ~6–8 h annotation; tooling: me)*

Unchanged from v1 ([ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)): 160 rows annotated blind, a second
annotator on ≥ 50 rows (or intra-annotator after ≥ 24 h, labelled as such), per-stratum rates,
Wilson intervals from the random stratum only. The decision tree is frozen in `cycle2.yaml`
(`audit_decision_tree`). `study audit-report` applies it mechanically once the sheet is filled.

**An LLM does not annotate the audit.** The audit measures whether *human* gold labels are
defensible; an LLM label would be a model opinion. H7 reports LLM agreement with the adjudicated
labels afterwards, as the declared link between the two.

### Step 2 — Independent evaluation data

| Set | Status |
|---|---|
| **Challenge set v1**, 305 rows, 10 categories ([data/challenge/README.md](../data/challenge/README.md)) | ✅ frozen by SHA-256 in `cycle2.yaml` v2, before any evaluation. One author wrote and labelled it: a native-speaker review is welcome, and corrections go to v2, reported next to v1 |
| Natural sample, 500–1,000 real sentences with a clear licence | 👤 only if a lawful source exists; otherwise a stated limitation |

### Step 3 — Cycle 2, track A

| Order | Work | Where | Declared rule |
|---|---|---|---|
| 1 | **H5** stacking finishes, rule applied, committed | laptop GPU (running) | macro-F1 +0.005, CI > 0, facility and others not lower |
| 2 | **H6** CE vs augmented seed-42 on the challenge set | laptop, minutes | switch if typed-noise accuracy CI > 0 and other rows drop ≤ 0.02 |
| 3 | **H7 pipeline + pilot**: prompt variants (≤ 4) on a 200-row train subset with Qwen3-1.7B; freeze the best | laptop GPU | pilot reported, not declared |
| 4 | **H7 API arm**: gpt-4o-mini, frozen prompt, zero-shot on the challenge set | API, ~USD 0.05 | Holm over the two LLM comparisons |
| 5 | **H7 declared**: Qwen3-4B zero- and few-shot on validation + challenge set | Kaggle T4 (notebook cell) | neutral F1 vs encoder, paired CI |

**How the GPT-4o-mini key is used.** It is read from the `OPENAI_API_KEY` environment variable or a
git-ignored `.env` at the repository root. It is never printed, never written to a result file and
never committed. Only the constructed challenge set is sent to the API until the owner confirms that
UIT-VSFC's licence allows sending its text to a third party. After that confirmation, validation and
few-shot runs are added under the same frozen prompt. Every call's model snapshot, token counts and
cost are logged next to the predictions.

### Step 4 — Confirm what is unconfirmed

C1, C3 and C5 come out of H6's evaluation on the challenge set. C2 comes out of H5. C4 needs the
laptop idle, so it runs last: three benchmark sessions under a fixed power mode. C6 waits for A1.

### Step 5 — Close Cycle 2

- Apply every declared rule (`evaluation/decisions.py`) and record the outcomes in `cycle2.yaml`'s
  results file; log any test touch in EVALUATION_PROTOCOL § 4 (none is planned).
- Update RESEARCH_REPORT (RQ list, H5–H7), STATUS, README, the model card; re-execute
  `02_results.ipynb`; re-run the review compliance audit (I3).
- Commit run artifacts as each batch finishes (ADR-023); rebuild the release zip.

---

## 4. What not to do

- No more neutral-F1 sweeps on the old validation set: 73 neutral examples cannot resolve the effects
  Cycle 1 measured.
- No decision on the official test.
- No new encoder family, tuning sweep or loss variant without a named question.
- No LLM output as evaluation truth, and no API call on data whose licence has not been confirmed.

## 5. Decisions needed from the owner

| # | Decision | Default until decided |
|---|---|---|
| 1 | Set `OPENAI_API_KEY` in your own terminal or in `.env`. Do not paste it into chat | API arm waits |
| 2 | Does UIT-VSFC's licence allow sending its text to OpenAI? | Challenge set only, zero-shot |
| 3 | Who annotates the audit; is a second annotator available? | Intra-annotator after 24 h |
| 4 | Run the Kaggle cell for Qwen3-4B | Only the pilot and the API arm are reported |
| 5 | Upload the model to the HF Hub after reviewing the dry run | Not published |
| 6 | Any lawful natural data source? | Stated as a limitation |

## 6. Budget

| Work | GPU | Money |
|---|---|---|
| H5 (5 topic fine-tunes) | ~20 laptop GPU-min | — |
| H6 (inference, 2 × 305 rows) | < 1 min | — |
| H7 pilot (prompt development + challenge set) | ~30 laptop GPU-min | — |
| H7 declared (Qwen3-4B) | ~1.5 Kaggle T4-h | — |
| H7 API arm | — | ~USD 0.05 (challenge); ~USD 0.30 with validation |

Weight-updating runs: 5 of 30 (H5 only). Everything else is inference.
