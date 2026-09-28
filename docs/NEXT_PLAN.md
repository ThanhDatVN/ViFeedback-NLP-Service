# Next Plan — v3: Cycle 3, close the measured gaps of the served model

**v3, 2026-09-27.** v2 closed every item that needed no owner input (its register is kept as
[Appendix A](#appendix-a--v2-register-closed-2026-09-27)). v3 starts from what is still **not met**,
the ROADMAP targets and the gaps Cycle 2 measured, and says how each is handled. It also sets what
the OpenAI API is for. The rules are frozen in `configs/experiments/cycle3.yaml` (v1 before the first
Cycle 3 run; v2 external evaluation; v3 S2a, ADR-029). The challenge v2 hash goes into v4.

Status key: ✅ done · ⏳ next, no owner input needed · 👤 needs the owner · ⏸ deferred, with the reason.

## Progress, 2026-09-28

Every step that needed no owner input has run. Development results are not confirmations. The
human-typed challenge v2 needs time and people the owner does not have now, so the confirmation moved
to **NEU-ESC test** (real forum posts from another university, human labels; ADR-030, `cycle3.yaml`
v5), with rules committed before the cells were computed. Challenge v2 stays optional (Appendix A).

| Step | Result | Status |
|---|---|---|
| V1: are the Cycle 2 drops real? (5 seeds, all reproduce exactly) | Teencode: noise (3/5 seeds, p = 0.09). Short factual: not confirmed (3/5, p = 0.02). **Contrast sentences: confirmed drop** 0.875 → 0.800 (4/5, p = 0.004, Holm-robust). Unaccented gain holds in 5/5 | ✅ |
| Evaluation-data research | [EVALUATION_DATA.md](EVALUATION_DATA.md): CheckList matrix, metrics, power, NEU-ESC, ViLexNorm, EduPulse cross-check, v2 protocol | ✅ |
| Case invariance (declared rule) | Served model flipped 1.07% of labels on capitalized input → **the service now lowercases** (0 flips, 0 validation changes) | ✅ |
| ViLexNorm invariance (real typing) | 17% of labels flip between a real comment and its human normalization, for CE and augmented alike | ✅ measured |
| U1 gpt-4o-mini on challenge v1 | neutral F1 0.955 vs 0.713 (CE); confounded by v1's construction; same pattern as EduPulse | ✅ |
| U2 label check | 15 rows for owner review (`results/studies/challenge/label_review_v1.csv`) | 👤 review |
| H7 gpt-4o-mini on NEU-ESC (zero-shot, frozen prompt) | macro-F1 **0.604** vs CE 0.494 (+0.109 [+0.094, +0.124]) and augmented 0.462 (+0.142) at seed 42; neutral F1 0.769 vs 0.526 / 0.474; over-calls negative (precision 0.43, recall 0.80). USD 0.21 for 6,613 posts | ✅ measured; reference only |
| **S2a** real-typing lexicon (ViLexNorm train) | Development gate **not passed**: ViLexNorm flips 0.169 → 0.177 (p = 0.13); validation +0.003 | ✅ negative |
| **S2b** diacritic restoration (train-only) | Stripped validation macro-F1 **0.686 → 0.857**; clean predictions unchanged; challenge unaccented 0.74 → 0.86; 0.2 ms. NEU-ESC confirmation: unaccented posts 0.270 → **0.374** (+0.104 [+0.091, +0.118]), posts as written −0.0006 | ✅ confirmed · **served** (ADR-031) |
| **S2b′** CE + restoration vs augmented + restoration (5 seeds, v4) | Development: unaccented 0.904 vs 0.880, contrast **0.875 vs 0.800** (p = 0.004). NEU-ESC confirmation: contrast 0.440 vs 0.437 (**p = 0.68**); the other checks favoured CE (overall +0.028, unaccented +0.036) | ✅ **not passed**: the augmented model stays |
| **S3** out-of-scope score | Development AUROC **0.949** (max-probability 0.862). NEU-ESC confirmation: 563 off-topic posts, AUROC **0.977** (max-probability 0.920), 86% flagged at 5% of validation | ✅ confirmed · **served** as `in_scope` (ADR-031) |
| **S5** careful INT8 | 178.5 MB, neutral agreement 0.959; macro-F1 drop +0.0004 but upper bound 0.0095 > 0.005: **not passed**. Latency indicative only (no steady pass): p50 6.4 ms vs 11.1 ms for FP32 | ✅ negative → Cycle 4 |
| NEU-ESC (6,613 real forum posts, 5 seeds per recipe) | macro-F1 CE **0.463**, augmented 0.434, S2a 0.433: all low (the model calls 30% of posts neutral, gold 69%); augmentation lower on real student text (s42: −0.033 [−0.042, −0.024]; 4/5 seeds) | ✅ measured |
| Serving release (ADR-031) | Re-exported with a `features` output (logit parity 1.7e-05, 100% label agreement); restorer and out-of-scope score attached, each SHA-256-checked by the service | ✅ |
| Neutral audit · Kaggle Qwen3-4B · HF upload · U2 label review | — | 👤 |
| Challenge v2 (human-typed) | Replaced as the confirmation set by NEU-ESC (ADR-030); still the only place to test typed teencode and the S2b′ question on this corpus's register | ⏸ optional |

---

## 1. What is not met, and where it is handled

| Target or gap | Now | Why | Handled in |
|---|---|---|---|
| **S1** sentiment test macro-F1 ≥ 0.84 (min 0.80 ✅) | 0.837 best (VnCoreNLP), 0.830 served | 0.003 short, inside seed spread (std 0.003–0.011). A sweep on the old validation set cannot resolve it | Not chased (§ 6). May move if Step 4 finds a representation fix |
| **S2** topic test macro-F1 ≥ 0.83 (min 0.79 ✅) | 0.804; `others` F1 0.55 | Stacking (H5) and multi-task (H4) gave nothing; `others` is a residual class | Step 7: error review only |
| **S4** neutral F1 ≥ 0.65 (min 0.55 ✅), and the +0.03 stretch | 0.576 served (test, 5 seeds) | Not the classifier (H1). Labels vs representation is undecided | Step 4 (audit → one branch) 👤 |
| **S7** served artifact ≤ 200 MB (target 120) | **540 MB** (FP32) | INT8 (136 MB) cost 0.088 neutral F1 and was blocked | Step 5 |
| Efficiency stretch: ≥ 1.5× over L3 with neutral loss ≤ 0.02 | INT8 1.7×, loss 0.088 | Same as S7 | Step 5 |
| **S8** Docker image ≤ 700 MB (min 1.2 GB ✅) | 1.02 GB (Python env 648 MB) | pyvi pulls scikit-learn and SciPy; transformers for one tokenizer | Step 6 |
| **S9** ≥ 30 coded error cases | 0 | The audit's 160 rows are not annotated | Step 4 👤 |
| **S10** clean-clone eval path < 15 min on CPU | Not timed | — | Step 6 |
| Teencode, challenge v1: 0.975 → 0.875 after augmentation | p = 0.13, one seed (ADR-028) | Possibly noise. If real: the augmentation's teencode map has **14 entries** | Step 0, then Step 2a |
| Short factual sentences: 0.767 → 0.633 | p = 0.22, one seed | Possibly noise | Step 0, then Step 2c |
| Unaccented text: 0.74 accuracy (the same model scores 0.95 on clean validation, other sentences) | Significant gain (0.32 → 0.74), still a gap | pyvi mis-segments unaccented text and the model sees rare forms | Step 2b |
| Off-topic input gets a confident label (0.86–0.90) | No signal to abstain | Max-probability does not separate it | Step 3 |
| Negation minimal pairs: both right in 10 of 15 | n = 15, too few to act on | — | Step 1 (v2 adds pairs); measured, not fixed |
| **No real user typing, no natural sample;** challenge v1 is constructed and single-labeller | — | — | Step 1 👤 |
| H7 declared runs (Qwen3-4B, gpt-4o-mini) | gpt-4o-mini done (challenge v1, NEU-ESC); Qwen3-4B pending | Needs Kaggle | Step 1 👤 |

---

## 2. What the OpenAI API is for

**Principle.** The API produces *measurements* and *candidate material a person checks*. It never
produces evaluation truth, never labels UIT-VSFC, and is not in the serving path.

| Use | What it does | Data sent | When | Cost |
|---|---|---|---|---|
| **U1** H7 reference (declared) | gpt-4o-mini zero-shot, label likelihood, frozen prompt, on challenge v1: per-class F1, latency, USD per 1k | challenge v1 (the project's own text) | as soon as the key is set | ≈ USD 0.05 |
| **U2** Label check of challenge v1 | Rows where gpt-4o-mini **and** Qwen both disagree with the gold label go on a review list. The owner decides; any correction goes to v1.1, reported next to v1 | none extra (U1's outputs) | with U1 | 0 |
| **U3** Teencode lexicon candidates | For the ~300 most frequent words, ask for common abbreviated and unaccented spellings. The owner accepts or rejects each candidate | a list of common words, not sentences | Step 2a | < USD 0.05 |
| **U4** Off-topic development set | ~500 generated off-topic sentences (weather, shopping, sport, spam, administrative questions) to choose the out-of-scope method and threshold. Never used for the final check | nothing from the corpus | Step 3 | ≈ USD 0.05 |
| **U5** Router measurement (conditional) | Only if the declared H7 runs show an LLM ahead on an identifiable slice: encoder first, LLM on that slice; quality, p95 latency, USD per 1k | challenge sets; validation only after the licence question | Step 4b | < USD 1 |

**Not for:** annotating the audit or train data; defining any gold label; classifying real student
feedback in production. Sending student text to a third party needs consent, so a router stays a
measurement unless it runs on a local model.

**Key handling** is unchanged: `OPENAI_API_KEY` in the owner's terminal or a git-ignored `.env`,
never in chat. The snapshot (`gpt-4o-mini-2024-07-18`) is recorded with every result, because a
retired snapshot cannot be re-run.

---

## 3. Cycle 3 — one specialization: reliability of the served model on real input

Cycle 2 moved the service to the augmented model and measured four input-level gaps: teencode beyond
the map, unaccented text still at 0.74, off-topic input unflagged, and a possible drift on short
factual sentences. They share one evaluation need, real typed input, and one deliverable: a service
that is robust to how people type and says when an input is not course feedback. Neutral waits on the
audit (Step 4, owner). Size (Step 5) is a CPU side track.

### Step 0 — Verify before fixing *(no owner input; ≈ 50 GPU-min)*

| Item | Method | Rule, fixed before running |
|---|---|---|
| **V1** Are the two drops real? | Retrain seeds 1337, 2024, 7, 31337 for CE and augmented, identical configurations, saving checkpoints. Each must reproduce its registry validation macro-F1 exactly (training is deterministic). Evaluate all ten checkpoints on challenge v1 | A drop is **confirmed** if augmented < CE on that category in ≥ 4 of 5 seeds **and** the pooled exact McNemar p < 0.05. Otherwise it is recorded as noise and leaves the model card's limits |
| **V2** Correct the Cycle 2 wording | ADR-028: constructed, not typed by users; the drops are hypotheses | ✅ done |

V1 also replaces H6's one-seed table with five-seed challenge numbers for every category.

### Step 1 — New evidence *(owner)*

| Item | What | Effort |
|---|---|---|
| **Challenge v2, human-typed** | About 200 sentences **typed on a phone as people really type**, by the owner and ideally 2–3 other people, without editing: 60 unaccented, 60 teencode or abbreviated, 30 short factual, 25 more negation pairs (50 rows), 30 off-topic. Labelled under the guide; a second person labels ≥ 50 rows for κ. Frozen by SHA-256 before any Cycle 3 model is trained. **v2 confirms Cycle 3; v1 becomes the development set** | 3–4 h |
| **H7 declared runs** | Kaggle cell 4f (Qwen3-4B), then U1 and U2 | 30 min + key |
| **Audit annotation** | Step 4a | 6–8 h |

### Step 2 — Robustness v2 *(≈ 10 weight-updating runs)*

**2a. Teencode lexicon v2** (only if V1 confirms the teencode drop, or if v2 shows teencode below
clean accuracy by > 0.05).
The map grows from 14 to 150–300 verified entries: U3 candidates the owner accepted, plus pairs from a
published Vietnamese lexical-normalization corpus if its licence allows. No challenge row (v1 or v2)
is used to build it. Same recipe (p = 0.3), 5 seeds; control = the served augmented recipe at the same
seeds.
*Rule:* teencode accuracy on v2 up with a paired CI above 0, pooled over seeds; unaccented not lower
by more than 0.02; validation macro-F1 non-inferior at 0.005; **no v2 category lower by more than
0.05** (ADR-027's lesson: per category, not only pooled).

**2b. Diacritic restoration front-end** (CPU; no classifier retraining).
A restorer trained on UIT-VSFC **train** text only, which is 99.86% diacritized: candidate accented
forms per syllable, a word-bigram model, Viterbi decoding. It runs only when an input's share of
accented characters is below a threshold set on train, so normal input is untouched.
*Rule:* unaccented accuracy on v2 up with a paired CI above 0 over the served model alone;
predictions on validation identical (restorer skipped); added p95 latency ≤ 5 ms.

**2c. Short factual drift** (only if V1 confirms it). One declared variant, chosen from V1's error
pattern; same per-category rule as 2a.

### Step 3 — Say when an input is not course feedback *(CPU, no training)*

Candidates: Mahalanobis distance on the encoder's sentence features (class means and a shared
covariance fitted on train), the energy score, and max-probability as the baseline. Method and
threshold are chosen on development data (v1's 20 off-topic rows + U4), with the threshold set so 95%
of validation is kept.
*Rule:* on v2, off-topic vs in-domain AUROC ≥ 0.90 and ≤ 5% of validation flagged. If met, the API
returns an `in_scope` score next to the label. It does not abstain on its own: the caller decides.

### Step 4 — Neutral: the audit decides *(owner-gated)*

**4a.** The owner annotates the 160-row sheet; `vifeedback study audit-report` applies the tree frozen
in `cycle2.yaml` and selects exactly one branch:

| Audit finding (confident neutral-error strata) | Next experiment | Runs |
|---|---|---|
| ≥ 30% incorrect gold | E04: label correction of train, 2 × 2 (labels × recipe) | 12 |
| ≥ 40% ambiguous | One soft-label run; report the ceiling; stop optimizing neutral on this benchmark | 3 |
| Otherwise | One representation experiment: ViSoBERT or task-adaptive pretraining on train text | 3–6 |

This is the only route to S4's target and the +0.03 stretch. It also closes S9.

**4b. Router** (conditional): only if the declared H7 runs show an LLM ahead of the encoder on an
identifiable slice (the pilot pointed at short factual sentences), with a paired CI above 0. Measured
with U5.

### Step 5 — Size: S7 and the efficiency stretch *(CPU, no training)*

Careful INT8: per-channel dynamic quantization, excluding the classifier head, with the last k layers
kept FP32 as variants. At most four recipes, chosen on a held-out stratified **train** subset, never
on validation. One acceptance on validation with the existing gate, plus neutral F1 loss ≤ 0.02.
*Rule:* ship if the gate passes, the artifact is ≤ 200 MB and p50 is ≥ 1.5× faster than L3. If no
recipe passes: distillation into a 6-layer student becomes Cycle 4 (track B).

### Step 6 — Engineering

- **Image (S8):** measure package sizes in the image's environment; trim what serving does not use
  (pyvi's scikit-learn/SciPy chain is the first candidate). The image smoke test and `make docker-e2e`
  must still pass.
- **S10:** time the clean-clone evaluation path on CPU in CI and record it.
- **Model card:** rebuilt after any serving change; the upload stays with the owner.

### Step 7 — Topic `others` *(descriptive, 1–2 h)*

An error review of `others` on validation: what it is confused with, and whether the class is a label
policy problem. No training. It decides whether topic deserves a later cycle; S2 is not chased now.

### Step 8 — Close Cycle 3

Apply the frozen rules; release only through the gate; confirm on v2; update STATUS, the report and the
model card; re-run the review compliance audit. No Cycle 3 decision uses the official test.

---

## 4. Order

| Week | Without the owner | The owner |
|---|---|---|
| 1 | Step 0 (V1); U1 and U2 once the key is set; Step 3 development; Step 5 recipes; Step 6 | Set the key; Kaggle cell 4f; type challenge v2; start the audit |
| 2 | Freeze v2, declare `cycle3.yaml`; Steps 2a–2c runs; Step 3 confirmation | Check the lexicon candidates (≈ 1 h); finish the audit |
| 3 | Apply the rules, release if passed; the Step 4 branch; documents | Review the model card; upload |

## 5. Budget

| Resource | Estimate |
|---|---|
| Weight-updating runs | ≤ 30: V1 8, Step 2 up to 10, Step 4 up to 12 |
| Laptop GPU | ≈ 3 h |
| Kaggle | ≈ 0.5 h (cell 4f) |
| API | < USD 1.5 in total |

## 6. What not to do

- No sweep for S1 or S2 on the old validation set: S1 is inside seed noise, S2's two interventions failed.
- No fix for a drop before Step 0 shows it is real.
- No challenge row, v1 or v2, used to build a lexicon, a threshold or a prompt.
- No LLM output as a gold label; no API call in the serving path.
- No decision on the official test.

## 7. Decisions needed from the owner

| # | Decision | Default until decided |
|---|---|---|
| 1 | Set `OPENAI_API_KEY` (terminal or `.env`) | U1–U5 wait |
| 2 | May UIT-VSFC text be sent to the API? | Challenge sets only |
| 3 | Who types challenge v2, and how many people | Cycle 3 cannot confirm without it |
| 4 | Audit annotation; a second annotator? | Step 4 waits; intra-annotator after 24 h |
| 5 | Kaggle cell 4f | H7 stays at pilot |
| 6 | Upload the model | Not published |

---

## Appendix A — v2 register (closed 2026-09-27)

Everything v2 could close without the owner, with its evidence. Items marked 👤 carry over into v3.

### A1. Outstanding issues (engineering, documentation)

| ID | Issue | What closes it | Who | Status |
|---|---|---|---|---|
| I1 | `mypy` is advisory in CI (35 errors in 13 files), so type errors can merge | Fix them; make `mypy` a blocking CI step | me | ✅ cde2174 |
| I2 | The Docker image has never served the real model. CI smoke-tests it without one | `make docker-e2e`: build, mount the released artifact, `/readyz` 200, golden `/v1/classify` cases | me | ✅ passes; it found `/version` reporting "unversioned", now fixed (f7b456c) |
| I3 | Status lines went stale: STATUS R8 ("XLM-R control running"), R10 ("benchmark pending"), P7 ("revisions not pinned"); REVIEW_COMPLIANCE ("H3 pending", an outdated "what happens next") | One consistency sweep, then the compliance audit re-run at Cycle 2 close | me | ✅ swept; compliance re-audited |
| I4 | Kaggle runs carry no git SHA: the code is uploaded as a zip. They do carry `source_sha256` | Map each recorded source hash to the commit whose `src/` produces it; record it | me | ✅ `results/provenance.json`: Kaggle runs → 5b2858c; 2 cRT runs from an uncommitted tree |
| I5 | The served model is not downloadable (R11: artifacts another researcher can fetch) | A publish command with a dry run, and a model card (data, per-class metrics, limits, manifest hash). The upload stays with the owner | me → 👤 | ✅ `serve publish` (dry run) · 👤 `--upload` |
| I6 | Local leftovers: blocked INT8 staging directories, `kaggle_results/` | Delete them. Keep `.previous-sentiment` (rollback) and the p9 checkpoint (H6) | me | ✅ 270 MB removed after a dry-run merge showed nothing unmerged |
| I7 | No explicit research-question list (review § 11, 🔶) | RQ1–RQ5 in RESEARCH_REPORT, each linked to its evidence | me | ✅ the report had Q1, Q2, Q4–Q6; Q3 (deferred), Q7, Q8 added |

### A2. Experiments not yet run or not yet decided

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

### A3. Results not yet confirmed

| ID | Claim | Why it is not confirmed | What confirms or refutes it | Status |
|---|---|---|---|---|
| C1 | H2 augmentation makes the model robust to missing diacritics | Shown only on *synthetic* noise (`strip_diacritics`, scripted teencode) | H6: `unaccented_typed` and `teencode_typed` rows, constructed (not typed by users) | ✅ confirmed for unaccented (0.32 → 0.74, p < 0.001); teencode not demonstrated (0.975 → 0.875, p = 0.13 at one seed; ADR-028) |
| C2 | TF-IDF beats PhoBERT on `facility` (0.921 vs 0.905) | One model each on dev, no interval | H5 descriptive: paired bootstrap on the same validation rows, plus the 5-fold out-of-fold view | ✅ refuted: −0.021 [−0.060, +0.015] |
| C3 | The model handles negation | 44 templated pairs, simple *không* forms | `negation_pair` rows: *đâu có*, *chẳng … chút nào*, *chưa bao giờ*, negated negatives, both models | ✅ measured: both pair members right in 60% (CE) / 67% (augmented) of 15 pairs, against 36/36 on the simple probe; n is small |
| C4 | Served p95 latency meets the 30 ms target | Met in one session (23.9 ms), missed in the latest (33.6 ms) | Three sessions on AC power in the same Windows power mode; report the median and range. The claim stays the ratio | ✅ met: served p95 19.2 and 20.0 ms in the two reportable sessions (all six passes 18.7–20.7 ms; `latency/sessions/`). The 33.6 ms session had PyTorch 1.6× slower too: machine state |
| C5 | Out-of-scope input behaves sensibly | Never measured (RESEARCH_REPORT § 7) | Confidence on the 20 `out_of_scope` rows vs `objective_neutral`, descriptive | ✅ measured: it does not. Mean confidence 0.86–0.90 on off-topic text |
| C6 | "The neutral gap is not in the classifier" (H1) | Rules out the head, but cannot tell label ambiguity from representation | A1 (human audit) | 👤 |
| C7 | LLM results on UIT-VSFC are fair | Pretraining contamination unknown | The challenge set is the primary LLM comparison; validation is secondary | by design |
