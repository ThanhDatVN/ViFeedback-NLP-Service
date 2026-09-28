# Next Plan — v4: what is still open after Cycle 3, and Cycle 4

**v4, 2026-09-28.** Cycle 3 closed every step that needed no owner input and confirmed its serving
changes on real student text (NEU-ESC, ADR-030/031); its outcome is kept as
[Appendix A](#appendix-a--cycle-3-outcome-v3-closed-2026-09-28), and the v2 register as
[Appendix B](#appendix-b--v2-register-closed-2026-09-27). v4 starts from a review of what is still
open: targets not met, holes found in the current system, experiments not run, results not confirmed.
It then proposes one specialization for Cycle 4 and says what is ready for the owner to run on Kaggle.

Status key: ✅ done · ⏳ next, no owner input needed · 👤 needs the owner · ⏸ deferred, with the reason.

---

## 1. Register of what is open

### 1a. Targets not met

| Target | Now | Why | Handled in |
|---|---|---|---|
| **S1** sentiment test macro-F1 ≥ 0.84 (min 0.80 ✅) | 0.830 served, 0.837 best | 0.003–0.010 short, inside seed spread; the test split is spent (28 uses) | Not chased (§ 6) |
| **S2** topic test macro-F1 ≥ 0.83 (min 0.79 ✅) | 0.804; `others` recall 0.53 | `others` is a residual class; stacking (H5) and multi-task (H4) gave nothing | Not chased; label-policy note only |
| **S4** neutral F1 ≥ 0.65 (min 0.55 ✅) | 0.576 served (UIT test, 5 seeds); 0.474 on NEU-ESC | Label ambiguity vs representation is undecided | Step 5, gated on the audit 👤 |
| **S7** served artifact ≤ 200 MB (target 120) | 540 MB FP32 | Careful INT8 (178.5 MB) could not show non-inferiority: 73 neutral validation examples give an upper bound of 0.0095 against a 0.005 margin | Step 4 |
| Efficiency stretch: ≥ 1.5× over FP32, neutral loss ≤ 0.02 | INT8 about 1.7× (indicative) | Same as S7 | Step 4 |
| **S8** Docker image ≤ 700 MB (min 1.2 GB ✅) | 1.02 GB | pyvi pulls scikit-learn and SciPy; transformers is kept for one tokenizer | Step 6 |
| **S9** ≥ 30 coded error cases | 0 | The audit's 160 rows are not annotated | Step 5 👤 |
| **S10** clean-clone evaluation path < 15 min on CPU | Not timed | — | Step 6 |

### 1b. Holes found in this review

| ID | Hole | Why it matters | What closes it |
|---|---|---|---|
| **G1** | `serve bench` times segmentation + ONNX only. Since ADR-031 the service also lowercases, restores diacritics and computes the out-of-scope score (a second graph output and a Mahalanobis distance) | The latency claim (C4, p95 19–20 ms ≤ 30 ms) predates the pipeline that is served now; S2b's "added p95 ≤ 5 ms" was measured in development only | Step 0 |
| **G2** | The `in_scope` threshold keeps 95% of UIT-VSFC validation; how often it flags **in-scope** student text from another institution was never measured | A partner university's real feedback could be flagged as off-topic | Step 0 (descriptive, by NEU-ESC topic) |
| **G3** | NEU-ESC test has now served the external evaluation, three v5 rules and H7 | Adaptive reuse of one test set | Cycle 4 selects on NEU-ESC **validation** only; each declared rule uses test once, and every use is logged |
| **G4** | The H7 rule (Cycle 2) was never applied | H7 has been open since Cycle 2 | `study h7-decide` (added 2026-09-28; `cycle3.yaml` v6) applies it once Qwen3-4B has run 👤 |
| **G5** | `kaggle_train.ipynb` clones into `/kaggle/working/repo`, which Kaggle saves with each version, data and checkpoints included | Corpus files end up in the notebook's output | Step 6, before its next use (the H7 notebook is fixed: it clones into `/tmp`) |
| **G6** | NEU-ESC's label policy is not UIT-VSFC's: annotators call 69% of posts neutral ("no emotion"); Toxic is merged into negative | Every cross-domain score mixes domain shift with policy shift, and training on NEU-ESC changes what `neutral` means for the service | Step 1 (declared data roles) and owner decision 3 |
| **G7** | Training on all of NEU-ESC train would teach the model that spam, news, job and club posts are in-domain | The out-of-scope score (ADR-031) would degrade silently | Cycle 4 trains only on in-scope topics (§ 3) and re-checks the score |
| **G8** | Real-typing instability (ViLexNorm: 17% of labels flip) is untouched after S2a failed | The largest robustness gap left | Not in Cycle 4 (one specialization); candidate for Cycle 5: a normalization model or consistency training on ViLexNorm pairs |
| **G9** | The augmentation's contrast-sentence drop is confirmed on constructed v1 but did not replicate on NEU-ESC | Which one describes UIT-register text is unknown | Only a human-typed challenge v2 settles it (optional 👤) |
| **G10** | The rollback copy is now the augmented model without the features output; the CE copy is gone | A rollback drops the out-of-scope score; CE needs a re-export | Documented; `make export CKPT=<ce checkpoint>` on demand |

### 1c. Experiments not run

| ID | Experiment | Blocked by | Status |
|---|---|---|---|
| **H7-4B** | Qwen3-4B, the six declared configurations, plus NEU-ESC zero-shot (`cycle3.yaml` v6) | A Kaggle run | 👤 notebook ready and verified (§ 8) |
| **A1** | Neutral audit (160 rows) | Owner annotation | 👤 tooling ready (`study audit-report`) |
| **E04** / soft labels / representation | The audit's branch | A1 | ⏸ gated |
| **U2** | 15-row label review of challenge v1 | Owner | 👤 `results/studies/challenge/label_review_v1.csv` |
| **U5** | Router measurement | H7-4B on NEU-ESC | Step 3, conditional |
| Challenge v2 | Human-typed challenge set | Owner time and people | ⏸ optional (ADR-030) |
| Track B | Distillation into a 6-layer student | Step 4's outcome | ⏸ |
| R8′ | Per-model learning rates, repeated CV for finalist selection | A named question | ⏸ |
| U3 | Teencode candidates from the API | Superseded by S2a (ViLexNorm) | Dropped |

### 1d. Results not confirmed

| Claim | Why it is not confirmed | What confirms or refutes it |
|---|---|---|
| gpt-4o-mini is better on neutral (challenge v1: +0.24 neutral F1 over CE) | v1 is written in the prompt's own convention (ADR-028) | `h7-decide` applies the declared rule; the stronger evidence is NEU-ESC (+0.24 neutral F1 over CE on real posts) |
| The augmentation hurts contrast sentences | Exploratory on v1; not replicated on NEU-ESC | Challenge v2 (optional) |
| INT8 is non-inferior | Power, not quality: the drop is +0.0004 | Step 4 |
| Served p95 ≤ 30 ms | Measured before ADR-031 | Step 0 (G1) |
| NEU-ESC comparisons with the LLM | Seed 42 encoders only (the served seed); 5-seed means reported beside them | Accepted as declared; Cycle 4 uses 5 seeds |

---

## 2. What the OpenAI API is for

**Principle** (unchanged): the API produces measurements and material a person checks. It never
produces evaluation truth, never labels UIT-VSFC, and is not in the serving path. UIT-VSFC text is
not sent. NEU-ESC text may be (owner, 2026-09-28).

| Use | Status |
|---|---|
| **U1** gpt-4o-mini reference on challenge v1 and NEU-ESC | ✅ USD 0.21 in total |
| **U2** Label check of challenge v1 | 👤 the review list is ready |
| **U3** Teencode candidates | Dropped (S2a used ViLexNorm) |
| **U4** Off-topic development set | ✅ used for S3 development |
| **U5** Router measurement | Only with a local model (Step 3): an API router would send student text to a third party |

Cycle 4 needs no API spending. Key handling is unchanged: `OPENAI_API_KEY` in the git-ignored `.env`.

---

## 3. Cycle 4 — one specialization: student text from other institutions

**Why this one.** It is the largest measured gap that needs no owner input to work on. On 6,613 real
posts from another university the served model scores macro-F1 0.462 (neutral F1 0.474), while a
zero-shot gpt-4o-mini reaches 0.604 (0.769). The gap is mostly the neutral boundary under a domain
shift, and NEU-ESC comes with human-labelled train (23,048) and validation (3,305) splits. A service
used outside UIT meets this first. Neutral on UIT-VSFC stays gated on the audit (Step 5); size stays a
CPU side track (Step 4).

**Question (Q9).** How much of the cross-institution gap does in-domain training data close, at what
cost to UIT-VSFC, and does a local LLM add anything beyond it?

### Step 0 — Close the review holes *(CPU, no training, before `cycle4.yaml`)*

| Item | Method | Rule, fixed before running |
|---|---|---|
| **G1** latency of the served pipeline | `serve bench` times the service's own path: lowercase → restorer → pyvi → ONNX logits and features → Mahalanobis. Clean test sentences and the same with diacritics stripped (the restorer's worst case); three sessions under the C4 protocol | Unchanged: p95 ≤ 30 ms; ADR-031's additions ≤ 5 ms at p95 |
| **G2** `in_scope` on other institutions' in-scope text | Flag rate per NEU-ESC topic on validation (Academic, Service, Other … vs the four off-topic topics) | Descriptive. If Academic or Service posts are flagged at more than twice the validation rate (10%), the limitation goes into the model card and the API docs |
| **G6** label policy | The served model's confusion on NEU-ESC validation by gold label and topic; how many gold-neutral posts are evaluative under the UIT guide is left to the owner (decision 3) | Descriptive |

### Step 1 — Declare `cycle4.yaml` *(before any Cycle 4 training run)*

- **Data roles.**
  - NEU-ESC train is for training only, restricted to in-scope topics (G7): 21,113 posts, with Spam, News, Jobs & Recruitment, and Club & Events excluded.
  - NEU-ESC validation is for every selection.
  - NEU-ESC test is used once per rule, and each use is logged (G3).
  - UIT-VSFC validation is used for non-inferiority.
  - UIT-VSFC test is not used.
- **Label mapping** as in v5 (Toxic → negative). Any change of policy is owner decision 3, declared
  before training.
- **Licence.** NEU-ESC is CC BY 4.0 (paper) and Apache-2.0 (card), gated. A model trained on it is
  not published until the owner confirms the gated conditions allow it (decision 2).

### Step 2 — H8: in-domain training data *(≤ 7 weight-updating runs, laptop or Kaggle)*

- **Recipes.** At most two, each chosen with one seed on NEU-ESC validation:
  - **(a) mixed:** UIT-VSFC train plus in-scope NEU-ESC train, under the served recipe.
  - **(b) sequential:** the served checkpoint, fine-tuned on in-scope NEU-ESC train at a lower learning rate.
- **Seeds and control.** The chosen recipe runs at 5 seeds. The control is the served recipe at the same seeds; those checkpoints already exist from V1.
- **Rule (draft; frozen in `cycle4.yaml`).** All of these must hold:
  - NEU-ESC test macro-F1 is higher, with a paired 95% CI above 0, pooled over seeds.
  - On UIT-VSFC validation (5-seed mean), macro-F1 is not lower by more than 0.005 and neutral F1 not lower by more than 0.02.
  - Stripped UIT-VSFC validation with the restorer is not lower by more than 0.01.
  - The out-of-scope AUROC on NEU-ESC off-topic posts stays ≥ 0.90.
- **If it passes.** Release through the gate with an ADR; the restorer and the out-of-scope score are refitted and re-accepted.

### Step 3 — H9: a local router *(conditional; measurement only)*

Only if Qwen3-4B on NEU-ESC test (H7-4B) is ahead of the served model with a paired CI above 0.
- **Design.** The encoder answers first; the LLM answers when the encoder's confidence is below τ. τ is set on NEU-ESC validation for a fixed routed share, which needs a second Kaggle run on validation.
- **Report.** Macro-F1, the share routed, GPU-seconds per 1k, and the gain over H8's model.
- **Why only a measurement.** The CPU service does not run a 4B model. Deployment is the owner's call.

### Step 4 — Size: S7 *(CPU, after Step 2 decides the served model)*

- **Power check before any rule.** Take the INT8 disagreement pattern observed on the held-out train subset: 1.35% of labels overall, 4.1% of FP32-neutral predictions. Estimate the one-sided upper bound reachable on a larger labelled acceptance set (UIT-VSFC validation + NEU-ESC validation, 4,888 posts).
- **If the 0.005 margin becomes demonstrable.** Declare **S5′**: same recipe, same margin, larger set, with an ADR.
- **Otherwise.** Distillation (track B) becomes Cycle 5.

### Step 5 — Neutral on UIT-VSFC *(owner-gated, unchanged)*

The owner annotates the 160-row sheet. `study audit-report` applies the tree frozen in `cycle2.yaml`
and selects one branch: E04 label correction, a soft-label ceiling, or one representation experiment.
This is the only route to S4 and S9.

### Step 6 — Engineering

- **S8.** Measure package sizes inside the image, then trim what serving does not use. pyvi's scikit-learn/SciPy chain is the first candidate. `docker-e2e` must still pass.
- **S10.** Time the clean-clone evaluation path on CPU in CI and record the result.
- **G5.** Move `kaggle_train.ipynb`'s clone to `/tmp`, with the same guard test as the H7 notebook.
- **Model card.** Rebuild it after any serving change. The upload stays with the owner.

---

## 4. Order

| Week | Without the owner | The owner |
|---|---|---|
| 1 | Step 0 (G1, G2, G6); `cycle4.yaml`; H8 recipe choice on NEU-ESC validation | Kaggle H7 notebook (§ 8); decisions 2 and 3; start the audit |
| 2 | H8 at 5 seeds and its confirmation; `study h7-decide`; Step 3 if triggered | U2 review; the audit; a second Kaggle run if Step 3 is triggered |
| 3 | Step 4 power check → S5′ or track B; release if H8 passes; documents; the audit branch | Review the model card; upload |

## 5. Budget

| Resource | Estimate |
|---|---|
| Weight-updating runs | ≤ 19: H8 up to 7, the audit branch up to 12 |
| Laptop GPU | ≈ 3 h, one job at a time with cool-downs |
| Kaggle | ≈ 1 h for the H7 notebook, ≈ 0.5 h more if Step 3 runs |
| API | none planned |

## 6. What not to do

- No selection on NEU-ESC test; no NEU-ESC off-topic post in training data.
- No sweep for S1 or S2 on the old validation set.
- No UIT-VSFC text to the API; no LLM output as a gold label; no LLM in the CPU service.
- No decision on the official test.
- No second specialization in Cycle 4: real typing (G8) and size (beyond Step 4) wait.

## 7. Decisions needed from the owner

| # | Decision | Default until decided |
|---|---|---|
| 1 | Run the Kaggle H7 notebook (§ 8) | H7 stays undecided |
| 2 | Do NEU-ESC's gated conditions allow publishing a model trained on it? | H8 runs; a model trained on NEU-ESC is not published |
| 3 | Label policy for other institutions' text: keep UIT-VSFC's (requests are negative, neutral is rare) or adopt NEU-ESC's (neutral = no emotion) | Keep UIT-VSFC's; report both |
| 4 | Neutral audit; a second annotator? | Step 5 waits; intra-annotator after 24 h |
| 5 | U2 label review (15 rows) | Challenge v1 stays as frozen |
| 6 | Upload the model (`serve publish --upload`) | Not published |
| 7 | Challenge v2 | Not written; NEU-ESC confirms instead (ADR-030) |

---

## 8. Kaggle: what is ready to run

**Notebook:** [`notebooks/kaggle_h7_llm.ipynb`](../notebooks/kaggle_h7_llm.ipynb). It runs H7's six
declared configurations and Qwen3-4B zero-shot on NEU-ESC test (`cycle3.yaml` v6). The encoder
predictions it compares against, the frozen prompt and the challenge set are committed files, so it
needs no checkpoint and no dataset upload.

**Verified on 2026-09-28 at commit `faf15da`.** The notebook ran end to end on the laptop with
`jupyter nbconvert --execute`. Two things differed from the Kaggle run: the model was Qwen3-0.6B,
since it fits 4 GB, and four of the seven configurations ran (challenge 0-shot and 6-shot, validation
0-shot, NEU-ESC 0-shot). What the run showed:
- The code cloned at the expected commit, and every frozen input was present.
- UIT-VSFC and NEU-ESC were fetched and matched their pinned hashes (NEU-ESC with the owner's token).
- The sanity check passed, and all four configurations finished and wrote their summaries and
  predictions (4.6 min).
- The summary table was printed and `h7_results.zip` was built (2.9 MB).
- `results merge --dry-run` lists exactly the new run folders.
- The zip holds no data file. A scan of it against all UIT-VSFC and NEU-ESC text (35,229 sentences of
  30 characters or more) matched no NEU-ESC post. It matched only three stock phrases inside the
  project's own challenge sentences.

**What only Kaggle can show:**
- Two GPUs in parallel. The laptop has one GPU, so the queue code ran as one queue.
- Qwen3-4B's memory on a T4: 8 GB of fp16 weights on 15 GB, and the token budget (12,288) was set
  for it.
- Kaggle's installed package versions. Cell 4 installs `transformers>=4.51` if the installed one is
  older.

Each cell stops loudly on an error, and a rerun in the same session resumes where the last one
stopped.

**The owner's steps** (about 5 minutes of clicking, then 40–70 minutes of background run):

1. Kaggle → *Create → New Notebook → File → Import Notebook*; import
   `notebooks/kaggle_h7_llm.ipynb` from the GitHub repository (or upload the file).
2. *Settings*: Accelerator **GPU T4 x2**, Internet **On**.
3. *Add-ons → Secrets*: `HF_TOKEN`, from the Hugging Face account that accepted the conditions on
   the NEU-ESC dataset page; tick *Attach to notebook*. Without it, only the NEU-ESC configuration is
   skipped.
4. *Save Version → Save & Run All (Commit)*; the browser can be closed.
5. From the version's *Output*, download `h7_results.zip` and extract it into `kaggle_results/`, never
   over the repository. Then run:
   ```
   vifeedback results merge kaggle_results --dry-run
   vifeedback results merge kaggle_results
   vifeedback study h7-decide
   ```

---

## Appendix A — Cycle 3 outcome (v3, closed 2026-09-28)

The v3 progress table as it stood when Cycle 3 closed. The rules are in `cycle3.yaml` v1–v6.

Every step that needed no owner input has run. Development results are not confirmations. The
human-typed challenge v2 needs time and people the owner does not have now, so the confirmation moved
to **NEU-ESC test** (real forum posts from another university, human labels; ADR-030, `cycle3.yaml`
v5), with rules committed before the cells were computed. Challenge v2 stays optional.

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

## Appendix B — v2 register (closed 2026-09-27)

Everything v2 could close without the owner, with its evidence. Items marked 👤 carried over into v3, and those still open into v4 § 1.

### B1. Outstanding issues (engineering, documentation)

| ID | Issue | What closes it | Who | Status |
|---|---|---|---|---|
| I1 | `mypy` is advisory in CI (35 errors in 13 files), so type errors can merge | Fix them; make `mypy` a blocking CI step | me | ✅ cde2174 |
| I2 | The Docker image has never served the real model. CI smoke-tests it without one | `make docker-e2e`: build, mount the released artifact, `/readyz` 200, golden `/v1/classify` cases | me | ✅ passes; it found `/version` reporting "unversioned", now fixed (f7b456c) |
| I3 | Status lines went stale: STATUS R8 ("XLM-R control running"), R10 ("benchmark pending"), P7 ("revisions not pinned"); REVIEW_COMPLIANCE ("H3 pending", an outdated "what happens next") | One consistency sweep, then the compliance audit re-run at Cycle 2 close | me | ✅ swept; compliance re-audited |
| I4 | Kaggle runs carry no git SHA: the code is uploaded as a zip. They do carry `source_sha256` | Map each recorded source hash to the commit whose `src/` produces it; record it | me | ✅ `results/provenance.json`: Kaggle runs → 5b2858c; 2 cRT runs from an uncommitted tree |
| I5 | The served model is not downloadable (R11: artifacts another researcher can fetch) | A publish command with a dry run, and a model card (data, per-class metrics, limits, manifest hash). The upload stays with the owner | me → 👤 | ✅ `serve publish` (dry run) · 👤 `--upload` |
| I6 | Local leftovers: blocked INT8 staging directories, `kaggle_results/` | Delete them. Keep `.previous-sentiment` (rollback) and the p9 checkpoint (H6) | me | ✅ 270 MB removed after a dry-run merge showed nothing unmerged |
| I7 | No explicit research-question list (review § 11, 🔶) | RQ1–RQ5 in RESEARCH_REPORT, each linked to its evidence | me | ✅ the report had Q1, Q2, Q4–Q6; Q3 (deferred), Q7, Q8 added |

### B2. Experiments not yet run or not yet decided

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

### B3. Results not yet confirmed

| ID | Claim | Why it is not confirmed | What confirms or refutes it | Status |
|---|---|---|---|---|
| C1 | H2 augmentation makes the model robust to missing diacritics | Shown only on *synthetic* noise (`strip_diacritics`, scripted teencode) | H6: `unaccented_typed` and `teencode_typed` rows, constructed (not typed by users) | ✅ confirmed for unaccented (0.32 → 0.74, p < 0.001); teencode not demonstrated (0.975 → 0.875, p = 0.13 at one seed; ADR-028) |
| C2 | TF-IDF beats PhoBERT on `facility` (0.921 vs 0.905) | One model each on dev, no interval | H5 descriptive: paired bootstrap on the same validation rows, plus the 5-fold out-of-fold view | ✅ refuted: −0.021 [−0.060, +0.015] |
| C3 | The model handles negation | 44 templated pairs, simple *không* forms | `negation_pair` rows: *đâu có*, *chẳng … chút nào*, *chưa bao giờ*, negated negatives, both models | ✅ measured: both pair members right in 60% (CE) / 67% (augmented) of 15 pairs, against 36/36 on the simple probe; n is small |
| C4 | Served p95 latency meets the 30 ms target | Met in one session (23.9 ms), missed in the latest (33.6 ms) | Three sessions on AC power in the same Windows power mode; report the median and range. The claim stays the ratio | ✅ met: served p95 19.2 and 20.0 ms in the two reportable sessions (all six passes 18.7–20.7 ms; `latency/sessions/`). The 33.6 ms session had PyTorch 1.6× slower too: machine state |
| C5 | Out-of-scope input behaves sensibly | Never measured (RESEARCH_REPORT § 7) | Confidence on the 20 `out_of_scope` rows vs `objective_neutral`, descriptive | ✅ measured: it does not. Mean confidence 0.86–0.90 on off-topic text |
| C6 | "The neutral gap is not in the classifier" (H1) | Rules out the head, but cannot tell label ambiguity from representation | A1 (human audit) | 👤 |
| C7 | LLM results on UIT-VSFC are fair | Pretraining contamination unknown | The challenge set is the primary LLM comparison; validation is secondary | by design |
