# Next Plan — experiments and improvements after Cycle 1

**Written 2026-09-27, after Cycle 1 closed.** Each step follows from a measured result, not from a
wish list. Steps run in order unless marked *parallel*. Every experiment step is declared in
`configs/experiments/` **before** its first run, with its decision rule in `evaluation/decisions.py`,
as Cycle 1 was.

## Where the evidence leaves us

| What Cycle 0–1 established | What it means for the next steps |
|---|---|
| Neutral errors are confident; boundary shifts, logit adjustment and cRT all move macro-F1 by noise (H1) | The gap is **not in the classifier**. Label ambiguity vs representation is the open question, and only a human audit can split it |
| Augmentation repairs missing diacritics (0.27 → 0.64 on test), clean accuracy unchanged, other noise unchanged (H2) | A deployable robustness gain exists. It is in-family only, so real noisy input is untested |
| XLM-R lost ~40% of its gap to preprocessing, but still trails PhoBERT (H3) | More encoders are only worth running with a specific question, such as raw-text or social-domain pretraining |
| Shared encoder: no gain at λ 0.3, sentiment cost at λ 1 (H4) | Multi-task is a serving-cost option, not a quality lever |
| INT8 is 1.7× faster but costs 0.088 neutral F1; static INT8 breaks the model | The efficiency stretch target (≥ 1.5× with neutral loss ≤ 0.02) is unmet by exactly the minority class |
| The test split has now been evaluated 28 times | **New claims need new evaluation data.** The official test remains for historical comparison only |

---

## Step 0 — Engineering debt and one deployment decision *(1–2 days; no GPU)*

| Task | Why | Done when |
|---|---|---|
| **0.1** Fix the 35 mypy errors; make `mypy` a blocking CI step | Type checking is advisory, so type errors can merge | CI fails on a type error |
| **0.2** Run the Docker image with the released artifact mounted: `/readyz` 200, `/v1/classify` golden cases | CI smoke-tests the image without a model; the real-model path was only simulated | A scripted `make docker-e2e` passes locally |
| **0.3** **Decide which model the service runs** (owner) | The H2-augmented model matches CE on clean text and is 2.3× more robust without diacritics. The decision rests on validation (H2 met its declared rule there); the test gate only confirmed it | Recorded as an ADR *and* in EVALUATION_PROTOCOL § 4 as a decision a test result touched; released through the gate |
| **0.4** Publish the served checkpoint + ONNX with a model card and SHA-256 (owner: HF account) | Review R11: artifacts another researcher can fetch | Model card lists data, metrics per class, limitations, the manifest hash |
| **0.5** Clean local leftovers (blocked INT8 staging, `.previous-sentiment`, `kaggle_results/`) | 1.3 GB of disk | — |

## Step 1 — The neutral audit *(owner: ~6–8 h of annotation; analysis: CPU)*

The single highest-value step: every remaining explanation for the neutral gap waits on it.

**Method** ([ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)).

1. `python -m vifeedback.cli study audit-sheet`, then annotate the 160 rows blind: model columns
   hidden until the label, subtype and gold assessment are filled.
2. Agreement: a second annotator on ≥ 50 rows, including all 40 random rows. If there is no second
   person, re-annotate the same rows after ≥ 24 h and report *intra*-annotator agreement.
3. Report per stratum: gold `agree / ambiguous / incorrect`, neutral subtypes, Cohen's κ. Estimate
   corpus-level rates **only** from the random stratum (Wilson intervals).

**Decision rule** (to be frozen in `cycle2.yaml` before annotation starts; thresholds are proposals):

| Audit finding, within the confident neutral-error strata | Next experiment |
|---|---|
| **≥ 30% "incorrect gold"** | **E04 label correction**: OOF-ranked relabelling of *train* only, then a 2×2 (original/corrected labels × CE/augmented), 3 seeds. Evaluation labels never change |
| **≥ 40% "ambiguous"** (mostly `mixed` / `insufficient_context`) | The task ceiling is annotation policy. Report it, try soft labels / label smoothing on neutral once, and stop optimizing neutral F1 on this benchmark |
| **Mostly "agree"**: gold defensible, model wrong | Representation. Candidates in Step 3's encoder option: raw-text pretraining (BamiBERT), social-domain pretraining (ViSoBERT), or task-adaptive pretraining (E13) |

## Step 2 — Independent evaluation data *(owner + CPU; parallel with Step 1)*

The official test has been used 28 times, and all robustness evidence so far is synthetic. Before
Cycle 2 makes a confirmatory claim, freeze new data it has never touched.

| Set | Size | Content | Status |
|---|---|---|---|
| **Challenge set v1** (constructed) | 300–500 | CheckList-style: negation minimal pairs (extend the 44), mixed aspects (teaching + facility), suggestions with and without cue words, unaccented and teencode text as typed by people rather than by a script, code-switching, long sentences, out-of-scope inputs | Author, label under the guide, commit, freeze **before** any Cycle 2 model is trained |
| **Natural sample** (if obtainable) | 500–1,000 | Real student feedback from a source with a clear licence, labelled by the guide, deduplicated against UIT-VSFC | Only with a lawful source; otherwise state its absence as a limitation |

The challenge set is also the first **natural-noise** test of H2's augmentation: a person's
unaccented typing is not the same as `strip_diacritics`.

## Step 3 — Cycle 2: one specialization *(≤ 30 weight-updating runs or the GPU-hour equivalent)*

The review's rule: choose **one**. Two are designed below; the owner picks.

### Option A (recommended for an AI-engineer profile) — Encoder vs LLM on the hard cases (Study E)

**Question.** Where the encoder is weakest (neutral, mixed aspects, suggestions), does an
instruction-tuned LLM do better, and at what cost?

| Item | Design |
|---|---|
| Model | One open instruction model around 4B parameters (the review names Qwen3-4B), revision pinned; Kaggle T4 (it does not fit the 4 GB laptop GPU in fp16) |
| Prompting | Label definitions from the annotation guide; structured output constrained to the three labels. Zero-shot, and few-shot with **two** demonstration draws from train only. Prompt development capped at a fixed budget on a train subset, never on validation |
| Evaluation | Validation + challenge set, paired per example against the deployed encoder; per-class P/R/F1; invalid-output rate; tokens, latency and GPU-hours per 1k sentences |
| Link to Step 1 | Agreement of the LLM with the audit's *adjudicated* labels, especially on items the audit marked `ambiguous`, which tests whether the LLM resolves the policy the way annotators did |
| Caveat | Pretraining contamination of a public benchmark is unknown; the challenge set is the cleaner comparison |
| Declared outcomes | *LLM ≥ encoder on neutral F1 on the challenge set* → a routing design (encoder first, LLM on low-confidence neutral) is Cycle 3's question. *LLM < encoder* → report cost-adjusted: the fine-tuned 135M encoder wins, which is itself the portfolio result |

Optional follow-up (E19): LLM labels for unlabelled or ambiguous **train** items, audited, against an
equal-size repeated-data control. The LLM never defines evaluation truth.

### Option B (for an ML-engineer profile) — Recover INT8's speed without losing neutral

**Question.** Can the 1.5×-with-≤ 0.02-neutral-loss target be met by a smaller or more carefully
quantized model?

| Arm | Design | Selection |
|---|---|---|
| B1 Distilled student | 6-layer PhoBERT student initialised from alternate teacher layers; CE + temperature-scaled KL to the (augmented) teacher; control: the same student on hard labels only; 3 seeds each | Dev macro-F1 as usual |
| B2 Careful INT8 | Per-channel dynamic INT8; exclude embeddings/classifier from quantization; percentile or entropy calibration for static | Recipe chosen on a **held-out train subset**, then one acceptance on validation (review § 8.4: never select on the acceptance set) |
| B3 Student + INT8 | The best of B1 through B2's recipe | — |

Report a quality/latency/size Pareto table on the reference CPU (ratios, per the latency protocol).
**Supported** if any artifact reaches ≥ 1.5× over L3 at p50 with neutral F1 loss ≤ 0.02 and macro-F1
non-inferior at 0.005.

### Either way

- Declare `configs/experiments/cycle2.yaml` (hypotheses, controls, budget, rules) before the first run.
- Save the finalist checkpoint whenever a hypothesis advances (Cycle 1 had to retrain for its gate).
- Commit run artifacts as each batch finishes (ADR-023).

## Step 4 — Quick win for topic *(parallel, CPU only, ~half a day)*

TF-IDF beats PhoBERT on `facility` (0.921 vs 0.905). **E08 stacking**: out-of-fold probabilities from
TF-IDF B4 and PhoBERT, a logistic-regression meta-model fitted on OOF only, evaluated on validation.
Declared success: topic macro-F1 +0.005 with 95% CI above 0 (paired by example), and `facility`/`others`
F1 not lower. It needs no GPU if the OOF topic probabilities are generated alongside another run.

## Step 5 — Close Cycle 2 and update the portfolio

- One confirmation on the **new** data (challenge set, natural sample), logged; the official test only
  for historical comparability.
- Update RESEARCH_REPORT, STATUS, README, and the model card; re-execute `02_results.ipynb`.
- Re-run the review compliance audit against the new state.

## Timeline and budget

| Week | Work | GPU |
|---|---|---|
| 1 | Step 0; Step 1 annotation (owner); Step 4 stacking | none |
| 2 | Step 2 challenge set (owner authoring, ~8–10 h); freeze v1; declare Cycle 2 | none |
| 3–4 | Step 3 (A or B) | A: ~6–10 Kaggle GPU-h · B: ~3–4 laptop GPU-h |
| 5 | Step 5: confirmation on new data, documents | < 1 h |

## What not to do

- No more sweeps for neutral F1 on the old validation set: 73 neutral examples cannot resolve the
  effects Cycle 1 measured, and three interventions already came back noise-sized.
- No new decision taken on the official test.
- No new encoder family without a named question it answers (Step 1's third branch is one).

## Decisions needed from the owner

1. Step 0.3: which model the service runs.
2. Step 1: who annotates, and whether a second annotator is available.
3. Step 2: whether any lawful natural data source exists.
4. Step 3: Option A (AI engineering) or Option B (ML engineering).
