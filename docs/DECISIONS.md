# Decision Log

One entry per decision that a future reader — or you in six weeks — would otherwise have to reverse-engineer
from the code. Append-only. Never edit a past entry; supersede it with a new one.

**Format:** `ADR-NNN · date · title · Status: Accepted | Superseded by ADR-MMM`
with **Context → Decision → Consequences**.

Per [ROADMAP § 5](ROADMAP.md#5-phase-plan), any gate that fails, any scope cut, and any addition to the
[§ 2 exclusion list](ROADMAP.md#2-scope-boundaries) requires an entry here before work continues.

---

## ADR-001 · 2026-09-22 · Macro-F1 is the headline metric · Accepted

**Context.** UIT-VSFC sentiment is 4.32% neutral (measured: 458 of 11,426 train, 167 of 3,166 test). On
the real test split, a classifier that never predicts neutral but is otherwise perfect reaches accuracy
0.947 and weighted F1 0.922, against a macro-F1 of 0.649. Most published results on this dataset report
weighted F1 or accuracy, which makes the minority class invisible.

**Decision.** Macro-F1 is the single headline metric for every claim, with weighted F1 and accuracy always
printed beside it.

**Consequences.** Absolute numbers will look lower than the "92–94%" commonly quoted for this dataset, and
Phase 2 must explicitly reconcile the metric definitions or the project will read as a failed reproduction.
Checkpoint selection, threshold tuning and the champion choice all optimize macro-F1, which makes Tier A of
Phase 4 (imbalance handling) the highest-value work in the project.

---

## ADR-002 · 2026-09-22 · Keep the official split unchanged · Accepted

**Context.** The official 11,426 / 1,583 / 3,166 split is a random ~70/10/20 partition. Cross-validation or
a temporal split would give a better-grounded estimate of generalization.

**Decision.** Use the official split verbatim. No re-splitting, no folding dev into train.

**Consequences.** Comparability with published work is preserved, which is what makes the reproduction
claim meaningful. The cost is that reported numbers are in-distribution and probably optimistic for a real
deployment; this is recorded as limitation L8 in the [data card](DATA_CARD.md#8-known-limitations)
and must appear in the model card.

---

## ADR-003 · 2026-09-22 · Single dataset; cross-domain deferred to Phase 8 · Accepted

**Context.** Several Vietnamese sentiment corpora exist (UIT-ViSFD, UIT-VSMEC, NEU-ESC, VLSP, AIVIVN).
Training on their union would likely raise absolute scores.

**Decision.** Train and evaluate on UIT-VSFC only. Other corpora are used solely for zero-shot cross-domain
evaluation in Phase 8, reported in a separate table.

**Consequences.** Lower absolute ceiling, preserved comparability. Any future change requires a new ADR.

---

## ADR-004 · 2026-09-22 · Word segmentation is treated as an open question, not a given · Accepted

**Context.** PhoBERT's model card mandates word-segmented input. Published work
([arXiv:2301.00418](https://arxiv.org/abs/2301.00418)) finds segmentation unnecessary for Vietnamese
sentiment classification. `py_vncorenlp` requires a JVM, which threatens both the Docker image size target
(S8) and the latency target (S5).

**Decision.** Treat it as hypothesis H2 and measure it on **both** axes — accuracy and latency — in
Phase 3. Segmentation runs offline during data preparation regardless, so no JVM appears in the training
loop or the serving path.

**Consequences.** Phase 3 becomes a joint accuracy-and-cost experiment rather than a checklist item, and it
produces the serving-pipeline decision. If H2 holds, the JVM leaves the deployment entirely and the
project's largest latency win arrives in Week 4.

---

## ADR-005 · 2026-09-22 · Both dynamic and static INT8 are pre-registered · Accepted

**Context.** ONNX Runtime dynamic INT8 quantization is widely reported as a speedup, but on CPUs without
AVX512-VNNI it can be *slower* than FP32. The reference machine's instruction-set support is unknown at
planning time.

**Decision.** Pre-register dynamic INT8 (L4), static INT8 (L5) and OpenVINO INT8 (L6) as separate ladder
steps in Phase 6, and record the reference CPU's instruction-set flags before benchmarking.

**Consequences.** A dynamic-quantization regression becomes a confirmed hypothesis (H3) rather than a dead
end, and the project still has two viable paths to the S5 latency target.

---

## ADR-006 · 2026-09-22 · Drop the lowercasing ablation condition · Accepted

**Context.** The Phase 3 preprocessing ablation included condition P5 (`norm` + lowercasing). Gate G0
measurement shows UIT-VSFC contains **0 sentences with any uppercase character** (0 of 16,175).

**Decision.** Remove P5 from the ablation matrix. Record the measurement in the data card instead.

**Consequences.** Five GPU runs saved, and one fewer flat line in the results table. The same
measurement also confirms the corpus is *not* word-segmented (1 sentence contains an underscore), so
the P0-vs-P1 segmentation contrast — the part of Phase 3 that matters — is measuring a real
difference. Generalizes to a rule: measure whether a preprocessing step is a no-op before spending
GPU hours proving it.

---

## ADR-007 · 2026-09-22 · Teencode and diacritics are robustness targets, not error-analysis targets · Accepted

**Context.** The project scope named negation, teencode and missing diacritics as the error-analysis
themes. Gate G0 measurement, over the full 16,175-sentence corpus:

| Phenomenon | Sentences | Share |
|---|---|---|
| Negation markers | 3,300 | 20.40% |
| Missing diacritics | 23 | 0.14% |
| Teencode probes | 26 | 0.16% (19 of which are `ok`) |

Negation is abundant. Teencode and undiacritized text are effectively **absent** — UIT-VSFC was
distributed pre-lowercased, pre-tokenized and orthographically clean. An error analysis of teencode
on this test set would be an analysis of roughly 26 sentences.

**Decision.** Split the theme in two rather than dropping it.

* **Negation** stays an error-analysis target, retargeted from *presence* to *scope*: a bag-of-words
  model already captures `không` → negative as a unigram (84.3% of negation-bearing sentences are
  negative), so the discriminating cases are scope inversions such as `không có gì để chê`.
* **Teencode and missing diacritics** become **robustness** targets, measured by the controlled
  perturbation suites as a macro-F1 drop under an induced distribution shift.

Also corrected: the `SUGGESTION` taxonomy entry claimed suggestions are usually `neutral`. Measured,
sentences containing `nên`/`cần`/`mong` are **91.1% negative** — the guideline treats a request for
change as implicit criticism. The taxonomy entry has been rewritten.

**Consequences.** The claim changes from "the model struggles with teencode" (unsupportable on this
benchmark) to "macro-F1 falls by X pp under de-diacritization, a shift real deployment traffic
exhibits and this benchmark does not" — narrower, measurable, and honest. The perturbation suites
gain importance rather than losing it, since they are now the *only* evidence on informal
orthography. Reported in its own section, never merged into the clean-test table.

---

## ADR-008 · 2026-09-22 · Revise the success criteria against the measured baseline · Accepted

**Context.** S1–S4 were pre-registered before any experiment. Gate G1 measured the baseline ladder and
falsified the estimates they rested on:

| Criterion | Assumed baseline | Measured baseline (dev) |
|---|---|---|
| S1 sentiment macro-F1 ≥ 0.76 | best baseline ~0.70 | **0.782** (B3 + tuned priors) |
| S2 topic macro-F1 ≥ 0.72 | best baseline ~0.65 | **0.774** (B3 + tuned priors) |
| S3 lift ≥ +0.08 over baseline | — | would demand PhoBERT ≥ 0.862 |
| S4 neutral F1 ≥ 0.40 | linear model ~0.05–0.30 | **0.503** (B2 + tuned priors) |

The best published macro-F1 on UIT-VSFC sentiment is ≈0.83. S3 as written therefore required beating
the published state of the art by ~3 points merely to pass, and S1/S4 would be satisfied by the TF-IDF
baseline alone — a criterion a transformer cannot fail is not a criterion.

The cause is identifiable and worth recording: the ranges were anchored on the literature's framing of
UIT-VSFC as noisy student feedback, and were not re-derived after Gate G0 measured the opposite — a
pre-lowercased, pre-tokenized, 99.86% diacritized corpus with a median length of 11 syllables. That is
close to ideal for TF-IDF.

**Decision.** Revise the thresholds, and record the revision rather than restating the original ones as
if they had been met. The revised bar is set **relative to the measured baseline**, so it cannot be
satisfied by the baseline itself.

| ID | Criterion | Was | **Revised minimum** | **Revised target** |
|---|---|---|---|---|
| S1 | Sentiment test macro-F1 | ≥ 0.76 | **≥ 0.80** | ≥ 0.84 |
| S2 | Topic test macro-F1 | ≥ 0.72 | **≥ 0.79** | ≥ 0.83 |
| S3 | Lift over the **tuned** baseline, sentiment | ≥ +0.08 | **≥ +0.025 and p < 0.05** | ≥ +0.05 |
| S4 | Neutral-class F1 | ≥ 0.40 | **≥ 0.55** | ≥ 0.65 |

**Consequences.**

* S3 is now a *significance* criterion rather than an effect-size one. On a 3,166-example test set with
  167 neutral examples, +0.025 macro-F1 that survives a paired bootstrap is a real result; +0.08 was
  never a realistic ask against a baseline this strong.
* The comparison must be against **B3 + tuned priors (0.782)**, the strongest baseline, never against
  B1 (0.737). Quoting the weaker number would inflate the headline lift by 45 points of relative
  improvement for free, and is exactly the failure mode ROADMAP § 9 rule 1 was written to prevent.
* The CV snippet's numbers move accordingly: the honest sentence is "0.78 → 0.8x", not "0.66 → 0.82".
  This is a smaller headline and a much more defensible one, and the interesting claim shifts onto the
  neutral class and the latency work.
* Standing rule adopted: **re-derive pre-registered ranges after the data profile is measured, and log
  the revision.** Recording a stale prediction and quietly forgetting it is the failure this ADR exists
  to prevent repeating.

---

## ADR-009 · 2026-09-22 · Train locally on the RTX 3050; keep Colab as overflow · Accepted

**Context.** The plan assumed all transformer fine-tuning would run on Colab, with the laptop reserved
as the CPU reference machine. The reference machine turns out to carry an **NVIDIA RTX 3050 Laptop GPU
(4.29 GB, compute 8.6)**. Measured: PhoBERT-base at `max_length=96`, batch 32, fp16 trains one epoch on
UIT-VSFC in **69 seconds**, so a 4-epoch run costs ~4.6 minutes and a 5-seed sweep ~23 minutes.

**Decision.** Run PhoBERT-base training locally. Colab is retained only for what does not fit in 4.3 GB
of VRAM — PhoBERT-large, XLM-R-large, and any batch-size-hungry Phase 4 recipe.

**Consequences.**

* Iteration speed rises sharply and the Colab session-limit risk (R4) largely disappears for Phase 2–4.
  R4 is downgraded from Medium/Medium to Low/Low for those phases.
* **A new constraint replaces it, and it is not optional.** The laptop is both the training machine and
  the latency reference machine. Training heats the package and Windows will down-clock it, so a
  benchmark run on a hot machine measures the thermal state, not the model. Rule adopted for Phase 6:
  *no benchmark runs while training is running, and a documented cool-down before any timed run.* The
  benchmark harness already requires 5 repetitions with a >10% disagreement triggering a re-run
  (docs/EVALUATION_PROTOCOL.md § Latency harness), which is the detector for a violation.
* The 4.3 GB ceiling is a real limit, not a formality: PhoBERT-large plus AdamW optimizer state will not
  fit. Those runs stay on Colab, and the split is recorded per run in `env.json` via `gpu_info()`.
* Compute budget in [EXPERIMENT_MATRIX § 3](EXPERIMENT_MATRIX.md#3-run-inventory-and-compute-budget)
  is revised: Phases 2–4 move from ~36 Colab GPU-hours to ~6 local GPU-hours.

---

## ADR-010 · 2026-09-22 · Unblock VnCoreNLP with a pip-packaged JDK; keep models outside the repo · Accepted

**Context.** Risk R1 said a missing JVM would make RDRSegmenter — condition **P1**, the canonical
PhoBERT pipeline and the heart of hypothesis H2 — unmeasurable on this machine. Confirmed at Gate G0:
no system Java. Three further obstacles surfaced while trying to remove it:

1. `py_vncorenlp.download_model()` shells out to `wget`, which does not exist on Windows, and pulls
   ~200 MB of NER, POS and dependency models we never use.
2. `py_vncorenlp.VnCoreNLP()` `chdir()`s into its model directory and never returns, silently
   relocating the calling process.
3. **VnCoreNLP resolves its model directory from the jar's own URL and never URL-decodes it.** With
   the repo at `D:\GitHub\ViFeedback NLP Service`, the space became `%20` and the segmenter died
   with `wordsegmenter.rdr is not found`.

**Decision.**

* Provide the JVM through **`jdk4py`**, an ordinary pip package (OpenJDK 25), rather than a
  system-level install. `ensure_java()` prefers a JVM already on PATH and falls back to it, so the
  project does not depend on how the developer's machine happens to be configured.
* Replace the downloader: fetch only the three `wseg` assets over `urllib`, idempotently.
* Restore the working directory around JVM construction.
* Keep VnCoreNLP's models **outside the repository**, under `~/.cache/vifeedback/vncorenlp`, and
  raise with an explanatory message if the resolved path contains a space.

**Consequences.**

* P1 is now measurable here, so Phase 3 can run the full ablation instead of recording a gap. All
  three segmenters verified to produce the expected output:
  `giảng viên nhiệt tình với sinh viên` → `giảng_viên nhiệt_tình với sinh_viên`.
* **R1 is downgraded for *measurement*, not for *deployment*.** Serving still needs a JVM, +~180 MB
  of image and a per-request JNI call. That cost is exactly what H2 is meant to weigh, so the
  experiment gets sharper rather than easier.
* The space-in-path defect is a genuine deployment hazard, not a local quirk: a Docker `WORKDIR` or a
  user checkout under `C:\Program Files` or `/home/my user/` would hit it. It is now a loud error
  with a stated cause instead of a confusing `IOException`, and it is one more entry on the cost side
  of the H2 ledger.

---

## ADR-011 · 2026-09-22 · Defer the Gate G2 test evaluation and bundle it into G4 · Accepted

**Context.** Gate G2 as written required "test evaluated exactly once and logged". Phase 2 finished
with 10 dev runs and no test evaluation, for a reason worth stating plainly: **the trainer did not
save checkpoints**, so evaluating the reference configuration on test would have meant retraining all
ten runs — roughly 50 GPU-minutes to produce a number that Phases 3 and 4 will supersede anyway.

**Decision.** Two changes.

1. **Checkpoint saving added** (`run_once(save_checkpoint=True)`), off by default because a
   135M-parameter checkpoint is ~540 MB and a 5-seed sweep would be 2.7 GB. On for anything that may
   become a champion, so a later test evaluation costs seconds rather than a retrain.
2. **The G2 test evaluation is deferred and bundled into G4.** At G4 a single test pass evaluates
   three things together: the tuned TF-IDF baseline, the Phase 2 reference configuration (for the
   literature reproduction), and the Phase 4 champion.

**Consequences.**

* This *tightens* protocol hygiene rather than loosening it. The test-set budget is ~8 evaluations
  for the whole project (docs/EVALUATION_PROTOCOL.md § 4); bundling spends one where the original
  plan spent two, and the deferred number was going to be superseded regardless.
* G2 is therefore assessed on its substantive criteria, all of which are met: PhoBERT beats the tuned
  baseline on dev macro-F1 for both tasks (+0.062 and +0.023, at 7.8× and 12.8× the seed std), seed
  std is reported, and the weighted-F1 reproduction lands inside the published range.
* The literature comparison is currently **dev-vs-published-test**, which is stated wherever it
  appears and resolved at G4. It is not treated as a completed reproduction until then.
* Generalizes to a rule: *save what an experiment might need later, before the experiment, not after
  discovering it is gone.* The cost of the omission here was one deferred gate; on a larger sweep it
  would have been the sweep.

---

## ADR-012 · 2026-09-22 · H2: segmentation helps and is cheap; serve with pyvi · Accepted
> **Partially superseded by ADR-018.** The serving decision stands. The claim that this
> *refuted* arXiv:2301.00418 was a misreading of a conditional conclusion and is retracted —
> the measurement **replicates** that paper's deep-learning finding.

**Context.** H2 predicted that word segmentation would (a) not improve accuracy and (b) dominate p95
latency, making it droppable for a large deployment win. Gate G3 measured both halves. Both are wrong.

| | H2 predicted | Measured |
|---|---|---|
| Accuracy effect | none | **+0.0234 macro-F1** (t = 8.58, p = 0.0010, 5/5 seeds, ranges non-overlapping) |
| Latency cost | dominates p95 | **0.606 ms p95** — 1.2% of the model's 50.8 ms |

~~The published finding this hypothesis was anchored on replicates *precisely* on the metric it
reported ... an artifact of aggregating over a 4% class.~~ **RETRACTED — see ADR-018.** The paper's
conclusion is conditional: segmentation may be unnecessary for *traditional classifiers*, and **is
necessary** for deep-learning models using BPE. PhoBERT is the latter, so our +0.0234 macro-F1
**agrees** with the paper. What remains is a clean independent replication with a quantified effect
size, a per-class breakdown and a latency cost the original does not report.

**Decision.** Segment, and serve with **pyvi**.

Segmenter *choice* is immaterial: VnCoreNLP 0.8670, pyvi 0.8643, underthesea 0.8618 — a 0.005 spread
inside a 0.007–0.010 seed std. Whether to segment *at all* is material. So take the accuracy and pay
the smallest price: pyvi costs −0.0027 macro-F1 versus VnCoreNLP (undetectable), runs **2× faster**
(0.311 ms vs 0.606 ms p95), needs no JVM, removes ~180 MB from the Docker image, and is immune to the
space-in-path defect of ADR-010.

**Consequences.**

* Risk **R1 is closed**, in the strongest available way: the deployment never needs a JVM, and it is
  not a workaround — it is what the measurement recommends.
* The project loses its most quotable predicted headline ("segmentation was 60% of p95 and I removed
  it") and gains a better one: *a published negative result does not replicate once the minority class
  is made visible.* That is a finding about the literature, not just about this pipeline.
* All downstream phases switch to the `seg_pyvi` variant as the default preprocessing. Phase 4's
  champion search and Phase 6's benchmark both run on segmented input.
* Two configuration-only wins are already banked: `max_length` 96 (Gate G0) and dynamic padding give
  **3.49×** on p95 (177.5 → 50.8 ms), meeting the S5 minimum before any ONNX or INT8 work.

---

## ADR-013 · 2026-09-22 · Seed-level paired t-test is the primary significance instrument · Accepted

**Context.** docs/EVALUATION_PROTOCOL.md § 3 named the per-run **paired bootstrap** the primary test.
At Gate G3 it disagreed with the seed-level evidence on the same comparison:

| Instrument | Verdict on P0 vs P1 |
|---|---|
| Paired t-test over seeds (n = 5) | +0.0234, t = 8.58, **p = 0.0010**, Cohen d = 3.84, 5/5 positive, ranges non-overlapping |
| Per-seed paired bootstrap on dev | **1/5** significant; BH-FDR retains 1 |

Not a contradiction. They estimate different things: the bootstrap estimates **evaluation-set sampling**
variance ("would this hold on another dev set of this size?"), the t-test estimates **training**
variance ("would this hold on another run?"). Dev carries **73 neutral examples** bearing one third of
the macro average, so resampling it moves macro-F1 by **±0.027** — wider than the +0.023 effect. The
test set would only narrow that to ~0.019.

**Decision.** For macro-F1 comparisons between configurations, the **seed-level paired test over the
5 canonical seeds is primary**; the paired bootstrap is reported alongside as the evaluation-set
uncertainty, not as the verdict. Both are always shown, and a disagreement is stated rather than
resolved by picking the favourable one.

**Consequences.**

* **Phase 4 selection changes.** Recipes must be compared on the 5-seed mean, not on a single dev run.
  A single-seed dev comparison cannot resolve anything below ~0.027 macro-F1, and almost every Tier A–C
  increment is expected to be smaller than that. Exploration at 3 seeds, finalists at 5, remains the
  rule — but the *decision* is now explicitly a seed-level one.
* **The project has a stated resolution floor**, which belongs in the write-up: on this dataset, an
  honest single-run dev claim cannot be finer than ~0.027 macro-F1. Any paper or repo reporting a
  +0.01 improvement on UIT-VSFC from one run is reporting noise.
* This is a *strengthening* of the protocol, and it was found by running the two tests and refusing to
  discard the inconvenient one.

---

## ADR-014 · 2026-09-22 · Kaggle is the preferred free GPU; Colab is the fallback · Accepted

**Context.** ADR-009 established that PhoBERT-base trains locally (3.6 GB measured of 4.29 GB), leaving
only `phobert-large` (~7.9 GB) and unfrozen `xlm-roberta-base` (~5.9 GB) needing an external GPU. The
original plan named Colab. Kaggle's free tier was not compared.

| | Kaggle free | Colab free |
|---|---|---|
| GPU | P100 16 GB, or 2× T4 16 GB | T4 16 GB, not guaranteed |
| Quota | **30 GPU-hours/week, stated** | unstated; throttled by prior use |
| Session | up to 9 h, rarely preempted | ~12 h, preemptible at any time |
| Output | `/kaggle/working` persisted with the notebook version | lost unless downloaded |
| Input | versioned private Datasets | manual upload each session |

**Decision.** Kaggle is the default external platform; Colab is kept as a fallback. Both notebooks are
maintained and both are logic-free wrappers around the same CLI.

**Consequences.**

* Raw GPU capability is not the deciding factor — both offer 16 GB, and the largest planned model needs
  7.9 GB. What decides it is that a **stated quota** and **persisted output** make a run reproducible,
  and risk R4 (lost session) is what actually threatened the plan.
* Kaggle's versioned Datasets give a cleaner path than re-uploading a zip each session, which matters
  because the repo has no git remote yet.
* Neither platform may produce a latency number. The reference machine is the laptop
  (AMD Ryzen 5 6600H, no AVX512-VNNI) and is recorded in every `env.json`; a p95 from a cloud VM is not
  comparable and does not enter the registry.
* External GPU remains **optional**. 47 runs and every gate through G3 have been completed with zero
  external compute, and Tiers A–D of Phase 4 need none either.

---

## ADR-015 · 2026-09-22 · Decision-threshold tuning must be cross-fitted; a Gate G1 conclusion is retracted · Accepted

**Context.** Gate G1 reported that per-class decision-prior tuning was "the single largest lever" at
**+0.035 macro-F1**, and concluded from it that the baseline's neutral-class failure was a
*decision-rule* problem rather than a *representation* problem. That conclusion was built on priors
that were **fitted on dev and scored on dev**.

Cross-fitting the same tuning (5-fold, priors never see the examples they are scored on) gives a very
different picture:

| Model | Untuned | Fit = eval | **Cross-fitted** | Optimism bias |
|---|---|---|---|---|
| Sentiment B1 | 0.7366 | 0.7720 | 0.7578 | +0.0143 |
| Sentiment B3 | 0.7547 | 0.7823 | **0.7708** | +0.0116 |
| Sentiment B5 (already class-weighted) | 0.7679 | 0.7745 | **0.7390** | +0.0355 |
| Topic B3 | 0.7529 | 0.7741 | 0.7516 | +0.0225 |
| **PhoBERT, raw** | 0.8436 | 0.8599 | **0.8422** | +0.0177 |
| **PhoBERT, segmented** | 0.8670 | 0.8773 | **0.8627** | +0.0145 |

**Decision.** Three changes.

1. **Retract the G1 claim.** On **PhoBERT the honest gain is zero** — −0.0014 and −0.0043, neither
   significant, 2/5 and 1/5 seeds positive. The entire apparent improvement was optimism bias. On
   TF-IDF a smaller real gain survives (+0.016 on sentiment B3), but topic gains nothing.
2. **`crossfit_class_priors()` and `prior_tuning_report()` added** to the codebase, with tests. Any
   reported tuned-threshold number must be cross-fitted. `tune_class_priors` stays as the right way to
   *fit* priors for deployment; it is the wrong way to *estimate what they are worth*.
3. **Phase 4 Tier A is re-scoped to training-time methods only** — class weighting, focal loss, logit
   adjustment. Post-hoc thresholding is off the table for PhoBERT because it does not generalize here.

**Why it fails, which is the interesting part.** Dev holds **73 neutral examples**. Cross-fitting
estimates the priors from ~58 of them and scores on ~15. That is far too little to fit a decision
boundary that transfers — the same 73-example bottleneck that makes the dev set underpowered for
significance testing (ADR-013) also makes it too small to tune a threshold on. Two apparently
unrelated methodological problems, one root cause.

Stacking makes it worse, not better: **B5 (class-weighted) plus tuned priors cross-fits to 0.7390,
*below* its own untuned 0.7679.** Correcting for imbalance twice overshoots.

**Consequences — the corrected numbers, which favour the model, not the author.**

| | Reported at G1/G3 | **Corrected** |
|---|---|---|
| Best honest sentiment baseline | 0.782 | **0.7708** (B3 + cross-fitted priors) |
| Best honest topic baseline | 0.774 | **0.768** (B4 LinearSVC — never used priors) |
| Sentiment lift over baseline | +0.085 | **+0.096** |
| Topic lift over baseline | +0.023 | **+0.029** |

The inflated baselines *understated* PhoBERT's advantage, so correcting them improves the project's
headline. That direction of error is worth stating plainly: the bug was not convenient, and it was
found by testing a result that had already been written up as a success.

**Standing rule adopted:** *any hyperparameter fitted on the evaluation set must be cross-fitted
before its benefit is reported.* This applies to thresholds, calibration temperature, ensemble weights
and soup coefficients — every one of them is coming later in this project.

---

## ADR-016 · 2026-09-22 · Scaling the encoder buys nothing; PhoBERT-base is the ship · Accepted
> **Narrowed by ADR-019 (review R8).** The shipping decision stands. The general claims do not:
> the evidence supports *"PhoBERT-base performs best among the configurations tested under this
> budget"*, not that encoder capacity has no remaining value. XLM-R was fed pyvi-segmented text its
> tokenizer never saw in pretraining, so its −0.024 mixes model and preprocessing effects.

**Context.** Phase 4 Tier E ran the two models that exceed the 4.29 GB laptop GPU on a Kaggle T4.
Sentiment, `seg_pyvi`, 5 seeds, dev:

| Model | Params | Macro-F1 | Neutral F1 | Δ vs base | Where |
|---|---|---|---|---|---|
| **`phobert-base`** | **135M** | **0.8643 ± 0.0092** | **0.663 ± 0.029** | — | laptop |
| `phobert-large` | 368M | 0.8560 ± 0.0036 | 0.636 ± 0.011 | **−0.0083** (1.2 pooled std) | Kaggle T4 |
| `xlmr-base` | 277M | 0.8403 ± 0.0063 | 0.604 ± 0.018 | **−0.0240** (3.0 pooled std) | Kaggle T4 |

**Decision.** Ship `phobert-base`. Close the model axis; neither larger model enters the champion
search.

**Consequences and reasoning.**

* **2.7× the parameters is not worth 1.2 seed-std in the wrong direction.** `phobert-large`'s seed
  range [0.8520, 0.8604] sits *inside* `phobert-base`'s [0.8494, 0.8750], so the difference is not
  even resolvable — and the point estimate is lower. Against the CPU latency objective it is
  strictly worse: 2.7× the FLOPs for no measurable accuracy.
* **`xlm-roberta-base` is clearly worse** (−0.024, 3.0 pooled std). Multilingual pretraining loses
  to Vietnamese-specific pretraining at the same scale, which is the result PhoBERT's own paper
  reports and this replicates.
* **`phobert-large` is unstable at this data size**, and visibly so: best epochs across seeds were
  [1, 4, 4, 4, 3], one seed peaked at **epoch 1** and degraded from there, and another started at
  macro-F1 0.7724. Dev-macro-F1 checkpoint selection is what keeps those runs usable at all.
* Two `xlmr-base` seeds **collapsed to neutral F1 = 0.000 at epoch 1** before recovering — the
  majority-class collapse the minority class invites, and a reminder that a 1-epoch budget would
  have produced a very different conclusion.
* **CafeBERT (560M) is not worth running.** The trend across 135M → 277M → 368M is flat-to-negative,
  and 560M would cost ~90 GPU-minutes to extend a line that is already answered.

**This closes Tier E as a negative result, and it is a useful one:** the remaining headroom on this
task is in the *minority class* and the *data*, not in encoder capacity. Tiers A, C and D keep their
priority; Tier E does not.

---

## ADR-017 · 2026-09-23 · ONNX export moves off the reference machine; the benchmark stays on it · Accepted
> **Narrowed by ADR-020.** Only INT8 *quantization* needs the blocked `onnx` package. With the
> TorchScript exporter pinned, FP32 export and its verification run on the reference machine.
> **Superseded by ADR-022.** The block applied only to freshly written DLLs; `onnx` 1.23 now imports
> and INT8 builds on the reference machine too.

**Context.** Phase 6's ONNX ladder (L3–L6) cannot run on the reference machine. Windows Application
Control blocks the `onnx` package's native extension:

```
ImportError: DLL load failed while importing onnx_cpp2py_export:
An Application Control policy has blocked this file.
```

`onnxscript` fails the same way, since it imports `onnx`. `torch>=2.6` routes `torch.onnx.export`
through `onnxscript`, so export is blocked end to end.

**`onnxruntime` is not affected** — it ships its own signed native libraries and loads cleanly, with
`CPUExecutionProvider` available.

This is a machine security policy, not a dependency problem. It is not fixable in code, and working
around it by disabling the policy is not a reasonable thing to do for a portfolio project.

**Decision.** Split the two operations along the line that already exists in this project:

| Operation | Where | Why |
|---|---|---|
| **Export + quantize** | Kaggle / Colab (Linux) | Deterministic and hardware-independent — an `.onnx` graph is the same artifact wherever it is produced |
| **Benchmark + serve** | **Reference machine only** | `onnxruntime` works here, and latency is meaningless anywhere else (ADR-014) |

An export cell is added to `notebooks/kaggle_train.ipynb`; the resulting `models/serve/<task>/`
directory comes back with the results archive and `vifeedback serve bench` runs against it locally.

**Consequences.**

* Phase 6 is **not blocked** — only its first step relocates. The measurement that matters still
  happens on the documented CPU.
* The PyTorch half of the ladder (L0–L2: `max_length`, dynamic padding, thread count) runs locally
  and is where the **3.49×** already came from, so the largest measured win is unaffected.
* **H3 remains testable.** The reference CPU has no AVX512-VNNI, and whether INT8 helps or hurts
  there is still answered by running the exported artifact under local `onnxruntime`.
* Risk register gains R11: a security policy can block a Python package's native extension while
  leaving a functionally adjacent one working. The failure mode is worth remembering — the error
  named the policy, but nothing in it suggested that the *runtime* would be fine.

---

## ADR-018 · 2026-09-26 · Retract the "refuted a published result" claim — I misread the paper · Accepted

**Context.** ADR-012, the README, BENCHMARK_COMPARISON and STATUS all claimed that this project
**falsified** the conclusion of [arXiv:2301.00418](https://arxiv.org/abs/2301.00418), and that the
published finding "word segmentation is unnecessary" was "an artifact of aggregating over a 4%
class". External review flagged this, and re-reading the abstract confirms the reviewer is right and
I was wrong.

The paper's conclusion is **conditional**, not general:

> "word segmentation maybe not be necessary for the Vietnamese sentiment classification corpus,
> which comes from the social domain" — for *traditional classifiers* (Naive Bayes, SVM) —
> but "word segmentation **is necessary** for Vietnamese sentiment classification when word
> segmentation is used before using the BPE method and feeding into the deep learning model."

PhoBERT is a deep-learning model using BPE. **Our +0.0234 macro-F1 from segmentation agrees with
the paper. It does not contradict it.** The paper also reports RDRsegmenter as the most stable
toolkit among {uitnlp, pyvi, underthesea}, which our own ordering reproduces
(vncorenlp 0.8670 > pyvi 0.8643 > underthesea 0.8618 on dev).

**How the error happened, because the mechanism matters more than the correction.** I anchored on
the paper's title — phrased as a question — and on the "under 1 percentage point" figure, and
treated the first clause of a two-clause conclusion as the whole conclusion. The
[PhoBERT repository](https://github.com/VinAIResearch/PhoBERT#notes) states the same requirement
explicitly, and I cited it in ADR-004 while still asserting the opposite two ADRs later. Nothing in
the measurement was wrong; the literature claim attached to it was never checked against the source.

**Decision.** Retract the refutation claim everywhere it appears. Replace it with what the
measurement actually supports:

> Quantified the effect of word segmentation in a PhoBERT pipeline — **+0.0234 macro-F1**
> (t = 8.58, p = 0.0010, 5/5 seeds, non-overlapping seed ranges) and **+5.42 pp neutral F1** —
> and measured the per-segmenter cost, independently reproducing both of the paper's
> deep-learning-relevant findings.

**Consequences.**

* The project loses its "field-level finding" and keeps a clean **independent replication** with a
  quantified effect size, a per-class breakdown and a latency cost the original does not report.
  That is a smaller claim and a true one.
* H2 in the README is **not** "falsified because the literature was wrong". Its *latency* half is
  still genuinely falsified by measurement (segmentation is 0.6 ms p95, not the majority of it);
  its *accuracy* half was a misreading of the source, not a finding.
* Standing rule: **a claim about someone else's paper is quoted from that paper, in the ADR, before
  it is written anywhere else.** Three documents repeated this claim because none of them carried
  the quote that would have refuted it.

---

<!-- Append new entries above this line. -->

---

## ADR-019 · 2026-09-27 · Narrow ADR-016 to the configurations tested · Accepted

**Context.** External review (R8) pointed out that ADR-016 drew general conclusions — "scaling the
encoder buys nothing", "multilingual pretraining loses to Vietnamese-specific pretraining",
"CafeBERT is not worth running" — from one recipe, four epochs and one preprocessing condition.
Checked against the runs, the review is right on three counts:

1. **XLM-R was evaluated only on `seg_pyvi` input.** pyvi joins syllables with underscores
   (`giảng_viên`). PhoBERT was pretrained on segmented text; XLM-R's SentencePiece model was not,
   and fragments these tokens. Its −0.024 therefore mixes a model effect with a preprocessing
   penalty that only one of the two models pays. No XLM-R **raw-text** run exists.
2. **The comparison does not isolate parameter count.** PhoBERT-base, PhoBERT-large and XLM-R differ
   in tokenizer, pretraining corpus, vocabulary and depth at once. The 135M → 277M → 368M "trend"
   is three different models, not a scaling curve.
3. **Budgets were matched in epochs, not in tuning.** Every model ran the base recipe's single
   learning rate. `phobert-large`'s best epochs [1, 4, 4, 4, 3] suggest that rate did not suit it;
   under-tuning the larger model is a known way to make it look worse.

**Decision.**

* **The shipping decision stands**: `phobert-base` + pyvi. It won every comparison actually run, and
  it is the cheapest at inference. That is sufficient to ship it and needs none of the general claims.
* **The conclusion is restated** as: *PhoBERT-base performs best among the configurations tested
  under this budget (base recipe, 4 epochs, one learning rate, pyvi input).*
* **Retracted as unsupported:** "encoder capacity has no remaining value", the multilingual-vs-
  monolingual generalization, and "CafeBERT is not worth running". CafeBERT is *not run, for cost*,
  which is a budget decision, not a finding.
* **Tier E is paused, not closed.** If it is reopened, the minimum controls are an XLM-R raw-text
  run (5 seeds, the one missing cell that confounds the existing result), a small learning-rate sweep
  for `phobert-large` on 3 seeds, and tokenizer-specific length profiles, all declared in
  `configs/experiments/` before running.

**Consequences.** Nothing shipped changes. The README and EXPERIMENT_MATRIX stop presenting Tier E
as a closed negative result and describe it as a budget-limited comparison with a known confound.

---

## ADR-020 · 2026-09-27 · Exported artifacts are released through a verified staging step (review R3) · Accepted

**Context.** Review R3 found that the export did not enforce the quality contract it documented.
Static INT8 calibration and the parity check used **raw** text while the service feeds
pyvi-segmented text. Parity covered 64 sentences and label agreement only. `logits_close` was
computed and ignored. Calibration drew on validation, the same set used for acceptance. Files were
written straight into the served directory. And the loader picked `model.quant.onnx` whenever one
existed, so an FP32 re-export could silently keep serving a stale INT8 graph.

**Decision.** `vifeedback serve export` now goes through `inference/release.py`:

1. Build in `models/serve/.staging-<task>-<timestamp>/`, keeping exactly one model file.
2. Calibrate static INT8 on a **stratified train** subset (300 sentences, every class at least 10),
   through the checkpoint's own preprocessing, which is inferred from the checkpoint name or given
   explicitly.
3. Accept on the **full validation set** through the same preprocessing, in padded batches and one
   sentence at a time. FP32 must be logit-close (atol 1e-3). INT8 must stay within **0.005
   macro-F1** of the PyTorch model (the pre-registered budget) with ≥ 99% label agreement.
4. Write `manifest.json`: the served file, its SHA-256, preprocessing, label map, max length,
   acceptance results and software versions. The loader serves exactly that file and refuses a
   checksum mismatch. The service takes its segmenter from the manifest unless `SEGMENTER` is set.
5. Only then replace the served directory, keeping the previous one as `.previous-<task>`. A failed
   check leaves the served artifact untouched and keeps staging for inspection.

**Evidence on the reference machine.** ADR-017 moved all export to Kaggle because `onnx` is
blocked. Pinning `torch.onnx.export(..., dynamo=False)` avoids `onnxscript` and `onnx` entirely, so
the FP32 release ran here. Over 1,583 validation sentences: max |logit diff| **8.2e-5**, label
agreement **100%**, batch-of-one identical to padded batches. INT8 still needs `onnx`, so it stays
on Kaggle.

**A measurement caught on the way.** The release reported macro-F1 **0.8672** for a checkpoint whose
registry row says **0.8634**. Segmentation parity was checked first (runtime pyvi equals the
materialized variant on all 1,583 sentences), and it was not the cause. Evaluation precision was:
fp16, used for every registry row, flips **one** neutral sentence whose top-two margin is 0.0000.
One neutral example is worth 0.004 macro-F1 on this dev set. That bounds how small a single-seed
difference can be and still mean anything, and it is recorded in `configs/experiments/cycle1.yaml`.

**Consequences.** R3 is closed for the export path. The latency benchmark (R10) is still to be
re-run on the released artifact.

---

## ADR-021 · 2026-09-27 · Add H4 (shared encoder) to Cycle 1; pool the finalist seeds · Accepted

**Context.** The compliance audit ([REVIEW_COMPLIANCE.md](archive/REVIEW_COMPLIANCE.md)) found that
Cycle 1 v1 omitted the one item from the review's *recommended starting point* not yet covered:
"test one shared sentiment/topic encoder" (E07, Study D). The review's example Cycle 1 allocation
included it too.

**Decision.** `configs/experiments/cycle1.yaml` becomes version 2, committed before any H4 run:

* **H4:** one PhoBERT-base encoder with a sentiment head and a topic head,
  `L = L_sentiment + λ·L_topic`, λ ∈ {0.3, 1}, 3 seeds each. The heads copy the single-task head
  architecture and learning rate, so sharing the encoder is the only change. The epoch is selected by
  the mean of the two dev macro-F1 scores. The decision rules (helps / negative transfer / no material
  difference) are declared and encoded in `evaluation/decisions.py` before the runs.
* **Budget:** still 30 runs. Finalist seeds move from per-hypothesis allowances into one pool of 4
  (at most 2 finalists × 2 seeds).

**What was known when this was written.** cRT had already missed its advance rule (+0.0028 mean
Δ, 2/3 wins). The logit-adjustment and augmentation runs had finished or were running, but their
decisions had not been computed or inspected. No H1–H3 criterion changes.

**Consequences.** Cycle 1 now covers every item of the review's recommended starting point. The
single-task controls are the P4 registry rows at the same seeds, valid because Cycle 1's stage-1 runs
reproduced them exactly under the current code.

---

## ADR-022 · 2026-09-27 · INT8 does not pass the quality gate; FP32 ONNX is the serving artifact · Accepted

**Context.** Both INT8 releases failed on Kaggle with no manifest. Reproduced on the reference machine
(`onnx` now imports: the host policy had blocked it only while its DLLs were freshly written):
quantizing the `ORT_ENABLE_ALL` graph raises *"Unable to find data type for weight_name … shape
inference failed"*. Fused contrib operators defeat the quantizer's shape inference.

**Fixes.**

* Quantization starts from the plain export after `quant_pre_process`, never from a fused graph.
* The offline FP32 graph is optimized at `ORT_ENABLE_EXTENDED` only. `ENABLE_ALL` adds CPU-specific
  layout transforms, so a graph optimized on one machine need not suit another (review R3). The
  serving session applies the full level itself on the machine that runs it.
* A failed build now writes its manifest with the error before stopping.

**Result, full validation set, non-inferiority margin 0.005 (one-sided 95%):**

| Artifact | Size | Macro-F1 (PyTorch 0.8672) | Neutral F1 | Label agreement | Gate |
|---|---:|---:|---:|---:|---|
| FP32 ONNX | 540 MB | 0.8672 | 0.672 | 100% | ✅ released |
| INT8 dynamic | 136 MB | −0.030 (upper bound 0.054) | **0.584** | 98.7% | ❌ blocked |
| INT8 static (per-tensor MinMax) | 136 MB | 0.342 | — | 54% | ❌ blocked |

**Decision.** Serve FP32 ONNX. INT8 as configured costs the minority class: dynamic quantization
leaves both majority classes unchanged and drops neutral F1 by 0.088. An accuracy-only check
(98.7% agreement) would have shipped it; the macro-F1 gate did not. Static per-tensor calibration
breaks the model outright, as expected for transformer activations with outliers.

**Consequences.** Phase 6's question ("does INT8 add anything on this CPU?") is answered: not at this
margin. Untried, and only worth trying with selection on a held-out subset rather than on the
acceptance set: per-channel weights, excluding sensitive layers, percentile/entropy calibration,
distillation to a smaller FP32 student (E15).

---

## ADR-023 · 2026-09-27 · Commit run artifacts as runs finish; merge other machines' results by command · Accepted

**Context.** The Kaggle output was extracted into the repository and `results/`, holding 19 finished
but **uncommitted** Cycle 1 laptop runs, was lost; it was not recoverable from the Recycle Bin. The
committed part came back from git. The decisions and their numbers survived in the run logs, but the
per-run artifacts (metrics, validation predictions, stage-1 robustness) did not. This is the second
loss of `results/` (see the earlier deletion restored at 5692a1d), and both happened to artifacts that
existed only on disk.

**Decision.**

1. **Regenerate, don't reconstruct.** The 19 runs are re-run with the same code and seeds. Training
   is deterministic (every stage-1 control reproduced the registry exactly), so the regenerated
   numbers are checked against the logged originals instead of being typed back in.
2. **Commit artifacts as each batch of runs finishes**, not when a cycle closes. Holding results
   uncommitted until they were "complete" is what made them losable.
3. **`vifeedback results merge <dir>`** replaces the manual merge: it refuses to run without a local
   registry, migrates the schema, appends only new run ids and never overwrites a run directory.
   `kaggle_results/` is gitignored as the staging folder.
4. `vifeedback study audit-sheet` rebuilds the local audit sheet (which holds corpus text and is
   never committed) from committed, text-free files.

**Cost.** About 2 GPU-hours to regenerate. Recorded in the ledger as regeneration, not new
experiments: the decisions were already taken under the declared rules.

---

## ADR-024 · 2026-09-27 · Cycle 1 closing gate: retrain the finalist to evaluate it on test · Accepted

**Context.** `cycle1.yaml` declares a closing gate: one logged test evaluation per finalist, with a
temperature fitted on validation and applied once to test. H2 (augmentation) is the only supported
intervention, so it is the finalist, against the deployed CE model. The Cycle 1 runs kept no
checkpoints, so evaluating the finalist on test means retraining it.

**Decision.** Retrain H2 at its 5 seeds with `--include-test --robustness` (phase 9, so run ids stay
distinct from the phase-8 decision runs; validation must reproduce them exactly), keeping seed 42's
checkpoint. `vifeedback study closing-gate` then evaluates both seed-42 checkpoints on test: the
official split, the frozen overlap-excluded slice, validation-fitted calibration, and the robustness
suites. The 5-seed test comparison uses G4's CE rows. Each test touch is logged with its reason.

**Budget.** 5 runs beyond the declared 30, recorded in the ledger. The plan assumed checkpoints that
were not kept; later cycles save the finalist checkpoint when a hypothesis advances.

**Guard.** Nothing measured at the gate feeds back into Cycle 1's decisions, which are already
recorded. The gate *reports*; any change it motivates belongs to Cycle 2 and is logged in
EVALUATION_PROTOCOL § 4 as a decision influenced by a test result.

---

## ADR-025 · 2026-09-27 · Cycle 2 declared: track A, a frozen challenge set, and the serving rule · Accepted

**Context.** Cycle 1 closed with the neutral gap located outside the classifier, a deployable
robustness gain (H2), and a test split used 28 times. NEXT_PLAN proposed one specialization; the owner
asked for the plan to be carried out.

**Decision.** `configs/experiments/cycle2.yaml`, committed before any Cycle 2 run:

* **Track A** (encoder vs LLM on the hard cases), the plan's recommendation for an AI-engineer
  profile. Track B (distillation, careful INT8) is deferred, not rejected.
* **A constructed challenge set** is the new confirmation data. Its hash is committed before any model
  is evaluated on it; the official test is for historical comparison only.
* **The serving model is decided by a declared rule** on the challenge set (H6), not by preference.
* **Topic stacking** (H5) with a fixed-C sparse model, so no hidden dev-set tuning enters the stack.
* **The audit decision tree** is frozen before any annotation exists.


---

## ADR-026 · 2026-09-27 · Cycle 2 v2: challenge set frozen; an API reference arm for H7 · Accepted

**Context.** ADR-025 declared the challenge set but not its content. The owner then offered an OpenAI
API key, which makes a stronger, closed LLM available at almost no cost (about USD 0.05 for the
challenge set).

**Decision.** `cycle2.yaml` version 2, committed before any model is evaluated on the challenge set
and before any H7 run:

* **`data/challenge/challenge_v1.csv` is frozen** by SHA-256 (305 rows, 10 categories;
  `data/challenge/README.md`). Every row was checked against UIT-VSFC and the negation probe after
  aggressive normalization; none matches.
* **H7 gains an API arm**: `gpt-4o-mini-2024-07-18`, temperature 0, label likelihood from the first
  token's top-20 log-probabilities. It is a reference point, not a serving candidate. The two LLM
  comparisons are Holm-adjusted.
* **Prompt development uses the open pilot model only**, so no prompt is tuned against the API.
* **Data egress rule.** Only data the project may send to a third party goes to the API: the
  constructed challenge set, yes; UIT-VSFC text, only after the owner confirms the licence allows it.
* **Two descriptive measurements**, deciding nothing: a paired bootstrap for the facility claim
  (TF-IDF vs PhoBERT), and model confidence on out-of-scope input.

**Unchanged.** Every v1 rule. H5 was running when v2 was written; its rule is untouched.

**Key handling.** The key is read from `OPENAI_API_KEY` (or a git-ignored `.env`), never logged,
never written to a result file, never committed.

---

## ADR-027 · 2026-09-27 · The service runs the H2-augmented model (Cycle 2 H6) · Accepted

**Context.** H2 (diacritic/teencode augmentation, p = 0.3) was supported on validation over 5 seeds
and confirmed at the Cycle 1 closing gate. Whether to serve it was left to a rule declared in
`cycle2.yaml` (H6) and evaluated on the frozen challenge set, whose typed noise was written by hand
rather than produced by the augmentation code.

**Evidence** (`results/studies/challenge/summary.json`, seed-42 checkpoints of both models):

| Rows | CE | Augmented | Difference |
|---|---:|---:|---|
| Typed noise (unaccented + teencode, 90) | 0.611 | 0.800 | **+0.189 [+0.067, +0.311]** |
| All other scored rows (195) | 0.867 | 0.867 | +0.000 (allowed: ≥ −0.02) |
| `unaccented_typed` (50) | 0.320 | 0.740 | +0.42 |
| `teencode_typed` (40) | 0.975 | 0.875 | **−0.10** |
| `objective_neutral` (30) | 0.767 | 0.633 | **−0.13** |
| Neutral F1 (285 scored rows) | 0.713 | 0.745 | +0.03 |

**Decision.** The declared rule is met, so the service switches. The augmented model was released
through the unchanged gate (FP32 ONNX, full validation: macro-F1 0.8644 in both PyTorch and ONNX,
max logit difference 1.7e-5; manifest `results/studies/export/laptop_fp32_augmented_manifest.json`).
The CE release is kept at `models/serve/.previous-sentiment` for rollback.

**What the rule did not see.** Pooling the two typed categories hid a regression inside one of them,
and "other rows" hid another. Hand-typed teencode, which the augmentation was meant to cover, got
*worse* (4 rows, e.g. *ko hỉu j hết* → neutral). Short factual sentences drift to `negative` with
confidence 0.5–0.7. Both are recorded here as known behaviour of the served model, and the next
declared rule will require non-inferiority per category, not only in pools.

**Test influence.** This is the first decision a test result influenced (EVALUATION_PROTOCOL § 4):
the closing-gate test numbers were known when H6 was declared.

---

## ADR-028 · 2026-09-27 · Correction: constructed, not typed; two drops are not yet regressions · Accepted

**What was claimed.** The Cycle 2 write-up (ADR-027, STATUS, RESEARCH_REPORT, README) called the
challenge set's noisy rows "hand-typed" and "real typing", and listed two per-category drops of the
served model as regressions.

**What is true.** The rows were *constructed* to imitate typing. None was typed by a real user on a
device. They are independent of the augmentation code, which is all H6's rule needed, but they are not
a sample of real typing. And at one seed the two drops are not significant:

| Category | CE right only | Augmented right only | Exact McNemar p |
|---|---:|---:|---:|
| `teencode_typed` (40) | 4 | 0 | 0.125 |
| `objective_neutral` (30) | 5 | 1 | 0.219 |
| `unaccented_typed` (50), for contrast | 5 | 26 | < 0.001 |

**Consequences.** H6's decision stands: its declared rule (pooled typed rows, paired CI above 0,
p = 0.003) is met, and the unaccented gain is significant alone. The two drops become hypotheses,
checked at five seeds before anything is built to fix them (NEXT_PLAN v3, step 0). A likely cause is
already visible: the augmentation's teencode map has 14 entries. Claims about real user typing wait for
a human-typed challenge set (v2).

---

## ADR-029 · 2026-09-27 · Cycle 3: serve lowercased input; run S2a now with a real-typing lexicon · Accepted

**Context.** V1 (five seeds of CE and augmented) and the external invariance tests declared in
`cycle3.yaml` v2 gave four results:

| Result | Evidence |
|---|---|
| The teencode drop at seed 42 was noise | 3/5 seeds, pooled exact McNemar p = 0.092 |
| Short factual sentences: suggestive, not confirmed | 0.747 → 0.660, 3/5 seeds, p = 0.019 |
| Contrast sentences (*nhưng*): a confirmed drop, found by the rule, not pre-specified | 0.875 → 0.800, 4/5 seeds, p = 0.004, survives Holm over 9 categories |
| Missing diacritics: the gain holds | 0.328 → 0.656, better in 5/5 seeds |
| Capitalized input changes labels | served model: 17 of 1,583 validation labels (1.07%) flip when the first letter is capitalized; UIT-VSFC has no uppercase letter in any split |
| Real informal typing changes labels, and augmentation does not help | ViLexNorm: 17.3% of labels flip between a real comment and its human normalization, for CE and augmented alike (p = 0.93) |

**Decisions.**

1. **The service lowercases input** (NFC, whitespace, lowercase, then segmentation:
   `normalize.model_text`), as the declared case rule required. Capitalization flips drop to 0 for all
   ten checkpoints, and no validation or challenge prediction changes, so the released artifact is
   unchanged.
2. **S2a runs now**, not after challenge v2. Its lexicon comes from ViLexNorm's human normalizations
   (train split only) instead of 14 hand-picked entries. `cycle3.yaml` v3 records the recipe, the
   lexicon hash and a development gate, and leaves the v2 confirmation rule unchanged. This decision
   follows the ViLexNorm result it responds to; that is why it is a new version and not an edit.
3. **The model card changes.** Teencode is no longer listed as a regression (not confirmed). Contrast
   sentences are listed (confirmed, exploratory). Short factual sentences stay listed as unconfirmed.

**Licence.** ViLexNorm is CC BY-NC-SA 4.0. The lexicon is built locally, and a model trained with it
inherits the non-commercial condition. If that model is ever served, the model card must say so. A
commercial-safe lexicon would come from U3: candidates the owner checks.

---

## ADR-030 · 2026-09-28 · Real student text replaces the human-typed challenge set as Cycle 3's confirmation data · Accepted

**Context.** Cycle 3's serving rules were to be confirmed on challenge v2, sentences typed by several
people. The owner has neither the time nor the people for it now. Meanwhile NEU-ESC became available:
6,613 test posts written by Vietnamese students on university forums, labelled by the dataset's
annotators, with no overlap with UIT-VSFC.

**Decision.** `cycle3.yaml` v5 moves the confirmation of S2b (restoration), S2b′ (CE + restoration)
and S3 (out-of-scope score) to cells built from NEU-ESC test. Each cell is either real text with real
labels (all posts; contrast sentences; short neutral posts; off-topic posts), or real text with real
labels and a known perturbation (all posts with diacritics stripped). A cell needs at least 100 rows to
decide anything. The owner also allowed NEU-ESC text to be sent to the OpenAI API, so H7 is repeated
on it with gpt-4o-mini, zero-shot. UIT-VSFC text is still not sent.

**What this costs.** NEU-ESC is forum writing, not course surveys, and its label policy differs
(69% neutral). A rule passing here shows the change holds on real student text; it does not show it on
the service's exact input. The unaccented cell's noise is synthetic, though the text and labels are
real. And the overall NEU-ESC scores (CE 0.463, augmented 0.434) were seen before this version was
written; the cells were not, and they are computed only after this commit. Challenge v2, if it is ever
written, is reported next to these results, never instead of them.

---

## ADR-031 · 2026-09-28 · The service restores diacritics and reports an out-of-scope score · Accepted

**Context.** `cycle3.yaml` v5 confirmed two Cycle 3 changes on NEU-ESC, real student text with human
labels (ADR-030), and rejected a third:

| Rule | Evidence on NEU-ESC test | Outcome |
|---|---|---|
| S2b: served model + diacritic restorer | posts with diacritics stripped: macro-F1 0.270 → 0.374 (+0.104 [+0.091, +0.118]); posts as written: −0.0006 | passed |
| S2b′: CE + restorer instead of augmented + restorer | contrast posts 0.440 vs 0.437 (p = 0.68); the challenge v1 advantage did not replicate | not passed |
| S3: Mahalanobis out-of-scope score | off-topic posts vs validation, AUROC 0.977 (max-probability 0.920) | passed |

**Decision.** The served model stays the H2-augmented checkpoint (ADR-027). Two parts are added to
the release, each declared in its manifest with a SHA-256, and a mismatch makes `/readyz` 503:

1. **A diacritic restorer** before segmentation (`serve add-restorer`), built from UIT-VSFC train only.
   It rewrites only essentially unaccented input. On the served graph it changes 0 of 1,583
   validation labels and lifts stripped validation from 0.685 to 0.857 macro-F1.
2. **An out-of-scope score** (`serve export --with-features`, `serve add-ood`): the negative
   Mahalanobis distance of the sentence feature to the nearest class mean. The API returns `in_scope`
   (threshold: 95% of validation kept) and `scope_score` next to the label. The service never
   refuses an input; the caller decides. Checked on the served graph itself before it was attached:
   AUROC 0.977 on the 563 NEU-ESC off-topic posts, 85% of them flagged.

The release is reproducible with `make export` (export with a `features` output, logit parity
1.7e-05 and 100% label agreement on validation; then `serve add-restorer`; then `serve add-ood`).
The previous artifact is kept as the rollback copy; the CE model can be re-exported from its
checkpoint.

**Not decided here.** Whether the service should switch to CE: its other NEU-ESC checks favoured it
(overall +0.028, unaccented +0.036), but the declared rule required the contrast advantage and did not
get it. A human-typed challenge set, if one is written, is where that question can be reopened.

---

## ADR-032 · 2026-09-28 · The out-of-scope score measures resemblance to UIT-VSFC, not topic · Accepted

**Context.** NEXT_PLAN v5 A2 measured the served `in_scope` flag on NEU-ESC validation, topic by
topic (`study served-neu-esc`, rule fixed in v5 before the run). Another university's posts about
academics and services are in scope; spam, news, jobs and club events are not.

| NEU-ESC validation | Posts | Flagged out of scope |
|---|---:|---:|
| Academic | 1,053 | 75.5% |
| Service | 237 | 64.6% |
| All in-scope topics | 3,026 | 77.1% |
| Off-topic topics | 279 | 84.2% |

Within NEU-ESC, the score separates in-scope from off-topic posts with **AUROC 0.573**: barely above
chance. S3's confirmation (AUROC 0.977, `cycle3.yaml` v5, ADR-031) compared NEU-ESC off-topic posts
with UIT-VSFC validation, so it mixed a change of institution with a change of topic, and the score
tracked the institution. This is the background-versus-semantic-shift distinction [Arora et al.,
2021]. The v5 design did not control for it; the confirmation holds only for what it measured.

**Decision.**
1. `in_scope` stays in the API, described as what it is: **resemblance to the training text** (course
   surveys from one university). `in_scope: false` on another institution's feedback is expected and
   is not evidence of an off-topic input. The model card and the README say so.
2. The A2 rule triggered: B4 refits the score with in-scope NEU-ESC training features
   (`cycle4.yaml` H8 rule 5). A topic-aware detector is a separate question (NEXT_PLAN v5 B4′). Any
   future out-of-scope claim must be measured *within* one source, in-scope against off-topic.

**Also measured (A1, A3).**
- The served pipeline, raw text in, has p95 26.9 ms (median of three sessions, range 23.4–39.5, the
  39.5 from an unsteady session). ADR-031's additions cost +2.2 ms on accented text and save time on
  unaccented text (S3 − S2 = −4.2 ms): the target of ≤ 30 ms and ≤ 5 ms is met.
- On in-scope NEU-ESC validation posts, the served model calls 67% of gold-neutral posts polar,
  mostly negative (1,172 of 2,054). Its "a request is negative" policy meets forum questions.

---

## ADR-033 · 2026-09-28 · H8 not passed: two heads close much of the NEU-ESC gap, but not by the declared rule · Accepted

**Context.** `cycle4.yaml` (v1 before any run, v2 reporting-only) asked whether in-scope NEU-ESC
training data closes the gap on another institution's posts.

| Recipe, seed 42 (validation) | UIT-VSFC | NEU-ESC (served head) | Eligible |
|---|---:|---:|---|
| Control (the served recipe) | 0.8644 | 0.4590 | — |
| mixed: one head, both datasets | 0.8318 | 0.7498 | no (−0.033) |
| sequential: served model, then NEU-ESC | 0.5958 | 0.7413 | no (−0.269) |
| **two heads**: shared encoder, one head per dataset | 0.8720 | 0.5326 (NEU-ESC head 0.7612) | **yes → confirmed** |

Confirmation, two heads against the control, five seeds each:
1. NEU-ESC test (6,050 in-scope posts): +0.111 [+0.102, +0.119], higher in 5 of 5 seeds (0.554 vs
   0.444). **Passed.**
2. UIT-VSFC validation macro-F1: −0.0045 against a 0.005 limit. **Passed.**
3. UIT-VSFC neutral F1: −0.010 against a 0.02 limit. **Passed.**
4. UIT-VSFC validation with diacritics stripped, through the restorer: −0.0107 against a 0.01 limit.
   **Not passed.**
5. The out-of-scope score refitted on UIT-VSFC + NEU-ESC training features: AUROC 0.886 (< 0.90), and
   it flags 31% of in-scope NEU-ESC validation posts (> 10%). **Not passed.** Within NEU-ESC it
   separates topics at 0.540.

The first confirmation call stopped in rule 5 on a swapped return order, after computing test
predictions it never showed. Both calls are in `neu_esc_test_uses.log`.

**Decision.** H8 is not passed; the served model stays. B3 (backbone) is not triggered; there is no
B5 release.

**What the cycle established.**
- **Label policy is the conflict.** One head trained on both datasets' labels loses UIT-VSFC
  (mixed −0.033; sequential −0.269, which forgets it), because the two corpora mean different
  things by `neutral`. Separate heads remove the conflict: the shared encoder improves UIT-VSFC at
  seed 42 and lifts the UIT-VSFC head on NEU-ESC by 0.11.
- **The NEU-ESC head reaches 0.76 on validation**, near the NEU-ESC authors' in-domain 77.7 on four
  classes. A service that labels other institutions' text by their annotators' policy is within
  reach, but it is owner decision 2.
- **Rule 5 tied a sentiment question to a scope detector that ADR-032 had just shown measures
  resemblance, not topic.** Refitting on more data did not make it a topic detector. Any release that
  widens the service to other institutions needs a topic-aware scope detector first (B4′).
- **Rule 4 is a real, small cost.** It measures unaccented UIT-VSFC text after the restorer and
  missed by 0.0007.

**What would change it.** NEU-ESC test has now served H8, so a new claim needs new confirmation data:
another institution's labelled text, or a fresh human-labelled sample. Until then, the two-heads
model is an exploratory result and is not published or served.

---

## ADR-034 · 2026-09-28 · `in_scope` comes from a topic-aware detector, not the Mahalanobis score · Accepted

**Context.** ADR-032 showed that the served `in_scope` (Mahalanobis on the sentence feature) tracks
the institution, not the topic: within NEU-ESC it separates in-scope from off-topic posts at AUROC
0.573 on validation and 0.595 on test. `cycle4.yaml` v3 declared B4′ before any fit: a detector
trained on labelled topics.

**Result** (`study b4prime`, `results/studies/cycle4/b4prime/decision.json`).
- **Selection.** Chosen on NEU-ESC validation from two candidates: TF-IDF logistic regression
  (AUROC 0.921, C = 1) over logistic regression on the served feature (0.869).
- **Confirmation on NEU-ESC test,** logged. All four declared conditions passed:
  1. within-source AUROC **0.922**;
  2. in-scope posts flagged **7.3%**, where the Mahalanobis score flagged about 77%;
  3. off-topic posts caught **72.3%**;
  4. UIT-VSFC validation flagged **0.2%**.
- **Generated off-topic sentences** (U4): AUROC 0.918.

Topic is a property of words. The encoder's feature was fine-tuned for sentiment, which is why it
separates topics less well.

**Decision.** `serve add-scope` attaches the detector to the release:
- `scope.npz` holds 70,377 terms, idf, coefficients and a threshold of −0.177, with its SHA-256 in
  the manifest.
- The service scores it with a numpy re-implementation (`serving/scope_tfidf.py`). It equals
  scikit-learn on all 12,032 evaluation texts (max |diff| 8e-15, same flags), so the runtime image
  still needs no scikit-learn.
- The Mahalanobis entry leaves the manifest (kept in `manifest.before-scope.json`). The graph keeps
  its `features` output, unused.
- The API's shape is unchanged. `scope_score` is now the detector's decision value (higher = in
  scope), and `in_scope` is that value at or above the threshold.

**Limits.**
- It judges topic from words: an unusual or very short course comment can be flagged, and off-topic
  text that uses course vocabulary can pass.
- Its off-topic training examples are four NEU-ESC topics; other kinds of off-topic text are covered
  only as far as U4 suggests.
- The model already on the Hugging Face Hub still carries the Mahalanobis file and the ADR-032
  caveat until the owner approves a new upload.

**Latency.** Three sessions with the detector served (`cycle4_scope_session{1,2,3}.json`) ran on a
busier machine than the morning's sessions: every rung was slower, including code that did not
change. S0, the pipeline before ADR-031, had p95 35–36 ms, against 22–25 ms in the morning. Measured
within each session:
- the served pipeline adds **+0.5 to +0.8 ms** over S0, against +1.3 to +3.1 ms with the Mahalanobis
  score;
- on unaccented input it is 6–15 ms faster than S0, because restored text tokenizes into fewer
  pieces.

The detector therefore costs less than the score it replaced. The absolute p95 target (≤ 30 ms) was
last shown at 26.9 ms with the heavier Mahalanobis step, and is to be re-measured on an idle machine.

---

## ADR-035 · 2026-09-28 · A larger acceptance set for careful INT8 (S5′), declared before it is run · Accepted

**Context.** The careful INT8 recipe (pc-head-last2, 178.5 MB) failed its non-inferiority test only
on power. The drop was 0.0004, but the upper bound was 0.0095 against a 0.005 margin, on 1,583
validation sentences. NEXT_PLAN v5 E1 asked, before any new rule was declared, whether a larger
labelled set (UIT-VSFC validation + NEU-ESC validation, 4,888) could show the margin
(`study int8-power`).

**Method.** INT8's disagreements were estimated on held-out data as P(INT8 label | source, gold,
FP32 label): the INT8 study's train subset, and 3,000 NEU-ESC train posts the model never saw.
INT8 predictions were then simulated on the acceptance set, where only FP32 was run, and the paired
bootstrap bound was computed, 40 simulations each.

**Findings.**
- **The simulation is calibrated.** On UIT-VSFC validation, where INT8 was really run, the observed
  bound (0.0095) falls at the 82nd percentile of the simulated bounds. That is inside the bulk, on
  the optimistic side.
- **The larger set has the power.**

  | Acceptance set | Median simulated bound | Share of simulations below 0.005 |
  |---|---:|---:|
  | UIT-VSFC validation alone | −0.0001 | 75% |
  | UIT-VSFC + NEU-ESC validation | −0.0008 | 92% |

- **INT8 is much less faithful off-domain.** It changes 12.3% of labels on NEU-ESC train posts,
  against 1.35% on UIT-VSFC train. The changes fall on low-confidence posts, which is why the
  simulated macro-F1 change stays near zero. The acceptance rule must therefore check the NEU-ESC
  part on its own, not only the pooled number.

An earlier draft of this ADR read the median against the observed bound, ignored the spread, and
called the simulation uncalibrated. The calibration check in the code says otherwise, and this
version follows it.

**Decision.** Declare S5′ in `cycle4.yaml` v4 before INT8 is run on NEU-ESC validation, with the
pooled set and per-source guards. UIT-VSFC validation's INT8 result is already known from Cycle 3
and is disclosed as such. If S5′ fails, S7 goes to distillation (track B).

---

## ADR-036 · 2026-09-28 · INT8 is not released: S5′ passed, but the release gate's fidelity rule holds it back · Accepted

**Context.** S5′ (ADR-035, `cycle4.yaml` v4) passed on UIT-VSFC + NEU-ESC validation:
- pooled upper bound of the macro-F1 drop 0.0006;
- NEU-ESC drop −0.0044;
- neutral F1 loss −0.0052;
- 178.5 MB.

The declared consequence was a release candidate through the release gate. Two practical notes on
the attempt:
- The Cycle 3 graph was released as it is (`serve export --prebuilt`), because Windows Application
  Control now blocks the `onnx` package's DLL on this machine and export cannot run.
- Releasing exactly the file S5′ tested is also closer to the declaration.

**Result** (`results/studies/int8_power/release_gate_attempt.json`). The gate blocked the release.
Its standing rule that an INT8 graph must agree with PyTorch on at least 99% of labels failed:
**91.4%** on the pooled set. On UIT-VSFC alone agreement is 99.2%, but there the non-inferiority
bound fails (0.0095), so the graph cannot pass the gate either way.

The gate also recorded that the INT8 graph's output depends on the batch. Scored alone and inside a
batch of 32, logits differ by up to 0.93 and one label in 32 changes. Dynamic quantization scales
activations per batch, so the label of a sentence would depend on what else arrived in the same
request.

**Decision.**
- The service stays on FP32 (unchanged: the release gate swaps nothing on failure).
- The 99% agreement rule stands. Lowering it after seeing the result would move the goalposts, and
  the batch dependence shows why fidelity matters beyond macro-F1.
- S7 (≤ 200 MB) goes to distillation (track B, E2): a 6-layer student has no per-batch activation
  scales and is judged on both corpora.

---

## ADR-037 · 2026-09-30 · Cycle 5 is declared in the owner's order; S7 is attempted with a 6-layer student stored in FP16 · Accepted

**Context.** Cycle 4 closed with three open lines:
- real typing flips 17% of labels;
- the serving artifact is 540 MB, and INT8 cannot be released (ADR-036);
- other institutions' text needs new labelled data.

The owner chose the order: (b) real typing, then (a) a smaller model, then (c) other institutions
(NEXT_PLAN v6).

**Decision.**
1. **H10** (`cycle5.yaml` v1, before any run): consistency training on ViLexNorm pairs, a real
   comment and its human normalization, with a KL term that asks both forms for the same prediction.
   - Two recipes: one-sided, where the normalized form is a fixed target, and R-Drop's symmetric KL.
   - Selection on a declared ViLexNorm development split (837 of the 8,372 training pairs, by a
     fixed permutation).
   - Confirmation on ViLexNorm test, whose two earlier uses are disclosed; every Cycle 5 use is
     logged.
   - Four non-inferiority guards on labelled data: UIT-VSFC, its neutral class, stripped text, and
     NEU-ESC validation.
2. **H11** (`cycle5.yaml` v2, before any H11 run and before H10 decided): a 6-layer student
   distilled from a 5-seed ensemble teacher, with its weights stored in FP16 and computed in FP32.
   - **Teacher.** Fixed by a written condition: H10's recipe if H10 passes, the served recipe
     otherwise. It does not depend on H10's numbers beyond its pass or fail.
   - **Rule.** ≤ 200 MB; a pooled drop bound ≤ 0.01 against the teacher; per-source guards;
     parity with the PyTorch student and no batch dependence.
3. **Why FP16 storage and not INT8 or vocabulary trimming.**
   - From the served config, neither 6 layers (370 MB) nor FP16 storage (270 MB) alone reaches
     200 MB; together they give 185 MB.
   - FP16 storage rounds weights once and computes in FP32, so nothing is scaled per batch: the
     failure that stopped INT8.
   - Vocabulary trimming would map unseen tokens to `<unk>` on exactly the informal text H10 targets.
4. **Export.** `onnx_export.export_fp16_storage` casts parametrized FP16 weights inside the traced
   graph, with the legacy exporter and without the `onnx` package (still blocked here).
   - Constant folding is off, and the graph skips the offline optimizer, so the casts survive into
     the file.
   - On a tiny RoBERTa the file is half the size, the logits equal the FP16-rounded model within
     1e-4, and a row scores the same alone and in a padded batch.

**Consequences.**
- Both H10 and a released H11 student learn from ViLexNorm text (CC BY-NC-SA 4.0). Their weights
  would carry that licence, which the owner decides before any upload (HUONG_DAN_THU_CONG § 4).
- The first H10 run shared the 4 GB GPU with another project's job and was stopped. The run guard
  now also waits for no other GPU compute process, in line with the owner's one-heavy-job rule.

---

## ADR-038 · 2026-09-30 · H10 not passed: consistency training made the model invariant by labelling informal text negative · Accepted

**Context.** `cycle5.yaml` v1 H10, run on 2026-09-30 by the declared code
(`results/studies/cycle5/h10/`).

**Selection (seed 42, ViLexNorm dev, 837 pairs).**

| | UIT-VSFC validation | Dev flip rate | Eligible |
|---|---:|---:|---|
| Control (served recipe) | 0.8644 | 0.135 | — |
| One-sided KL | 0.8541 (−0.0103) | 0.007 | no (limit −0.01) |
| Symmetric KL (R-Drop) | 0.8689 (+0.0045) | 0.002 | **yes → confirmed** |

**Confirmation** (symmetric, five seeds each side; ViLexNorm test, 1,045 pairs, use logged):

| Rule | Result | |
|---|---|---|
| (1) flip rate lower | 16.6% → 0.3%, −0.164 [−0.180, −0.149] | pass |
| (2) UIT-VSFC validation macro-F1 | −0.0016 (limit −0.005) | pass |
| (3) neutral F1 | −0.0029 (limit −0.02) | pass |
| (4) stripped text through the restorer | −0.0065 (limit −0.01) | pass |
| (5) NEU-ESC validation macro-F1 | **−0.0866** (limit −0.01) | **fail** |

**What happened.** The invariance is degenerate:
- **ViLexNorm.** The consistency-trained models label 99.4–99.5% of the test comments *negative*,
  in both forms. The control labels 56–83% negative, with the rest split between neutral and
  positive. A model that gives informal social-media text one label cannot flip.
- **NEU-ESC.** The same shift reaches real student posts. Predicted negative rises from 58% to 76%
  of NEU-ESC validation. Neutral F1 falls in every seed (seed 42: 0.47 → 0.30), because the
  informal register now pushes predictions towards negative.
- **UIT-VSFC** is untouched: its clean, labelled text anchors the cross-entropy term.

The consistency term had one cheap solution: collapse the unlabeled, off-domain pairs to a single
class. Nothing in it forbids that. UDA avoids it with confidence masking, sharpening and in-domain
unlabeled data; H10 had none of the three. Guard (5) was declared for exactly this possibility
(NEXT_PLAN v6 § 2) and caught it.

**Decision.**
- H10 is not passed, and nothing is released.
- Under `cycle5.yaml` v2, H11's teacher is the served recipe.
- ViLexNorm test has now served three rules (the Cycle 3 measurement, S2a, H10), so any new
  real-typing claim needs new confirmation data.

**What a next attempt would need** (not declared):
- a flip metric that a constant prediction cannot game, for example agreement with the control's
  label on the normalized form, reported next to each form's label distribution;
- a guard on the predicted class distribution of the unlabeled text;
- in-domain unlabeled pairs (student posts and their normalizations), or UDA's confidence mask.

---

## ADR-039 · 2026-09-30 · H11 passed: a 6-layer student stored in FP16 matches the served model at 185 MB · Accepted

**Context.** `cycle5.yaml` v2 H11, run by the declared code (`results/studies/cycle5/h11/`). H10 did
not pass (ADR-038), so by the written rule the teacher is the served recipe's 5-seed ensemble.

**Selection (seed 42).**

| Student | UIT-VSFC validation | NEU-ESC validation (all) | Mean |
|---|---:|---:|---:|
| teacher-alternate (layers 1, 3, …, 11 of the same-seed teacher) | 0.8614 | 0.4160 | 0.6387 |
| **pretrained-first6** (PhoBERT's first six layers) | **0.8730** | **0.4360** | **0.6545** |

**Confirmation** (pretrained-first6, five seeds against the teacher's five seeds; validation only):

| Rule | Result | Limit |
|---|---|---|
| (1) FP16-storage graph | **185.1 MB** | ≤ 200 MB |
| (2) pooled UIT-VSFC + NEU-ESC validation (4,888), seed-averaged drop | 0.0024, one-sided 95% bound **0.0078** | ≤ 0.01 |
| (3a) UIT-VSFC validation macro-F1 drop | 0.00001 | ≤ 0.01 |
| (3b) neutral F1 drop | −0.0045 (the student is better) | ≤ 0.03 |
| (3c) stripped text through the restorer, drop | 0.0032 | ≤ 0.015 |
| (4) graph vs the FP16-rounded PyTorch student | max logit diff 1.1e-5, 100% of 1,583 labels, batch vs single 0.0 | ≤ 1e-4, 100%, ≤ 1e-4 |

All hold: **H11 passed.**

5-seed means: UIT-VSFC 0.8685 for both; NEU-ESC in scope 0.4382 against 0.4367. The student has
92.5 M parameters against 135 M and 6 layers against 12.

**Why pretrained-first6 won.** Copying alternate fine-tuned layers keeps a representation built for
12 layers and cut in half. Starting from PhoBERT's first six pretrained layers lets distillation
shape a 6-layer network directly. With a 5-seed ensemble teacher and 28,648 unlabeled transfer
texts, it matched the teacher. Its seeds also vary less on NEU-ESC than the teacher's single seeds
(0.416–0.449 against 0.384–0.469).

**Decision.**
1. **S7 (≤ 200 MB) is met by a model that passed its rule.** The student becomes the release
   candidate. It goes through the release gate as an FP16-storage graph, with the restorer and the
   scope detector re-accepted.
2. **Serving.** As declared, it replaces the served model only after three latency sessions on an
   idle machine show served p95 ≤ 30 ms. Until then the served model is unchanged.
3. **Hub.** Its transfer set holds ViLexNorm text (CC BY-NC-SA 4.0). Uploading it needs the owner's
   licence decision and approval (HUONG_DAN_THU_CONG § 4–5).
4. **Test.** UIT-VSFC test is evaluated once for the student, for its card and S1, after a
   declaration (`cycle5.yaml` v3). No selection uses it.

---

## ADR-040 · 2026-09-30 · The 6-layer student is served; on test it is 0.013 macro-F1 below the 12-layer model · Accepted

**Context.** H11 passed on validation (ADR-039). `cycle5.yaml` v3, declared before any of the
numbers below, set how the candidate replaces the served model: three latency sessions, with the
idle-machine condition waived by the owner. It also declared a closing gate on UIT-VSFC test, run
once and reported without deciding anything.

**Release** (`serve export --quantize fp16-storage`, into `models/candidate/sentiment`).
- **Graph.** 185.1 MB. Against the PyTorch student with the same FP16-rounded weights: max logit
  difference 1.09e-5, 100% of 1,583 validation labels.
- **Restorer.** Re-accepted: 0 validation labels changed, stripped validation 0.663 → 0.851.
- **Scope detector.** Re-attached (the same file).
- **API.** All 12 real-model API tests pass.

**Latency** (S1, raw text through the served pipeline, laptop as it was):

| | Session 1 | Session 2 | Session 3 | Median |
|---|---:|---:|---:|---:|
| Student | 11.6 ms | 12.9 ms | no steady pass (11.7, 11.5) | **12.3 ms** |
| 12-layer, same sessions | 22.0 ms | 24.8 ms | 22.1 ms | 22.1 ms |

The limit is 30 ms. Every pass of every session is below it, and the student is 1.8x faster.

**Decision (as declared).** The student replaces the served model in `models/serve/sentiment`. The
12-layer release is kept in `models/serve/.previous-sentiment`.

**Closing gate** (test, once, logged; `results/studies/cycle5/h11/closing_gate/`).

| Test macro-F1 | 42 | 1337 | 2024 | 7 | 31337 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Student | 0.8175 | 0.8192 | 0.8080 | 0.8253 | 0.8138 | 0.8168 |
| 12-layer (Cycle 1 closing gate) | 0.8371 | 0.8298 | 0.8179 | 0.8309 | 0.8325 | 0.8296 |

- **The gap.** Student minus teacher: **−0.0129 [−0.0204, −0.0054]**, lower in 5 of 5 seeds.
- **Seed 42.** Neutral F1 0.545 against 0.592; no diacritics 0.609 against 0.635; temperature 1.49.
- **On validation** the two were equal (5-seed mean 0.8685 each).

**Reading.**
- Validation holds 73 neutral sentences. Selecting epochs on it and comparing on it could not
  resolve a gap of this size. The pooled rule was dominated by NEU-ESC, where the student is as
  good as its teacher.
- The UIT-VSFC drop on test (0.013) is larger than the 0.01 margin the H11 rule applied on
  validation. By declaration the test figure decides nothing, and it is not used to reverse the
  release: that would be selection on test.
- It is stated wherever the student is described: the card (Limitations), the README and STATUS.
- S1's minimum (≥ 0.80) is still met at 0.817.
- **S4's minimum (neutral F1 ≥ 0.55) is missed by the student:** 0.545 at seed 42, against 0.576
  for the 12-layer model. The project's focus is the neutral class, so the served model is the
  owner's call (decision 9).

**Consequences.**
- **S7 is met** by the served artifact: 185 MB against 540 MB.
- **Hub.** The 12-layer model stays on the Hub
  (`Datk4/vifeedback-sentiment-phobert`, CC BY-NC 4.0). The student's bundle is built for its own
  repository (`Datk4/vifeedback-sentiment-phobert-6l`) with CC BY-NC-SA 4.0 (ViLexNorm in its
  transfer set). Uploading it needs the owner's approval.
- **Going back.** Returning to the 12-layer model is one folder swap, and the owner may choose it
  where accuracy matters more than size and speed (HUONG_DAN_THU_CONG § 10).
- **Lesson for size and speed rules.** Non-inferiority on 1,583 validation sentences is too weak a
  check. A future rule needs a larger acceptance set, for example a held-out part of train kept out
  of both training and selection, next to NEU-ESC.

---

## ADR-041 · 2026-09-30 · The service reports calibrated confidence (temperature scaling) · Accepted

**Context.** Cycle 0 found the model overconfident (T ≈ 1.5 in four independent fits). Both closing
gates confirmed on test that a temperature fitted on validation transfers:
- the 12-layer model: NLL 0.262 → 0.207, ECE 0.043 → 0.015 (Cycle 1);
- the student: NLL 0.251 → 0.208, ECE 0.043 → 0.016 (ADR-040).

The service still reported the raw softmax, so `confidence` overstated how often the label is right.
Temperature scaling was recommended as worth shipping and never shipped.

**Decision.**
- `serve add-temperature` fits T on UIT-VSFC validation logits of the released graph itself and
  records it in the manifest. The service divides the logits by T before the softmax.
- Labels cannot change: dividing by one positive number keeps every argmax, and the command refuses
  a release where any validation label moved.
- Both releases carry their temperature:
  - the served student: T = 1.491, validation ECE 0.032 → 0.013;
  - the 12-layer release kept for a swap back: T = 1.551, validation ECE 0.034 → 0.017.

**Consequences.**
- `confidence` and `probabilities` in the API are calibrated. For example, a short factual sentence
  the model calls negative now reports 0.55 rather than a near-certain value.
- The logits the ONNX graph outputs are unchanged. Users of the published files divide by the
  card's T themselves, as the card already says.

---

## ADR-042 · 2026-10-01 · H10b passed: anchoring consistency on a frozen teacher cuts real-typing flips without collapse · Accepted

**Context.** H10 failed because its own-prediction consistency let the model call all informal
text negative (ADR-038). `cycle5.yaml` v4, declared before any H10b run, made three changes:
- **the target** is the frozen 5-seed teacher's probabilities on the normalized form;
- **the metrics** are agreement with the teacher's label and the label-distribution distance
  (`label_tv`), which a constant prediction cannot game;
- **the confirmation data** are 1,500 fresh ViLexNorm training pairs that no model in the
  comparison has seen.

**Selection (seed 42, dev 837 pairs).**

| | UIT-VSFC | Agreement | Flips | label_tv |
|---|---:|---:|---:|---:|
| Control | 0.8644 | 0.828 | 0.135 | 0.031 |
| anchored_orig | 0.8617 | **0.8626** | 0.109 | 0.019 |
| anchored_both | 0.8709 | **0.8626** | 0.105 | 0.008 |

Both recipes were eligible and tied on dev agreement (the declared criterion). The declaration had
no tie-break, and the selection code kept the first recipe in declared order, `anchored_orig`. That
is recorded here as a gap in the declaration; `anchored_both` would also have been a legitimate
choice.

**Confirmation** (anchored_orig, five seeds each side, 1,500 confirmation pairs):

| Rule | Result | |
|---|---|---|
| (1) agreement with the teacher's label | +0.061 [+0.048, +0.075] | pass |
| (2) flip rate | −0.061 [−0.073, −0.049] (per seed: 11.3–13.0% against 13.5–20.7%) | pass |
| (3) label_tv, 5-seed mean | 0.013 (limit 0.10; the control's is 0.108) | pass |
| (4) UIT-VSFC validation | +0.0014 | pass |
| (5) neutral F1 | +0.003 | pass |
| (6) stripped text through the restorer | +0.0028 | pass |
| (7) NEU-ESC validation | −0.0039 (limit −0.01) | pass |

**H10b is passed.** This is the first intervention in the project that reduces real-typing
instability. The 12-layer model now changes about 12% of labels between a comment and its
normalization, against about 18% for the served recipe. It keeps its label distribution on
informal text, and it varies less across seeds. The served recipe's own distribution on informal
text drifts (label_tv 0.108); H10b's does not.

**Decision.**
- As declared, the seed-42 model is a 12-layer release candidate and a candidate teacher for a later
  student.
- Its weights would carry CC BY-NC-SA 4.0 (ViLexNorm).
- Which model the service runs stays the owner's decision 9, now with three options: the student,
  the 12-layer model, or H10b's 12-layer model.
- A closing gate on test for H10b is declared separately (`cycle5.yaml` v5) before it runs.

---

## ADR-043 · 2026-10-01 · Decision 9: the service runs the H10b model · Accepted

**Context.** Three releases were ready:
- the 12-layer served recipe (p9 s42);
- the 6-layer student, served since ADR-040;
- the 12-layer H10b candidate (ADR-042).

| | 12-layer (p9) | Student | H10b |
|---|---|---|---|
| Size / served p95 | 540 MB / 22.1 ms | 185 MB / 12.3 ms | 540 MB / same architecture as p9 |
| Test macro-F1, 5 seeds | 0.830 | 0.817 | 0.824 (−0.006 [−0.019, +0.007] vs p9's recipe) |
| Test neutral F1, seed 42 | 0.592 | 0.545 | 0.550 |
| Real-typing flips | about 18% | about 11% (dev) | about 12% (confirmed) |
| Challenge v1, served pipeline | 0.851 | 0.860 | 0.916 |

The recommendation was H10b. Its real-typing gain was confirmed by a declared rule on fresh pairs,
it leads on the constructed hard cases, and on test it cannot be told apart from the served recipe
at five seeds. The owner approved it on 2026-10-01.

**Decision.**
- **Served.** `models/serve/sentiment` holds the H10b release: FP32, restorer, scope detector,
  T = 1.349. The student is kept in `models/serve/.student-sentiment` and the 12-layer p9 release in
  `models/serve/.previous-sentiment`.
- **Rebuild.** `make export` rebuilds H10b.
- **API.** All 40 contract tests pass on the served release.

**Consequences.**
- **S7** (≤ 200 MB) is met by a released, tested artifact (the student) but not by the served one.
  Size was traded for accuracy and robustness.
- **S4's minimum** is met at seed 42 by the smallest margin (neutral F1 0.550).
- **Latency.** The served pipeline is the p9 architecture, 22.1 ms in the Cycle 5 sessions. Its own
  session waits for a quiet machine: `study latency` refuses to run while another process is on the
  GPU, and the one session run on 2026-10-01 had no steady pass on the served pipeline
  (`cycle5_h10b_session1.json`, model-only p95 34.4 ms against 20.5 ms for the same architecture).
- **Hub.** The owner uploaded the H10b bundle to `Datk4/vifeedback-sentiment-phobert` on
  2026-10-01 (Hub commit `037edfb`), licensed CC BY-NC-SA 4.0 because of the ViLexNorm training
  text. `serve reproduce` from the Hub: 8 files verified against `SHA256SUMS`, validation macro-F1
  0.8617 reproduced (manifest 0.8617) in 131 s. No stale file is left on the Hub.
- **Erratum to ADR-042.** The agreement gain is +0.061 (0.0615), not +0.062; corrected in place.

