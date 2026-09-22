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

<!-- Append new entries above this line. -->
