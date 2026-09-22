# Status, Open Problems and Next Experiments

**Updated:** 2026-09-22, after Gate G3 and the ADR-015 correction.
Living document. Results live in [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md); decisions in
[DECISIONS.md](DECISIONS.md); this file says **where the project stands, what is wrong with it, and
what to run next.**

---

## 1. Progress

| Gate | Phase | Status | Evidence |
|---|---|---|---|
| **G0** | Foundations & data integrity | ✅ | `results/data_report.json`, 42 tests |
| **G1** | Baselines & evaluation harness | ✅ | 22 runs, harness pinned against sklearn |
| **G2** | PhoBERT reproduction | ✅ *(test eval deferred to G4, ADR-011)* | 10 runs, 5 seeds × 2 tasks |
| **G3** | Word-segmentation ablation | ✅ | 15 runs + latency benchmark |
| G4 | Improvement ladder | ⬜ next | — |
| G5 | Error analysis & robustness | ⬜ | — |
| G6 | CPU inference optimization | ⬜ | partially pre-empted, see [§ 3.6](#p6--the-quantization-headroom-is-smaller-than-planned) |
| G7 | Service, CI, release | ⬜ | — |

**CV checklist: 5/10** — items 1, 2, 3, 4, 5 complete.

**Compute so far:** 47 training runs, ~4.5 GPU-hours, all on the laptop RTX 3050. Zero external GPU
used to date.

---

## 2. Results

Dev split. Sentiment unless stated. Full tables in
[EXPERIMENT_MATRIX § 5](EXPERIMENT_MATRIX.md#5-result-tables-to-fill).

| Model | Preprocessing | **Macro-F1** | Weighted F1 | Neutral F1 |
|---|---|---|---|---|
| Majority class | — | 0.225 | 0.343 | 0.000 |
| TF-IDF word+char + priors *(fit=eval, optimistic)* | raw | ~~0.782~~ | 0.906 | 0.497 |
| **TF-IDF word+char + cross-fitted priors** | raw | **0.7708** | — | — |
| PhoBERT-base, 5 seeds | raw (P0) | 0.8436 ± 0.0079 | 0.9427 | 0.6139 ± 0.0222 |
| **PhoBERT-base, 5 seeds** | **VnCoreNLP (P1)** | **0.8670 ± 0.0072** | 0.9529 | **0.6680 ± 0.0184** |
| PhoBERT-base, 5 seeds | pyvi (P2b) | 0.8643 ± 0.0098 | 0.9512 | 0.6628 ± 0.0291 |
| PhoBERT-base, 5 seeds | underthesea (P2) | 0.8618 ± 0.0063 | 0.9506 | 0.6560 ± 0.0178 |
| Topic: TF-IDF B4 LinearSVC *(honest best)* | raw | **0.768** | 0.874 | others 0.479 |
| Topic: PhoBERT-base, 5 seeds | raw (P0) | 0.7971 ± 0.0018 | 0.8889 | others 0.568 ± 0.011 |

### Against the revised success criteria (ADR-008)

| ID | Criterion | Min | Target | Current (dev) | |
|---|---|---|---|---|---|
| S1 | Sentiment macro-F1 | 0.80 | 0.84 | **0.867** | ✅ target exceeded |
| S2 | Topic macro-F1 | 0.79 | 0.83 | 0.797 | ⚠️ min met, target far |
| S3 | Lift over honest baseline | +0.025, p<0.05 | +0.05 | **+0.096**, p=0.001 | ✅ target exceeded |
| S4 | Neutral F1 | 0.55 | 0.65 | **0.668** | ✅ target exceeded |
| S5 | p95 latency | ≤60 ms | ≤30 ms | **51.1 ms** | ✅ min met, target open |
| S6–S10 | inference / errors / release | — | — | not started | ⬜ |

**These are dev numbers.** The test set has not been touched (0 evaluations logged). S1–S4 are
provisional until G4.

### Four findings worth keeping

1. **PhoBERT's advantage is almost entirely the minority class.** Weighted F1 +0.037, macro-F1 +0.062,
   **neutral F1 +0.117**. Through weighted F1 alone the transformer looks barely worth the GPU.
2. **The published "word segmentation is unnecessary" finding replicates on its own metric and fails
   on macro-F1.** Under 1 pp on accuracy and weighted F1 exactly as reported; **+2.34 pp macro-F1** and
   **+5.42 pp neutral F1**. A conclusion in the literature turns out to be an artifact of aggregating
   over a 4% class (ADR-012).
3. **TF-IDF beats PhoBERT on the `facility` topic** (0.921 vs 0.905, 9× the seed std). Distinctive
   vocabulary is what TF-IDF represents best, and a contextual model has nothing to add there.
4. **Decision-threshold tuning does not generalize here, and a Gate G1 conclusion was retracted
   because of it** (ADR-015). Cross-fitting removes the *entire* apparent gain on PhoBERT. The
   inflated baselines had been *understating* PhoBERT's advantage, so the correction raises the
   headline lift from +0.085 to **+0.096**.

---

## 3. Open problems

Ordered by how much they threaten the conclusions.

### P1 — The dev set cannot resolve the effects we are measuring
**Measured.** Dev holds **73 neutral examples** bearing one third of the macro average, so a bootstrap
CI on dev macro-F1 is **±0.027**. Most Phase 4 increments will be smaller than that. The test set would
only narrow it to ~0.019.

*Consequence:* single-run dev comparisons are uninformative at this project's effect sizes. ADR-013
makes the seed-level paired test primary, which resolved P0-vs-P1 at p = 0.001 where the bootstrap
could not.

*Directions:*
- **(a) Repeated stratified k-fold CV on `train` only** for Phase 4 selection — 5 folds × 3 seeds gives
  15 estimates instead of 5, without touching dev or test, so ADR-002 is untouched. Cost: 3× the GPU
  time per recipe.
- **(b) Raise seeds to 10** for the final two or three candidates.
- **(c) Report a stated resolution floor** (~0.027 single-run dev) in the write-up. Any UIT-VSFC result
  claiming a +0.01 improvement from one run is reporting noise, and saying so is a contribution.

### P2 — Topic is the weak task and we have not tried to fix it
Topic macro-F1 0.797 barely clears the S2 minimum, versus 0.867 for sentiment. `others` F1 is 0.568.
**The segmentation ablation has not been run for topic at all** — only sentiment.

*Directions:* run P1/P2b for topic (10 runs, ~50 min); expect a similar +0.02. Then Tier A on `others`.
The topic ceiling is genuinely capped by a **71.07% inter-annotator agreement** and a catch-all class,
so the write-up should report topic performance **relative to the IAA** rather than as a flat number.

### P3 — Measured gold-label noise bounds the achievable ceiling
**~4.8% disagreement on textually identical train/test pairs** (3/63 sentiment, 2/63 topic).
Politeness formulae (`cám ơn thầy`) are annotated inconsistently between `neutral` and `positive`, and
that phrase family is a visible part of the neutral class.

*Directions:* a confident-learning audit of `train` in Tier D, with the top-100 suspected labels
inspected by hand. **Never clean test** — report what is found and treat it as a ceiling estimate.

### P4 — `facility` regression is unexplained and possibly exploitable
PhoBERT is *worse* than TF-IDF on one class. Currently reported, not acted on.

*Direction:* a **hybrid** — stack TF-IDF probabilities with PhoBERT logits for topic, or route by
class. This is unusual enough to be interesting and is grounded in a measured observation rather than
a hunch.

### P5 — Robustness is untested because the benchmark cannot test it
The corpus is 99.86% diacritized and 99.84% teencode-free (ADR-007), so informal-orthography
robustness cannot be observed here — only **induced**.

*Direction:* the Phase 5 perturbation suites are now the *only* evidence on this axis and should be
reported in their own section, never merged into the clean-test table.

### P6 — The quantization headroom is smaller than planned
`max_length` 96 (Gate G0) and dynamic padding already deliver **3.49×** (p95 177.5 → 50.8 ms), meeting
the S5 minimum with no ONNX and no INT8. And the reference CPU has **no AVX512-VNNI**, so INT8's
expected benefit is uncertain (H3).

*Consequence:* Phase 6's question changes from *"can we hit 60 ms?"* (already yes) to **"does ONNX/INT8
add anything on top of two free configuration changes, on a CPU without VNNI?"** That is a sharper
question and the answer may legitimately be *no* — which is still a result, and one worth reporting.

### P9 — Anything fitted on the evaluation set is suspect until cross-fitted
Threshold tuning looked worth +0.035 and was worth **zero** on PhoBERT once cross-fitted (ADR-015).
Root cause: 73 neutral dev examples cannot support a transferable decision boundary — the same
bottleneck as P1.

*Direction:* the standing rule from ADR-015 now applies to everything still to come that fits a
parameter on dev — calibration temperature, ensemble weights, model-soup coefficients, and the
Phase 6 quantization calibration set. Each must be cross-fitted before its benefit is claimed.

### P7 — No test evaluation has happened yet
Deferred to G4 (ADR-011). Every S1–S4 number above is dev. Phase 2/3 runs saved no checkpoints, so
the G4 test pass will need a retrain of the champion configuration — budget ~25 min for it.

### P8 — H2, the project's headline hypothesis, was wrong
Both halves falsified (ADR-012). The planned CV headline ("removed segmentation, 60% of p95") is gone.

*Direction:* the replacement is better and should be led with: *a published negative result does not
replicate once the minority class is made visible.* That is a claim about the literature, not just
about this pipeline.

---

## 4. Next experiments — Phase 4

All laptop-feasible unless marked. Selection on the 5-seed mean per ADR-013.

### Tier A — imbalance *(highest expected value)*
Re-scoped by ADR-015 to **training-time methods only**. Post-hoc thresholding was tested first
(it is free) and cross-fits to no gain on PhoBERT, so the imbalance correction has to happen during
training or not at all.

| Recipe | Runs | Note |
|---|---|---|
| ~~threshold tuning on PhoBERT probabilities~~ | — | **dropped** — cross-fitted gain is zero (ADR-015) |
| `classweight` (balanced / sqrt / effective) | 15 | 3 schemes × 5 seeds |
| `focal` γ ∈ {1, 2} | 10 | |
| `logit-adjust` τ ∈ {0.5, 1.0} | 10 | principled for long tails, free at inference |

### Tier B — stability
Best epochs varied 2–4 across seeds, so the schedule is not settled.

| Recipe | Runs |
|---|---|
| LLRD 0.9 / 0.95 | 10 |
| lr ∈ {1e-5, 3e-5} × epochs ∈ {6, 10} with early stopping | 20 (3 seeds first) |

### Tier C — regularization and robustness
| Recipe | Runs | Note |
|---|---|---|
| label smoothing 0.05 / 0.1 | 10 | |
| R-Drop α ∈ {0.5, 1.0} | 10 | strong on small data |
| **FGM** ε ∈ {0.5, 1.0} | 10 | also hardens against the Phase 5 perturbations — pays twice |

### Tier D — data-centric
| Recipe | Runs | Note |
|---|---|---|
| synthetic teencode + diacritic-stripping augmentation | 10 | adds data *and* targets P5 |
| back-translation for neutral | 10 | needs a translation model |
| confident-learning label audit | 5 | addresses P3 |

### Tier E — model axis
| Model | Where | VRAM | Runs |
|---|---|---|---|
| `phobert-base-v2` | **laptop** | ~3.6 GB | 5 |
| `visobert` | **laptop** | ~3.0 GB | 5 |
| `xlm-roberta-base`, embeddings frozen | **laptop** | ~3.6 GB | 5 |
| `xlm-roberta-base`, full | Kaggle/Colab | ~5.9 GB | 5 |
| `phobert-large` | **Kaggle** | ~7.9 GB | 5 |
| multi-task (shared encoder, 2 heads) | **laptop** | ~3.6 GB | 5 |

Multi-task is justified by measurement, not fashion: Cramér's V(sentiment, topic) = **0.344**, and
`facility` is 95.6% negative while `lecturer` is 62.1% positive. There is real signal to share, and it
halves the number of served models.

### Tier F — ensembling
| Recipe | Note |
|---|---|
| **model soup** (weight averaging across seeds) | free at inference — the right trade under a latency budget |
| seed logit ensemble | N× cost; include on the Pareto plot to show it losing |
| **TF-IDF × PhoBERT hybrid for topic** | motivated by P4 |

**Estimated Phase 4 cost:** ~120 laptop runs ≈ 10 GPU-hours, plus 10 runs on Kaggle.

---

## 5. Experiment expansion — beyond the current scope

Ranked by value per hour. Each needs an ADR before it starts (ADR-003 boundary).

1. **Knowledge distillation** into a 4–6 layer student. Best latency story available, and the only
   remaining large lever on S5's 30 ms target if quantization disappoints (P6). *Laptop.*
2. **Cross-domain zero-shot** on UIT-ViSFD or NEU-ESC. One evaluation run; a domain-shift number is
   strong evidence of maturity and directly addresses limitation L6. *Laptop.*
3. **LLM reference row.** Zero/few-shot classification as a single comparison line with cost and
   latency beside it. Positions the fine-tuned encoder honestly instead of ignoring the elephant.
4. **Calibration** (ECE, reliability diagram, temperature scaling). The service returns probabilities;
   it should be able to say how well calibrated they are. Already instrumented — `ece_10bin` is
   computed on every run. *Free.*
5. **LLM-as-teacher pseudo-labelling** for neutral, then retrain. Addresses the 458-example bottleneck
   at its root.
6. **Repeated CV selection** (P1 direction (a)) — a methodological upgrade that raises confidence in
   every Phase 4 conclusion.
7. **Human re-annotation of 100 neutral examples** to estimate the true ceiling against measured label
   noise (P3). Cheap, and almost nobody does it.

---

## 6. Compute plan

| Work | Where | Why |
|---|---|---|
| Phases 0–3, Tier A–D, most of Tier E, Tier F | **Laptop RTX 3050** | 69 s/epoch, no session limits, artifacts in one place |
| `phobert-large`, unfrozen `xlm-roberta-base` | **Kaggle P100 16 GB** | exceeds 4.29 GB; Kaggle's 30 h/week quota is stated, Colab's is not (ADR-014) |
| **All latency benchmarking, API, Docker, CI** | **Laptop only** | the reference machine is documented in `env.json`; a p95 from a cloud VM is not comparable |

Notebooks: [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb) (preferred),
[`notebooks/colab_train.ipynb`](../notebooks/colab_train.ipynb). Both are logic-free wrappers around
the same CLI, so a platform-only bug is impossible.

**Thermal constraint (ADR-009):** the laptop is both the training machine and the latency reference.
Training drove the GPU to 83 °C; the segmentation benchmark above was taken at 44 °C idle. No
benchmark runs while training runs, and the 5-repetition / >10%-disagreement rule in
[EVALUATION_PROTOCOL § Latency harness](EVALUATION_PROTOCOL.md#latency-harness) is the detector for a
violation.
