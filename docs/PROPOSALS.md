# Proposals — techniques, models and workflow

**Written:** 2026-09-22, after Gate G3 and ADR-015.

Every proposal below is anchored to a **measurement from this project**, not to a technique's general
reputation. Expected gains are labelled as **hypotheses to be falsified** — ADR-008 and ADR-015 are
recent reminders that this project's predictions have been wrong in both directions.

---

## 1. What the measurements are telling us

Five facts constrain everything worth proposing.

### F1 — The errors have an ordinal signature, and it is almost perfectly symmetric

Mean confusion over 5 seeds, PhoBERT + segmentation, dev:

| true \ pred | negative | neutral | positive |
|---|---|---|---|
| **negative** | 687.2 | 6.2 | 11.6 |
| **neutral** | **15.2** | 43.0 | **14.8** |
| **positive** | 17.6 | 6.6 | 780.8 |

Of 73 true neutrals: **20.8% go to negative, 20.3% go to positive.** That symmetry is the textbook
signature of a middle class on an ordinal scale — the model cannot decide which side it falls on.

**59.4% of all errors involve the middle class, which is 4.6% of the data.**

And the model is *conservative* about it: neutral precision 0.77, recall **0.59**. It misses 41% of
neutrals rather than over-predicting them.

**Nothing in the current setup knows that `negative < neutral < positive`.** A 3-way softmax treats
the labels as unordered, discarding the one piece of structure that speaks directly to the dominant
error mode.

### F2 — Post-hoc correction fails for a *sample-size* reason, not a conceptual one
Threshold tuning cross-fits to zero gain (ADR-015) because dev has 73 neutral examples. The idea
— "shift the decision boundary toward the minority class" — is sound; the *place* it was applied is
too small. **The same correction applied on `train` (458 neutral) has 6× the data.**

### F3 — Topic is a strong prior for neutral, and it is free
Cramér's V(sentiment, topic) = 0.344. Neutral is **28.3% of `others`** and **2.5% of `lecturer`** —
an 11× difference. A model that predicts topic knows something real about whether sentiment is neutral.

### F4 — TF-IDF beats PhoBERT on `facility` (0.921 vs 0.905, 9× seed std)
Distinctive vocabulary is where sparse features win. Currently reported, not exploited.

### F5 — The dev set cannot resolve effects below ~0.027 macro-F1
(ADR-013.) Most of what follows is expected to be smaller than that, so **the selection procedure
itself has to change before the techniques are worth running.**

---

## 2. Techniques

Ranked by (expected gain × confidence) ÷ cost.

### T1 — Ordinal-aware head: CORN ⭐ *highest value*

**Motivated by F1.** Replace the 3-way softmax with a rank-consistent ordinal head:
two binary classifiers predicting `P(y > negative)` and `P(y > neutral)`.
[CORN](https://arxiv.org/abs/2111.08851) achieves rank consistency through conditional training sets
rather than CORAL's weight-sharing constraint, and reports substantially better results for it.

*Why it should help here specifically:* the neutral class is defined by *not* being at either
extreme. Under CORN it is predicted as the *conjunction* `P(>neg) high AND P(>neu) low`, so both
majority classes contribute evidence for it — instead of competing with it for softmax mass.

*Also brings better metrics.* Ordinal errors are not equal: `positive → negative` is worse than
`positive → neutral`. Add **MAE over ordinal labels** and **quadratic weighted kappa** alongside
macro-F1. Currently a 1-step and a 2-step error cost the same, which is wrong for this label set.

| | |
|---|---|
| Cost | ~1 day to implement; 5 runs (~25 min) |
| Applies to | sentiment only — topic is genuinely unordered |
| Hypothesis | neutral F1 +0.02 to +0.05; macro-F1 +0.01 to +0.03 |
| Risk | The ordinal assumption may be wrong: some "neutral" cases are *off-topic* or *no opinion*, not *middling* sentiment ([DATA_CARD § 6](DATA_CARD.md#6-surface-and-linguistic-profile--measured) lists 6 sub-types). If the gain is zero, that is itself a finding about what the label means |

### T2 — Decoupled training: cRT / τ-normalization ⭐ *best cost-to-value*

**Motivated by F2.** [Decoupling Representation and Classifier](https://arxiv.org/abs/1910.09217)
(ICLR 2020) is the standard long-tail recipe and is exactly the training-time version of the
correction that failed post-hoc:

* **stage 1** — train normally on the natural distribution (the representation is *not* the problem);
* **stage 2** — freeze the encoder, re-initialize and retrain **only the classifier head** with
  class-balanced sampling.

*Why it fixes what ADR-015 broke:* the head is 768×3 = 2,304 parameters, so it can be retrained on
**train** (458 neutral) instead of dev (73). Six times the data for the part that needs it.

**τ-normalization is nearly free**: rescale each class's classifier weights by `‖w_c‖^(-τ)` with no
retraining at all — the weight norm correlates with class frequency, and τ is one scalar.

| | |
|---|---|
| Cost | stage 2 is < 1 min per run; τ-norm is seconds |
| Hypothesis | neutral F1 +0.02 to +0.06 |
| Risk | Low. τ must be cross-fitted (ADR-015 standing rule) |

### T3 — Multi-task: topic as an auxiliary head ⭐

**Motivated by F3.** One shared encoder, two heads, joint loss. Already in the Phase 4 plan; F3
upgrades it from "worth trying" to "well-motivated": the auxiliary signal is 11× more predictive of
neutral for `others` than for `lecturer`.

Two operational bonuses: it halves the number of served models, and it halves the Phase 6 benchmark
surface.

| | |
|---|---|
| Cost | 5 runs (~25 min); the trainer needs a two-head model class |
| Hypothesis | macro-F1 +0.005 to +0.02 on both tasks; the real win may be operational |
| Risk | Task interference. Sweep the loss weight λ ∈ {0.3, 0.5, 1.0} |

### T4 — Sparse–dense hybrid for topic

**Motivated by F4.** Stack TF-IDF probabilities with PhoBERT logits for topic (a small logistic
meta-learner), or route `facility` by a sparse rule.

*Why it is interesting:* it is a measured observation that contradicts the usual "transformers
dominate" narrative, and acting on it is unusual enough to be memorable in a write-up.

| | |
|---|---|
| Cost | ~2 h; no GPU (both models' predictions already exist) |
| Hypothesis | topic macro-F1 +0.01 to +0.02, concentrated in `facility` |
| Risk | The meta-learner's weights are fitted on dev → **must be cross-fitted** (ADR-015) |

### T5 — Supervised contrastive with class-aware sampling

Plain SupCon **collapses on imbalanced data** — the majority class dominates the loss and the feature
space degenerates ([CVPR 2025](https://arxiv.org/abs/2503.17024)). The fixes are class-aware sampling
(CA-SupCon) or the K-class positive-set rule (KCL), which equalize positives per anchor.

| | |
|---|---|
| Cost | ~1 day; 10 runs |
| Hypothesis | neutral F1 +0.01 to +0.04 |
| Risk | **Medium-high.** 458 neutral examples across batches of 32 means ~1.3 neutral per batch — near the regime where SupCon is documented to fail. Needs class-aware sampling to be viable at all |

### T6 — Synthetic neutral generation with an LLM

**Motivated by the root cause:** 458 examples. Generate 500–1,000 synthetic neutral sentences in the
UIT-VSFC register (short, lowercase, student feedback), filtered by a round-trip classifier check.

| | |
|---|---|
| Cost | API cost + ~half a day |
| Hypothesis | neutral F1 +0.02 to +0.05 |
| Risk | **Distribution drift.** Synthetic neutrals may be more prototypical than real ones, which are a heterogeneous grab-bag of ≥6 phenomena. Must be validated by training on synthetic and evaluating on *real* dev only |

---

## 3. Models

| Model | Params | Fits 4.29 GB? | Why try it |
|---|---|---|---|
| `vinai/phobert-base-v2` | 135M | ✅ | Same size, +120 GB OSCAR pretraining. Cheapest possible upgrade |
| `uitnlp/visobert` | ~97M | ✅ | Social-media pretrained, SentencePiece. **Smaller and faster** — a Pareto candidate for Phase 6, not just accuracy |
| `Fsoft-AIC/videberta-base` | — | ✅ likely | **DeBERTa** disentangled attention — a genuinely different inductive bias, not another RoBERTa |
| `Fsoft-AIC/videberta-xsmall` | — | ✅ | A distillation *target* baseline for Phase 8 |
| `xlm-roberta-base`, embeddings frozen | 277M (85M trainable) | ✅ | Multilingual control, made to fit |
| `uitnlp/CafeBERT` | **560M** | ❌ Kaggle | XLM-R-large continued on 18 GB Vietnamese; reported best on several Vietnamese tasks |
| `vinai/phobert-large` | 368M | ❌ Kaggle | Conflicts with the CPU latency objective — run it for the table, do not plan to ship it |

**Recommendation:** run the four local models first. `videberta-base` and `visobert` are the
interesting ones — a different architecture and a smaller/faster candidate. CafeBERT and
PhoBERT-large are one Kaggle session, for the comparison table only.

---

## 4. Pipeline and workflow proposals

These matter more than any single technique, because F5 says the current selection procedure cannot
distinguish the techniques above from noise.

### W1 — Repeated k-fold CV on `train` for selection ⭐ *do this before anything else*

**The blocking problem.** Dev resolves nothing below ~0.027 macro-F1 (ADR-013). Every technique in
§ 2 is *hypothesized* to be smaller than that. Running them under the current procedure would produce
a table of statistically meaningless numbers.

**Proposal:** select on **5-fold CV over `train` only** × 3 seeds = 15 estimates per recipe, with dev
kept as a clean confirmation set and test still locked. ADR-002 is untouched — the official split
survives, because folds are cut inside `train`.

Gains: ~3.9× tighter standard error, and 458 neutral examples per fold-union instead of 73.
Cost: 3× GPU per recipe (~75 min per recipe at 5 seeds → acceptable at 69 s/epoch).

### W2 — Ordinal-aware and cost-aware metrics
Add **MAE** and **quadratic weighted kappa** for sentiment. A `positive → negative` error is worse
than `positive → neutral`, and macro-F1 cannot express that. Cheap, and it makes T1 measurable.

### W3 — Minority-class dashboard as the primary view
Every result table currently leads with macro-F1. Given F1 (59.4% of errors involve the middle class),
the leading view should be **neutral precision / recall / F1 and the two off-diagonal cells**. The
aggregate is a summary of that, not the other way round.

### W4 — Regression gate in CI
CI compares each run against the registry's best for that configuration and fails on a drop beyond
the seed std. Turns `registry.csv` from a log into a **test oracle**.

### W5 — Challenge-set-driven development, moved earlier
The Phase 5 perturbation suites currently sit after Phase 4. Building them **now** gives a second
evaluation axis for every Phase 4 recipe at no extra training cost — FGM and augmentation are
explicitly expected to trade clean accuracy for robustness, and without the suites that trade is
invisible until it is too late to act on.

### W6 — Config inheritance for recipes
`TrainConfig` is flat, so every recipe is a fresh construction. A small YAML inheritance layer
(`base.yaml` → `tier_a/focal.yaml`) makes a 40-recipe sweep declarative and makes "one axis at a
time" enforceable by diffing configs rather than by discipline.

---

## 5. Recommended order

| # | Item | Cost | Why first |
|---|---|---|---|
| 1 | **W1** repeated CV selection | 1 day + 3× GPU | Everything else is unmeasurable without it |
| 2 | **T2** τ-normalization | ~1 h | Nearly free; directly targets what ADR-015 broke |
| 3 | **W2** ordinal metrics | ~2 h | Needed to evaluate T1 |
| 4 | **T1** CORN ordinal head | 1 day | Strongest evidence-backed hypothesis (F1) |
| 5 | **T2** cRT stage 2 | ~4 h | Same motivation, more machinery |
| 6 | Models: `phobert-base-v2`, `visobert`, `videberta-base` | 15 runs | Cheap, local, and `visobert` may be a Pareto win |
| 7 | **T3** multi-task | 1 day | F3 support + operational benefit |
| 8 | **W5** challenge sets early | 1 day | Second axis for everything after it |
| 9 | **T4** sparse–dense hybrid | ~2 h | Small, unusual, memorable |
| 10 | **T6** synthetic neutrals | ~1 day + API | Attacks the root cause; highest variance |
| 11 | **T5** CA-SupCon | 1 day | Highest risk at this batch size |

**Items 1–6 are all laptop-feasible.** Kaggle is needed only for CafeBERT and PhoBERT-large, and only
for the comparison table.

---

## 6. What this would change about the project's story

Currently the headline is *"PhoBERT beats TF-IDF by +0.096 macro-F1, and a published negative result
does not replicate once the minority class is visible."*

With T1/T2 the story could become sharper: *"the dominant error mode on a 3-class sentiment task with
a 4% middle class is ordinal confusion — 59% of errors involve a class holding 4.6% of the data, split
almost exactly evenly between the two poles — and treating the labels as ordered rather than
categorical addresses it directly."*

That is a claim about **the task**, not about a model, and it is the kind of thing a reviewer
remembers. It also transfers: every 3-class sentiment dataset with a small neutral class has this
structure, and almost none of them model it.
