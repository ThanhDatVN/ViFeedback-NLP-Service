# Where this project stands against published UIT-VSFC results

**Updated:** 2026-09-22, after Gate G3.

Short answer to "do we have an advantage yet?": **on methodology, clearly yes. On the score, probably
— and it is not yet proven, because our numbers are dev and theirs are test.** This document states
both halves precisely rather than picking the flattering one.

---

## 1. The published landscape

| Source | Model | Sentiment | Topic | **Metric actually reported** |
|---|---|---|---|---|
| Nguyen et al., KSE 2018 (corpus paper) | Maximum Entropy | ~0.88 | >0.84 | weighted F1 |
| Deep-learning comparison study | Bi-LSTM + Word2Vec | 0.92 | 0.896 | weighted F1 |
| Recent PhoBERT work | PhoBERT | ~0.94 F1 / 94.5% acc | — | weighted F1 / accuracy |
| **BamiBERT (2026)** | BamiBERT | 93.86 acc / **83.41 macro-F1** | — | both |
| Independent reproduction | PhoBERT | **~0.83 macro-F1** | — | macro |

**Only two of five report macro-F1.** The rest report weighted F1 or accuracy on a corpus that is
4.32% neutral — where a model that never predicts neutral scores 0.947 accuracy and 0.922 weighted F1
([DATA_CARD § 4](DATA_CARD.md#4-label-schemes-and-class-distribution--measured)).

---

## 2. Our numbers

PhoBERT-base, VnCoreNLP segmentation, `max_length` 96, **5 seeds, dev split**:

| | Ours (dev) | Best published | Gap |
|---|---|---|---|
| Sentiment accuracy | **0.9545 ± 0.002** | 94.5% / 93.86% | +0.9 pp |
| Sentiment weighted F1 | **0.9529 ± 0.002** | ~0.94 | +1.3 pp |
| **Sentiment macro-F1** | **0.8670 ± 0.0072** | **0.8341** (BamiBERT) | **+3.3 pp** |
| Topic macro-F1 | 0.7971 ± 0.0018 | — *(none report macro)* | — |
| Topic weighted F1 | 0.8889 ± 0.003 | 0.896 (Bi-LSTM) | −0.7 pp |

### The caveat that must travel with these numbers

**Ours are dev; theirs are test.** Dev is 1,583 examples with 73 neutral; test is 3,166 with 167.
They are not the same measurement, and the dev/test gap on this corpus is unknown to us because the
test set has been evaluated **zero** times (ADR-011 deferred it to Gate G4).

So the honest statement today is: *"we exceed the published macro-F1 by 3.3 points on dev, and have
not yet measured test."* Anything stronger would be the exact failure this project was built to
avoid.

Resolving it costs ~25 minutes: retrain the champion with checkpointing and run one locked test pass.
That is Gate G4.

---

## 3. What is genuinely comparable, and what is not

| Claim | Comparable? | Why |
|---|---|---|
| Our weighted F1 vs published weighted F1 | ⚠️ dev vs test | Same metric, different split |
| Our macro-F1 vs BamiBERT's macro-F1 | ⚠️ dev vs test | Same metric, different split |
| Our accuracy vs published accuracy | ⚠️ dev vs test | Same metric, different split |
| Our topic macro-F1 vs anything | ❌ | **Nobody publishes topic macro-F1** |
| Our latency vs anything | ❌ | **Nobody publishes latency at all** |
| Our seed variance vs anything | ❌ | **Nobody publishes seed variance** |

The last three rows are the interesting ones: they are not gaps in our evidence, they are gaps in
**the literature's**.

---

## 4. The advantages that do not depend on the score

These hold regardless of how Gate G4 lands, and they are what a reviewer would find hard to match.

### 4.1 Reported quantities nobody else reports

| Quantity | Us | Published work |
|---|---|---|
| Macro-F1 with 5-seed mean ± std | ✅ 0.8670 ± 0.0072 | 2 of 5 report macro; **none** report seed variance |
| Per-class F1 **with bootstrap CI** | ✅ neutral 0.688 **[0.593, 0.772]** | none |
| Train↔test leakage, measured and published | ✅ 0 exact / 1.74% normalized | none |
| Gold-label noise, measured | ✅ ~4.8% on identical text | none |
| CPU p95 latency on documented hardware | ✅ 50.8 ms, CPU flags recorded | none |
| Ordinal metrics (QWK, MAE, adjacent acc) | ✅ 0.947 / 0.064 / 0.982 | none |
| Statistical resolution floor | ✅ ~0.027 macro-F1 on dev | none |
| Every number traceable to a `run_id` | ✅ 49 runs | n/a |

### 4.2 A finding about the literature, not just about a model

The most-cited preprocessing result on this task — *word segmentation is unnecessary for Vietnamese
sentiment classification* ([arXiv:2301.00418](https://arxiv.org/abs/2301.00418)) — **replicates
exactly on the metric it reported and fails on macro-F1**:

| Metric | P0 raw → P1 segmented |
|---|---|
| Accuracy | +0.96 pp ← *"under 1 percentage point", as published* |
| Weighted F1 | +1.02 pp ← *as published* |
| **Macro-F1** | **+2.34 pp** |
| **Neutral F1** | **+5.42 pp** |

Significant at **p = 0.0010** (paired t over 5 seeds, Cohen d = 3.84, non-overlapping seed ranges).

A published conclusion turns out to be an artifact of aggregating over a 4% class. That is a
contribution to the field's practice, and it does not depend on our score being higher than anyone's.

### 4.3 A documented self-correction

[ADR-015](DECISIONS.md) retracts one of our own Gate G1 conclusions: decision-threshold tuning looked
worth +0.035 macro-F1 and was worth **zero** once cross-fitted. The correction *raised* the reported
lift (+0.085 → +0.096) because the inflated baseline had been understating the model.

Most published work has no mechanism that would surface this, because the practice of fitting a
threshold on dev and reporting the dev score is widespread and rarely questioned.

### 4.4 Deployment evidence

No published UIT-VSFC result reports latency, model size, or hardware. We report p95 on a named CPU
with its instruction-set flags, which is the difference between a benchmark number and a deployable
system.

---

## 5. What would make the numerical claim airtight

| Step | Cost | Closes |
|---|---|---|
| Gate G4 test evaluation of the champion | ~25 min | The dev/test caveat — the blocking item |
| Report **test** macro-F1 ± std over 5 seeds | included | Directly comparable to BamiBERT's 0.8341 |
| Publish topic macro-F1 | done | A number nobody else has published |
| Evaluate on the deduplicated test set | ~1 min | Pre-empts "your 1.74% overlap inflated it" |
| Report against the 71.07% topic IAA | ~1 h | Frames the topic ceiling honestly |

---

## 6. Honest summary

**What can be claimed today:**

* the most thoroughly measured UIT-VSFC result we are aware of — seed variance, per-class CIs,
  leakage, label noise, latency, and ordinal structure, none of which appear in the published work;
* a **replication failure of a published preprocessing conclusion**, shown to be metric-dependent;
* a documented retraction of one of our own claims, with the corrected numbers.

**What cannot be claimed yet:**

* state of the art. Our 0.8670 is dev; BamiBERT's 0.8341 is test. Until Gate G4 the comparison is
  suggestive and nothing more.

**The durable advantage is the second list being short and the first being long.** Scores get beaten
by the next model; a measurement protocol that exposes what the scores hide does not.
