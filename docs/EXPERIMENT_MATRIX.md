# Experiment Matrix

Companion to [ROADMAP.md](ROADMAP.md). This file defines **what gets run, in what order, and where the
number goes when it comes back.** Every table below is a template to be filled from
`results/registry.csv` — never retyped by hand.

---

## Table of contents

1. [Run ID scheme](#1-run-id-scheme)
2. [Matrix axes](#2-matrix-axes)
3. [Run inventory and compute budget](#3-run-inventory-and-compute-budget)
4. [Expected ranges (pre-registered)](#expected-ranges-pre-registered)
5. [Result tables to fill](#5-result-tables-to-fill)
6. [Experimental hygiene rules](#6-experimental-hygiene-rules)

---

## 1. Run ID scheme

```
<phase>-<task>-<model>-<preproc>-<recipe>-s<seed>
```

Examples: `p1-sent-tfidfchar-raw-lrbal-s42` · `p3-sent-phobert-P0-base-s1337` ·
`p6-sent-phobert-P0-base-onnxint8s-s42`

Rules: lowercase, hyphen-separated, no spaces, and the id is both the `results/runs/` directory name and
the `run_id` column in `registry.csv`. A run that cannot be named under this scheme is a run whose
configuration you have not actually decided on.

---

## 2. Matrix axes

The full Cartesian product is ~10,000 runs and is not the plan. The plan is **one axis at a time**, each
sweep anchored to the current best configuration on every other axis (a coordinate-descent search, not a
grid search).

| Axis | Levels | Swept in |
|---|---|---|
| **Task** | sentiment (3-class), topic (4-class), multi-task | All phases |
| **Preprocessing** | P0 raw · P1 RDRSegmenter · P2 underthesea · P3 +NFC/cleanup · P4 +teencode · P5 +lowercase · P6 +emoji | Phase 3 |
| **Model** | B0 majority · B0b random · B1 tfidf-word · B2 tfidf-char · B3 tfidf-union · B4 LinearSVC · M1 phobert-base · M2 phobert-base-v2 · M3 visobert · M4 xlm-r-base · M5 phobert-large · M6 multitask · M7 soup/ensemble · M8 distilled student | Phases 1, 2, 4 |
| **Recipe** | base · classweight · focal · logit-adjust · llrd · rdrop · fgm · labelsmooth · ema · aug-bt · aug-teencode · threshold-tuned | Phase 4 |
| **Runtime** | L0 torch-fp32 · L1 dynpad · L2 threads · L3 onnx-fp32 · L4 onnx-int8-dyn · L5 onnx-int8-static · L6 openvino-int8 · L7 no-segment · L8 distil | Phase 6 |
| **Seed** | 42 · 1337 · 2024 · 7 · 31337 | Every transformer run |
| **Split** | dev (all selection) · test (gates only, logged) | Every run |

**The one-axis-at-a-time rule is what makes the ablation table readable.** A result that changed two
things at once is not an ablation; it is an anecdote.

---

## 3. Run inventory and compute budget

| Phase | Configurations | Seeds | Runs | Where | Est. compute |
|---|---|---|---|---|---|
| P1 Baselines | 6 models × 2 tasks | 3 | ~36 | Laptop CPU | < 2 h total |
| P2 PhoBERT | 1 config × 2 tasks | 5 | 10 | Colab T4 | ~4 h |
| P3 Preproc ablation | 7 preproc × 2 tasks (partial: 7 × sent, 3 × topic) | 5 | ~50 | Colab T4 | ~12 h |
| P4 Improvement | ~16 recipes/models × sentiment, ~6 × topic | 3 → 5 for finalists | ~75 | Colab T4 | ~20 h |
| P5 Error analysis | perturbation suites (4) × 2 tasks, inference only | 1 | 8 | Laptop CPU | < 1 h |
| P6 Inference | 8 runtime steps × 2 tasks | n/a | 16 benchmarks | Laptop CPU | ~6 h (mostly benchmarking) |
| **Total** | | | **~195 runs** | | **~45 h compute** |

**Colab budget discipline.** ~36 GPU-hours across weeks 3–5 is about 12 h/week. That fits free-tier T4
allowances only if each run is short and resumable. Enforce three rules:
keep every single run under 30 minutes; checkpoint to Drive at the end of every epoch; and queue runs from
a config list so that a killed session costs one run, not one evening.

**Cut order if the GPU budget runs out:** drop topic-task seeds from 5 → 3 · drop P3 conditions P5/P6 ·
drop Tier E models M4/M5 · drop Tier F entirely. Never drop below 5 seeds on the runs that appear in the
final headline table.

---

## Expected ranges (pre-registered)

Recorded **before any experiment runs** so that a surprise is visible as a surprise. These are
predictions to be falsified, not results. Reasoning is given so that a miss can be diagnosed rather than
merely noted.

### Sentiment (3-class; overall distribution ≈ 49.8% positive, 45.8% negative, **4.3% neutral**)

| Model | Accuracy | Weighted F1 | **Macro-F1** | Neutral F1 | Reasoning |
|---|---|---|---|---|---|
| B0 majority | ~0.50 | ~0.33 | **~0.22** | 0.00 | Structural floor for 3 classes with one dominant |
| B1 tfidf-word + LR | 0.88–0.90 | 0.87–0.89 | **0.60–0.68** | 0.05–0.25 | Linear model has almost no signal for 458 neutral examples |
| B2 tfidf-char + LR | 0.88–0.91 | 0.87–0.90 | **0.63–0.70** | 0.10–0.30 | Char n-grams absorb teencode / missing diacritics |
| B5 + class weights | 0.85–0.89 | 0.86–0.89 | **0.66–0.72** | 0.25–0.38 | Neutral recall rises, precision falls; accuracy dips |
| M1 phobert-base | 0.92–0.94 | 0.92–0.94 | **0.78–0.84** | 0.45–0.58 | Consistent with published UIT-VSFC results |
| M1 + Tier A/C | 0.92–0.94 | 0.92–0.94 | **0.80–0.86** | 0.52–0.65 | Imbalance handling is where S4 is won |

**The central observation this project is built on:** between B1 and M1 the *weighted* F1 moves ~4 pp —
unremarkable — while *macro* F1 moves ~15 pp. Same models, same data; one framing is interesting and one
is not. Published figures of "92–94% F1" on UIT-VSFC are weighted, and a classifier that never predicts
neutral still scores ~0.93 weighted. That is the whole argument for the headline metric choice.

### Topic (4-class: lecturer, training program, facility, others)

Class distribution is **measured in Phase 0** — it is not reliably documented upstream and must not be
assumed. Two properties are known in advance and both lower the expected ceiling: inter-annotator
agreement was only **71.07%** (versus 91.20% for sentiment), and `others` is a catch-all, which makes its
boundary intrinsically fuzzy.

| Model | Accuracy | Weighted F1 | **Macro-F1** |
|---|---|---|---|
| B1 tfidf-word + LR | 0.84–0.88 | 0.83–0.87 | **0.55–0.68** |
| M1 phobert-base | 0.88–0.92 | 0.88–0.92 | **0.70–0.80** |

Expect `facility` (small) and `others` (fuzzy) to be the two weak classes, and expect the topic ceiling to
sit below the sentiment ceiling — **because of the annotation, not the model.** Saying so in the write-up,
with the 71% IAA as evidence, is a stronger result than quietly reporting a lower number.

### Latency (reference laptop CPU, batch = 1)

| Configuration | p95 | Note |
|---|---|---|
| L0 torch fp32, pad to max_length | 80–250 ms | Depends heavily on the `max_length` chosen in Phase 0 |
| L1 + dynamic padding | 25–80 ms | Short sentences → potentially the single largest win |
| L3 onnx fp32 + graph opt | 20–60 ms | |
| L4 onnx int8 dynamic | **0.5×–2.5× vs L3** | Genuinely two-sided: without AVX512-VNNI this can be *slower* than FP32 |
| L5 onnx int8 static | 1.8×–3.0× vs L3 | |
| L6 openvino int8 | 2×–5× vs L3 | Typically the best on Intel CPUs |
| VnCoreNLP segmentation alone | 5–50 ms/sentence | JVM call. **Could exceed the model's own inference time** |

The last row is the one to measure first in Phase 3. If segmentation costs more than inference, the
ablation and the optimization are the same experiment, and the project's most quotable result comes out of
Week 4 rather than Week 7.

---

### Pre-registration scorecard — Gate G1

The expected ranges above were written before any experiment ran. Scoring them honestly:

| Prediction | Predicted | Measured | Verdict |
|---|---|---|---|
| B1 word + LR, sentiment macro-F1 | 0.60–0.68 | **0.737** | **Falsified — too low by ~0.06** |
| B2 char + LR, sentiment macro-F1 | 0.63–0.70 | **0.753** | **Falsified — too low** |
| B2 > B1 on sentiment | yes | +0.016 | Confirmed, smaller than expected |
| Best baseline, sentiment macro-F1 | 0.66–0.72 | **0.782** | **Falsified — too low by ~0.06** |
| B1 word + LR, topic macro-F1 | 0.55–0.68 | **0.749** | **Falsified — too low** |
| `facility` and `others` both weak | both weak | facility 0.921, others 0.493 | **Half wrong** |
| Neutral F1 for a linear model | 0.05–0.30 | 0.353 → 0.503 tuned | Under-estimated |
| Majority-class floor, sentiment | ~0.22 | 0.225 | Confirmed |
| Majority-class floor, topic | — | 0.210 | — |

**Diagnosis: the baselines were systematically under-predicted, and the cause is identifiable.** The
estimates were anchored on the published *literature* framing of UIT-VSFC as a hard, noisy corpus of
student feedback. Gate G0 then measured the opposite ([DATA_CARD § 6](DATA_CARD.md#6-surface-and-linguistic-profile--measured)):
the corpus is pre-lowercased, pre-tokenized, 99.86% diacritized and 99.84% teencode-free, with a
median length of 11 syllables and highly formulaic phrasing. That is close to an ideal setting for
TF-IDF, and the expected ranges were not revised after G0 measured it. **The lesson is sequencing:
pre-registered ranges should be re-derived once the data profile is known, and the revision recorded
— not left stale and then quietly forgotten when the results arrive.**

**Consequence for the success criteria.** S3 required PhoBERT to beat the tuned baseline by
**≥ +0.08 macro-F1**. Against a 0.782 baseline that demands ≥ 0.862, while the best published
macro-F1 on this dataset is ≈0.83. **S3 as written is very likely unreachable, and it was set against
a baseline estimate that is now known to be wrong.** Revised in ADR-008 rather than silently dropped.

---

## 5. Result tables to fill

Copy these into the final report. Leave `—` where a run was not done; never leave a blank.

### 5.1 Baseline ladder — MEASURED at Gate G1 (dev split, seed 42)

All selection on dev; test untouched. Every row is a `run_id` in `results/registry.csv`.

#### Sentiment

| Model | Acc | Weighted F1 | **Macro-F1** | F1 neg | **F1 neu** | F1 pos |
|---|---|---|---|---|---|---|
| B0 majority | 0.509 | 0.343 | **0.225** | 0.000 | 0.000 | 0.674 |
| B0b stratified random | 0.474 | 0.474 | **0.347** | 0.476 | 0.056 | 0.510 |
| B1 word 1-2g + LR | 0.910 | 0.902 | **0.737** | 0.921 | 0.353 | 0.936 |
| B1 + tuned priors | 0.900 | 0.904 | **0.772** | 0.913 | 0.468 | 0.936 |
| B2 char_wb 3-5g + LR | 0.905 | 0.901 | **0.753** | 0.918 | 0.410 | 0.930 |
| B2 + tuned priors | 0.888 | 0.894 | **0.776** | 0.908 | **0.503** | 0.918 |
| B3 word+char union + LR | 0.911 | 0.906 | **0.755** | 0.925 | 0.403 | 0.936 |
| **B3 + tuned priors** | 0.905 | 0.906 | **0.782** | 0.919 | 0.497 | 0.931 |
| B4 union + LinearSVC | 0.912 | 0.907 | **0.764** | 0.926 | 0.432 | 0.933 |
| B5 union + LR, class-weighted | 0.895 | 0.899 | **0.768** | 0.907 | 0.466 | 0.931 |
| B5 + tuned priors | 0.896 | 0.901 | **0.774** | 0.913 | 0.483 | 0.927 |

> ⚠️ **Corrected at ADR-015.** The `+ tuned priors` rows above are **optimistically biased**: the
> priors were fitted on dev and scored on dev. Cross-fitted (honest) values:
>
> | Model | Untuned | Fit = eval | **Cross-fitted** | Bias |
> |---|---|---|---|---|
> | B1 | 0.7366 | 0.7720 | 0.7578 | +0.0143 |
> | B2 | 0.7526 | 0.7761 | 0.7582 | +0.0178 |
> | **B3** | 0.7547 | 0.7823 | **0.7708** | +0.0116 |
> | B5 | 0.7679 | 0.7745 | **0.7390** | +0.0355 |
>
> **Honest sentiment baseline: 0.7708** (B3 + cross-fitted priors). Note B5 — correcting for
> imbalance twice (class weights *and* priors) lands *below* its own untuned score.

#### Topic

| Model | Acc | Weighted F1 | **Macro-F1** | F1 lecturer | F1 train_prog | F1 facility | **F1 others** |
|---|---|---|---|---|---|---|---|
| B0 majority | 0.727 | 0.612 | **0.210** | 0.842 | 0.000 | 0.000 | 0.000 |
| B0b stratified random | 0.539 | 0.543 | **0.237** | 0.705 | 0.151 | 0.057 | 0.037 |
| B1 word 1-2g + LR | 0.877 | 0.870 | **0.749** | 0.933 | 0.758 | 0.881 | 0.423 |
| B1 + tuned priors | 0.879 | 0.875 | **0.770** | 0.932 | 0.761 | 0.913 | 0.474 |
| B2 char_wb 3-5g + LR | 0.858 | 0.852 | **0.734** | 0.919 | 0.705 | 0.862 | 0.452 |
| B2 + tuned priors | 0.863 | 0.857 | **0.754** | 0.919 | 0.710 | 0.890 | **0.497** |
| B3 word+char union + LR | 0.871 | 0.867 | **0.753** | 0.928 | 0.740 | 0.864 | 0.480 |
| **B3 + tuned priors** | 0.875 | 0.872 | **0.774** | 0.928 | 0.754 | **0.921** | 0.493 |
| B4 union + LinearSVC | 0.880 | 0.874 | **0.768** | 0.933 | 0.753 | 0.908 | 0.479 |
| B5 union + LR, class-weighted | 0.840 | 0.847 | **0.744** | 0.904 | 0.734 | 0.890 | 0.449 |
| B5 + tuned priors | 0.878 | 0.872 | **0.765** | 0.930 | 0.754 | 0.914 | 0.461 |

> ⚠️ **Corrected at ADR-015.** Cross-fitted, topic prior tuning gains nothing (B3: 0.7529 untuned →
> 0.7516 cross-fitted). The **honest topic baseline is B4 LinearSVC at 0.768**, which never used
> priors at all.

#### Four findings from the ladder

1. ~~**Threshold tuning is the single largest lever.**~~ **RETRACTED — see ADR-015.** The +0.035
   was measured with priors fitted and scored on the same data. Cross-fitted, TF-IDF keeps a smaller
   real gain (+0.016 on B3) and **PhoBERT keeps none at all** (−0.001 raw, −0.004 segmented, neither
   significant). The diagnostic conclusion drawn from it — "the neutral failure is a decision-rule
   problem" — does not hold for the transformer, and Phase 4 Tier A is re-scoped to training-time
   methods only. Root cause: 73 neutral dev examples are too few to fit a threshold that transfers,
   the same bottleneck that makes dev underpowered for significance (ADR-013).

2. **Class weighting and threshold tuning are near-substitutes, not additive.** B5 (class-weighted,
   0.768) lands within noise of B1 + priors (0.772), and stacking them (B5 + priors, 0.774) adds
   almost nothing over either alone. They attack the same problem from opposite ends of training.

3. **Character n-grams help sentiment (+0.016) and hurt topic (−0.015).** The asymmetry is
   explicable rather than noise: topic classification keys on distinctive *content vocabulary*
   (`phòng học`, `wifi`, `giáo trình`), where word features are precise and character n-grams blur
   morpheme boundaries; sentiment keys on *polarity and modality markers*, where sub-syllable
   generalization pays. Worth stating because it contradicts the usual "char n-grams always help
   noisy text" heuristic — which, given this corpus is not noisy ([DATA_CARD § 6](DATA_CARD.md#6-surface-and-linguistic-profile--measured)), is exactly what one should expect on reflection.

4. **`facility` is easy; `others` is hard.** The pre-registration guessed both minority classes would
   be weak. Measured, `facility` reaches F1 0.921 on 70 dev examples — its vocabulary is highly
   distinctive — while `others` tops out at 0.493. `others` is a catch-all with no vocabulary of its
   own, which is the same fact the 71.07% topic IAA reports from the annotator's side.

### 5.2 PhoBERT reproduction — MEASURED at Gate G2 (dev split, 5 seeds)

`vinai/phobert-base`, condition P0 (raw), `max_length` 96, batch 32, lr 2e-5, 4 epochs, fp16,
checkpoint selected on dev macro-F1. Trained locally on the RTX 3050 (ADR-009), 69 s/epoch.

| Task | **Macro-F1** | Weighted F1 | Accuracy | Bal. acc | MCC | best epochs |
|---|---|---|---|---|---|---|
| Sentiment | **0.8436 ± 0.0079** | 0.9427 ± 0.0020 | 0.9449 ± 0.0021 | 0.8189 | 0.8974 | 4,2,4,4,2 |
| Topic | **0.7971 ± 0.0018** | 0.8889 ± 0.0025 | 0.8906 ± 0.0028 | 0.7843 | 0.7496 | 4,3,3,3,4 |

#### Versus the tuned baseline (B3 + priors)

Baselines are the **cross-fitted honest** values (ADR-015), not the optimistic ones first reported.

| Task | Honest baseline | PhoBERT (raw) | PhoBERT (segmented) | Δ vs baseline |
|---|---|---|---|---|
| Sentiment | 0.7708 (B3 + cross-fitted priors) | 0.8436 | **0.8670** | **+0.096** |
| Topic | 0.768 (B4 LinearSVC) | 0.7971 | *not yet run* | **+0.029** |

Per class, which is where the story actually is:

| Task | Class | Baseline F1 | PhoBERT F1 | Δ |
|---|---|---|---|---|
| Sentiment | negative | 0.919 | 0.957 | +0.038 |
| Sentiment | **neutral** | 0.497 | **0.614** | **+0.117** |
| Sentiment | positive | 0.931 | 0.960 | +0.029 |
| Topic | lecturer | 0.928 | 0.941 | +0.013 |
| Topic | training_program | 0.754 | 0.775 | +0.021 |
| Topic | **facility** | **0.921** | 0.905 | **−0.016** |
| Topic | others | 0.493 | 0.568 | +0.075 |

#### Literature reconciliation

| Source | Metric reported | Value | Ours (dev) |
|---|---|---|---|
| Nguyen et al. 2018 (MaxEnt) | weighted F1 | ~0.88 | 0.943 |
| Bi-LSTM + Word2Vec | weighted F1 | 0.92 | 0.943 |
| Recent PhoBERT work | weighted F1 / acc | ~0.94 / 94.5% | 0.943 / 0.945 |
| BamiBERT (2026) | **macro-F1** | **0.8341** | **0.8436** |

**Reproduction confirmed.** Our weighted F1 (0.943) and accuracy (0.945) sit inside the published
range, and our macro-F1 (0.844) is slightly above the only published macro figure. The 10-point gap
between the two columns is not a discrepancy — it is the same model measured two ways, which is the
thesis of this project stated as a number rather than an argument.

#### Four observations

1. **PhoBERT earns its place on the minority class, and almost nowhere else.** Weighted F1 moves
   +0.037 over the baseline; macro-F1 moves +0.062; neutral F1 moves **+0.117**. A reader looking only
   at weighted F1 would conclude the transformer was barely worth the GPU.

2. **Sentiment seed variance is 4× topic's** (std 0.0079 vs 0.0018). The cause is structural: dev has
   73 neutral examples, so one third of the macro average rests on a class where a handful of
   flipped predictions moves F1 by points. This is the concrete justification for the 5-seed policy —
   on sentiment, any claimed gain under ~0.008 is indistinguishable from the seed.

3. **TF-IDF beats PhoBERT on `facility`** (0.921 vs 0.905). Not noise: it exceeds topic's seed std by
   9×. `facility` is the class with the most distinctive vocabulary (`phòng học`, `máy lạnh`, `wifi`,
   `máy chiếu`), and distinctive vocabulary is exactly what TF-IDF represents best. A contextual model
   has nothing to add and a little to lose. Worth reporting because it is the kind of result that
   gets quietly dropped from a results table.

4. **Checkpoint selection matters and is visible.** Best epochs varied 2–4 across seeds; two sentiment
   seeds peaked at epoch 2 and then declined. A fixed 4-epoch schedule without dev-macro-F1 selection
   would have shipped a worse model for 40% of seeds.

### 5.3 Word-segmentation ablation — MEASURED at Gate G3 (sentiment, dev, 5 seeds each)

**Hypothesis H2 is falsified on both axes.** It predicted that segmentation would not help accuracy
and would dominate p95 latency. Both halves are wrong, and the way they are wrong is the finding.

#### Accuracy

| ID | Condition | **Macro-F1** | Weighted F1 | Accuracy | Neutral F1 | Δ macro vs P0 |
|---|---|---|---|---|---|---|
| P0 | raw (no segmentation) | 0.8436 ± 0.0079 | 0.9427 | 0.9449 | 0.6139 ± 0.0222 | *ref* |
| **P1** | **VnCoreNLP RDRSegmenter** | **0.8670 ± 0.0072** | 0.9529 | 0.9545 | **0.6680 ± 0.0184** | **+0.0234** |
| P2 | underthesea | 0.8618 ± 0.0063 | 0.9506 | 0.9524 | 0.6560 ± 0.0178 | +0.0182 |
| P2b | pyvi | 0.8643 ± 0.0098 | 0.9512 | 0.9531 | 0.6628 ± 0.0291 | +0.0207 |

P0 macro-F1 range **[0.8355, 0.8566]**; P1 range **[0.8598, 0.8751]** — **no overlap**. All five
segmented seeds beat all five unsegmented seeds.

#### Relation to the published result

> **Corrected (ADR-018).** An earlier version described this as refuting
> [arXiv:2301.00418](https://arxiv.org/abs/2301.00418). That was a misreading: the paper's
> conclusion is conditional — segmentation may be unnecessary for *traditional classifiers* and
> **is necessary** for deep-learning models using BPE. PhoBERT is the latter, so this measurement
> **replicates** the paper. What it adds is a quantified effect size, a per-class breakdown and a
> per-segmenter latency cost.


Measured four ways, because the size of the effect depends on the metric:

| Metric | P0 → P1 | Effect |
|---|---|---|
| Accuracy | 0.9449 → 0.9545 | **+0.96 pp** |
| Weighted F1 | 0.9427 → 0.9529 | **+1.02 pp** |
| **Macro-F1** | 0.8436 → 0.8670 | **+2.34 pp** |
| **Neutral F1** | 0.6139 → 0.6680 | **+5.42 pp** |

The effect is **5.6x larger on the minority class** than on the aggregate. That is the project's
own thesis — metric choice determines what a result looks like — showing up inside a preprocessing
ablation, and it is the part worth carrying into the write-up.

#### Significance — and a methodological problem worth its own entry

Two instruments disagree, and the disagreement is informative:

| Test | Result |
|---|---|
| Paired t-test over seeds (n=5, paired by seed) | +0.0234, sd 0.0061, **t = 8.58, p = 0.0010**, Cohen d = 3.84, 5/5 positive |
| Per-seed paired bootstrap on dev | **1/5 seeds** significant; BH-FDR keeps 1 |

Not a contradiction — they estimate different variances. The bootstrap estimates *evaluation-set
sampling* variance; the t-test estimates *training* variance. The dev set has **73 neutral examples**
carrying one third of the macro average, so resampling it moves macro-F1 by **±0.027** — wider than
the +0.023 effect being tested.

**The dev set is underpowered for within-run macro-F1 comparisons at this project's effect sizes.**
Even the test set (167 neutral) would only narrow the half-width to ~0.019. The seed-level paired
test is the correct instrument here, and it is adopted as such in ADR-013.

#### Latency — the premise that was off by two orders of magnitude

Per-sentence, batch = 1 (the serving workload), reference CPU at 44 °C idle, JVM warm:

| Backend | p50 | **p95** | p99 | Throughput |
|---|---|---|---|---|
| none | 0.000 ms | 0.000 ms | 0.001 ms | — |
| **pyvi** | 0.105 ms | **0.311 ms** | 0.555 ms | 8,207/s |
| VnCoreNLP | 0.255 ms | **0.606 ms** | 0.960 ms | 5,730/s |
| underthesea | 0.368 ms | **1.192 ms** | 2.272 ms | 2,212/s |

Against the model it feeds — PhoBERT-base FP32 on the same CPU, batch 1, 6 threads:

| Configuration | p50 | **p95** | vs 256 |
|---|---|---|---|
| pad to `max_length` 256 (model default) | 154.4 ms | **177.5 ms** | — |
| pad to `max_length` 96 (Gate G0 decision) | 79.3 ms | **90.2 ms** | 1.97× |
| **dynamic padding** | 39.1 ms | **50.8 ms** | **3.49×** |

**Segmentation is 1.2% of end-to-end p95 (VnCoreNLP) or 0.6% (pyvi).** H2 predicted it would exceed
the transformer's own cost. It is two orders of magnitude below it.

#### Serving decision: **pyvi**

| | VnCoreNLP | pyvi |
|---|---|---|
| Macro-F1 | 0.8670 ± 0.0072 | 0.8643 ± 0.0098 (**−0.0027, well inside seed std**) |
| p95 | 0.606 ms | **0.311 ms** |
| Runtime dependency | JVM (~180 MB in image) | pure Python |
| Known hazard | dies on any path containing a space (ADR-010) | none |

Segmenter *choice* does not matter — the P1/P2/P2b spread (0.005) sits inside the seed std. Whether
to segment *at all* matters (+0.023, p = 0.001). So the right move is to take the accuracy and pay
the smallest possible price: **pyvi**, which removes the JVM, ~180 MB of image, and the space-in-path
defect at a cost of −0.003 macro-F1 that no test can distinguish from noise.

#### Two free wins already banked, before any quantization

`max_length` 96 from Gate G0 and dynamic padding together give **3.49×** on p95 (177.5 → 50.8 ms).
End-to-end with pyvi is ≈ **51.1 ms p95**, which already meets the S5 *minimum* (≤ 60 ms) with no
ONNX export and no INT8. The S5 target (≤ 30 ms) is what Phase 6 must earn.

### 5.4 Improvement ladder

| Tier | Run ID | Change | Macro-F1 mean ± std | Δ vs previous best | > seed std? | Kept? |
|---|---|---|---|---|---|---|
| — | | phobert-base reference | | *ref* | — | ✓ |
| A | | class weights | | | | |
| A | | focal loss γ=2 | | | | |
| A | | logit adjustment | | | | |
| A | | threshold tuning | | | | |
| B | | LLRD | | | | |
| B | | lr/epoch sweep best | | | | |
| C | | label smoothing | | | | |
| C | | R-Drop | | | | |
| C | | FGM adversarial | | | | |
| D | | back-translation aug | | | | |
| D | | teencode/diacritic aug | | | | |
| D | | label-noise audit | | | | |
| E | `p4-sent-phobert-base-seg_pyvi-base` | **phobert-base (135M)** | **0.8643 ± 0.0092** | *ref* | — | ✅ **ship** |
| E | `p4-sent-phobert-large-seg_pyvi-base` | phobert-large (368M), Kaggle T4 | 0.8560 ± 0.0036 | −0.0083 | no (1.2 std) | ✗ |
| E | `p4-sent-xlmr-base-seg_pyvi-base` | xlmr-base (277M), Kaggle T4 | 0.8403 ± 0.0063 | −0.0240 | yes, **worse** | ✗ |
| E | | phobert-base-v2 | | | | not run |
| E | | ViSoBERT | | | | not run |
| E | | CafeBERT (560M) | | | | **dropped** — ADR-016 |
| E | | multi-task (2 heads) | | | | |
| F | | seed ensemble | | | | |
| F | | model soup | | | | |

The **"> seed std?"** column is the point of the table. A +0.004 macro-F1 gain against a ±0.011 seed std is
not an improvement, and marking it as one is the most common way these projects go wrong.

### 5.5 Robustness (Phase 5)

| Perturbation | Macro-F1 | Δ vs clean | Most-affected class |
|---|---|---|---|
| Clean test | | *ref* | |
| De-diacritized | | | |
| Teencode-injected | | | |
| Char noise 5% | | | |
| Char noise 10% | | | |
| Negation probe set (accuracy) | | | |

Repeat the whole table after the Phase 5 mitigation, and report **both** clean and perturbed columns so
that any robustness-for-accuracy trade is visible rather than hidden.

### 5.6 Inference benchmark (Phase 6, reference CPU)

| Step | Configuration | Size (MB) | p50 (ms) | **p95 (ms)** | p99 (ms) | Throughput b=32 (req/s) | Macro-F1 | Δ F1 (pp) | Within budget? |
|---|---|---|---|---|---|---|---|---|---|
| L0 | torch fp32, pad max | | | | | | | *ref* | — |
| L1 | + dynamic padding | | | | | | | | |
| L2 | + thread tuning | | | | | | | | |
| L3 | onnx fp32 + O3 | | | | | | | | |
| L4 | onnx int8 dynamic | | | | | | | | |
| L5 | onnx int8 static | | | | | | | | |
| L6 | openvino int8 | | | | | | | | |
| L7 | + no segmentation | | | | | | | | |
| L8 | distilled student | | | | | | | | |

**Reference machine** — fill once, cite everywhere: CPU `[model]`, `[n]` cores / `[m]` threads,
AVX2 `[y/n]`, AVX512-VNNI `[y/n]`, RAM `[n]` GB, OS `[…]`, power plan `[…]`, AC `[y/n]`,
ORT `[version]`, threads `[n]`.

---

## 6. Experimental hygiene rules

1. **The test set is evaluated at gates only.** Every test evaluation is appended to
   `results/test_evaluations.log` with date, run id, and the reason. A growing log is a warning sign, and
   it being visible is the point.
2. **Selection happens on dev. Always.** Including threshold tuning, early stopping, and model choice.
3. **5 seeds or it is not a result.** Three seeds are acceptable during exploration; anything that reaches
   the final tables is re-run at five.
4. **One axis per comparison.** If two things changed, it is not an ablation.
5. **Negative results are committed,** with the same detail as positive ones. Deleting a failed run from
   the registry is falsifying the record.
6. **Every figure regenerates from the registry** via a script in `src/vifeedback/evaluation/`. No
   hand-made plots — a number that cannot be regenerated cannot be trusted at review time.
7. **`env.json` is written on every run:** git SHA, package versions, CPU/GPU, and seed. Latency numbers
   without an `env.json` are anecdotes.
