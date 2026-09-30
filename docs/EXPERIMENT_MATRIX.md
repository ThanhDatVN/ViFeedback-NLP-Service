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
| A | `p8-sent-…-logit-adjust` | logit adjustment τ=1 | → § 5.7 (Cycle 1 H1) | | | |
| A | `p8-sent-…-crt` | balanced classifier re-training (cRT) | → § 5.7 (Cycle 1 H1) | | | |
| A | | threshold tuning | cross-fitted gain 0 (ADR-015); Cycle 0 boundary test +0.004 | | no | ✗ |
| B | | LLRD | | | | |
| B | | lr/epoch sweep best | | | | |
| C | | label smoothing | | | | |
| C | | R-Drop | | | | |
| C | | FGM adversarial | | | | |
| D | | back-translation aug | | | | |
| D | `p8-sent-…-aug-diac-teen` | teencode/diacritic aug, 30% exposure | → § 5.7 (Cycle 1 H2) | | | |
| D | | label-noise audit | OOF ranking + 160-row audit sheet ready; human audit pending (Study A) | | | |
| E | `p4-sent-phobert-base-seg_pyvi-base` | **phobert-base (135M)** | **0.8643 ± 0.0092** | *ref* | — | ✅ **ship** |
| E | `p4-sent-phobert-large-seg_pyvi-base` | phobert-large (368M), Kaggle T4 | 0.8560 ± 0.0036 | −0.0083 | no (1.2 std) | ✗ |
| E | `p4-sent-xlmr-base-seg_pyvi-base` | xlmr-base (277M), Kaggle T4, **pyvi input** | 0.8403 ± 0.0063 | −0.0240 | yes, **worse** | ✗ — confounded: XLM-R was never pretrained on segmented text (ADR-019) |
| E | `p8-sent-xlmr-base-raw-base` | xlmr-base, **raw input**, Kaggle | 0.8499 ± 0.0048 | −0.0144 | yes, worse | ✗ — +0.0097 over its pyvi run (H3); the rest of the gap is the model |
| E | | phobert-base-v2 | | | | not run |
| E | | ViSoBERT | | | | not run |
| E | | CafeBERT (560M) | | | | not run, for cost — a budget decision, not a finding (ADR-019) |
| E | | multi-task (2 heads) | | | | |
| F | | seed ensemble | | | | |
| F | | model soup | | | | |

The **"> seed std?"** column is the point of the table. A +0.004 macro-F1 gain against a ±0.011 seed std is
not an improvement, and marking it as one is the most common way these projects go wrong.

### 5.5 Robustness — MEASURED in Cycle 0 (validation, deployed checkpoint)

Suite version 1 (`evaluation/robustness.py`). Perturbations applied to raw text, then segmented with pyvi.
Paired on the same 1,583 sentences; 95% bootstrap CI over sentences. **Validation, not test:** the test
split is reserved for the closing gate, when these suites run once, frozen.

| Perturbation | Changed | Macro-F1 | Δ vs clean [95% CI] | Neutral recall (clean 0.562) | Predicted neutral (clean 3.2%) |
|---|---:|---:|---|---:|---:|
| Clean validation | — | 0.8634 | *ref* | 0.562 | 3.2% |
| `nodiacritic` | 99.7% | **0.268** | −0.595 [−0.634, −0.555] | 0.795 ⚠ | **64.1%** |
| `nodiacritic-50` | 97.2% | 0.652 | −0.211 [−0.252, −0.175] | 0.589 | 11.9% |
| `teencode-30` | 19.3% | 0.848 | −0.015 [−0.029, −0.005] | 0.548 | 3.5% |
| `teencode-100` | 49.4% | 0.821 | −0.042 [−0.067, −0.017] | 0.589 | 4.5% |
| `charnoise-5` | 79.1% | 0.784 | −0.079 [−0.111, −0.050] | 0.452 | 3.7% |
| `charnoise-10` | 92.3% | 0.769 | −0.094 [−0.133, −0.058] | 0.521 | 5.1% |
| Negation probe, positive → negated (36 pairs) | — | pair accuracy **1.00** | — | — | — |
| Negation probe, negative → negated (8 pairs, labels arguable) | — | pair accuracy 0.25 | — | — | — |

⚠ Neutral recall *rises* without diacritics because predictions collapse into neutral. Read alone, that
column would call a broken pipeline an improvement.

The mitigation run is Cycle 1 H2 (§ 5.7): both the clean and the perturbed columns are reported, so a
robustness-for-accuracy trade stays visible.

### 5.6 Inference benchmark (Phase 6, reference CPU)

| Step | Configuration | Size (MB) | p50 (ms) | **p95 (ms)** | p99 (ms) | Throughput b=32 (texts/s) | Macro-F1 | Δ F1 (pp) | Within budget? |
|---|---|---|---|---|---|---|---|---|---|
| L0 | torch fp32, pad to 96 | 540 | 117.4 | 120.5 | — | 12.0 | 0.8672 | *ref* | — |
| L1 | + dynamic padding | 540 | 46.7 | 64.1 | — | 24.1 | 0.8672 | 0.00 | ✓ |
| L2 | + thread tuning | | | | | | | | |
| L3 | onnx fp32, EXTENDED offline + ALL at load — **released and verified** (ADR-020/022) | 540 | **15.5** | **33.6** | — | 24.9 | 0.8672 (= PyTorch fp32) | 0.00 | ✓ parity 8.2e-5 |
| L4 | onnx int8 dynamic | 136 | 9.0 | 20.4 | — | 40.9 | 0.8370 | −3.02 | ✗ blocked: neutral F1 0.672 → 0.584 (ADR-022). 1.7× faster than L3: H3 falsified |
| L5 | onnx int8 static (per-tensor MinMax) | 136 | — | | | | 0.342 | −52 | ✗ blocked: breaks the model (ADR-022) |
| L6 | openvino int8 | | | | | | | | |
| L7 | + no segmentation | | | | | | | | |
| L8 | distilled student | | | | | | | | |

Throughput counts **texts** per second, not HTTP requests (R10). Measured 2026-09-27 on the reference
CPU, two steady passes in rotated order agreeing within 1% (`results/studies/latency/reference_cpu.json`).
Absolute times vary between laptop sessions; ratios reproduce (see STATUS § 4 P6).

**Reference machine** — fill once, cite everywhere: CPU `[model]`, `[n]` cores / `[m]` threads,
AVX2 `[y/n]`, AVX512-VNNI `[y/n]`, RAM `[n]` GB, OS `[…]`, power plan `[…]`, AC `[y/n]`,
ORT `[version]`, threads `[n]`.

### 5.7 Research Cycle 1 — DECIDED by pre-registered rules (validation)

Declared in `configs/experiments/cycle1.yaml` before any run; decisions computed by
`vifeedback study cycle1` (`results/studies/cycle1/decisions.json`). Run ids `p8-*`.

| Hypothesis | Declared prediction | Result (validation, seed-paired vs same-seed control) | Decision by the declared rule |
|---|---|---|---|
| **H1a** logit adjustment τ=1 | < 0.01 macro-F1; neutral recall ↑, precision ↓ | +0.0029 mean [−0.047, +0.053], 2/3 seeds; neutral P 0.79→0.61–0.70, R 0.56→0.66–0.67 | **Not advanced** (needs ≥ +0.005). Prediction confirmed |
| **H1b** balanced head retraining (cRT) | same | +0.0028 mean [−0.025, +0.031], 2/3 seeds; neutral P → 0.59–0.69, R → 0.64–0.77 | **Not advanced.** Seed spread 0.0012 vs 0.011 for CE (observation) |
| **H2** diacritic/teencode augmentation, 30% | ≥ 20% less `nodiacritic-50` degradation, ≤ 0.005 clean loss | **5 seeds** (finalist): degradation 0.217 → 0.124 (**−43%**); full no-diacritic 0.28 → 0.65; teencode −54%; clean mean +0.004, but per seed −0.006 … +0.030 (median −0.002); character noise unchanged (−4%) | **Supported at 5 seeds.** Targeted, in-family robustness at no material clean cost |
| **H3** XLM-R on raw text | raw − pyvi > 0 | Kaggle, same session, 5 seeds: **+0.0097** [+0.0014, +0.0179], 4/5 seeds, p = 0.032. Raw XLM-R 0.8499 vs PhoBERT-base 0.8643 | **Supported**: segmented input cost XLM-R ~0.01. Raw mean below the 0.8523 withdrawal threshold, so ADR-016's narrowed conclusion stands: ~40% of the gap was preprocessing, ~60% remains |
| **H4** shared encoder, λ=0.3 | helps / harms / no difference | sentiment +0.0012, topic +0.0005 (3 seeds) | **No material difference**: one model serves both tasks at half the inference cost, no loss detected |
| **H4** shared encoder, λ=1 | same | sentiment −0.0056 (0/3 seeds), topic +0.0031 (2/3) | **Negative transfer on sentiment** by the rule (interval still spans 0) |


Full discussion: [STATUS § 3](STATUS.md#3-cycle-1--declared-hypotheses-and-their-outcome).

### 5.8 Research Cycle 2 — track A, confirmed on new data

Declared in `configs/experiments/cycle2.yaml` (v1 ADR-025; v2 ADR-026, which froze the challenge set
by SHA-256 before any evaluation). No Cycle 2 decision uses the official test.

| Hypothesis | Data | Result | Decision by the declared rule |
|---|---|---|---|
| **H5** topic stacking, TF-IDF B4 × PhoBERT (out-of-fold meta-model) | validation | Stacked − PhoBERT 5-fold ensemble **−0.0102** [−0.0228, +0.0013]; facility 0.929 → 0.922, others 0.621 → 0.585 | **Not supported.** Descriptive: facility F1, TF-IDF − PhoBERT −0.021 [−0.060, +0.015]: the "TF-IDF wins facility" gap does not hold up |
| **H6** serve the H2-augmented model? | challenge set (305) | Typed noise (90 rows): accuracy 0.611 → 0.800, **+0.189** [+0.067, +0.311]; other rows 0.867 → 0.867 | **Switch** (ADR-027). Hidden by pooling: `teencode_typed` 0.975 → 0.875, `objective_neutral` 0.767 → 0.633 |
| **H7 pilot** Qwen3-1.7B, label likelihood, frozen prompt | validation | macro-F1 0.680 (0-shot), 0.655 / 0.653 (6-shot); neutral F1 0.29 / 0.26 / 0.26 vs encoder 0.66 | Pilot, not declared. Neutral precision 0.16–0.18: the LLM over-calls neutral |
| **H7 pilot** | challenge set | macro-F1 0.62 / 0.62 / 0.60 vs 0.77 (CE), 0.83 (augmented) | `objective_neutral` 0.97–1.00 (encoder 0.63–0.77); `mixed_aspect` 0.15–0.20 and `suggestion_cue` 0.40–0.60 (encoder 0.88–1.00) |
| **H7 declared** Qwen3-4B, zero-shot | challenge set (285) | macro-F1 0.819; neutral F1 0.901 vs 0.713 (CE): +0.187 [+0.106, +0.277] | Holm p 0.0004 with the API arm: `llm_better_on_neutral` |
| **H7 declared** Qwen3-4B, 6-shot (s1 / s2) | challenge set | macro-F1 0.843 / 0.851; neutral F1 0.810 / 0.846 | reported, not tested |
| **H7 declared** Qwen3-4B, 0 / 6-shot (s1 / s2) | validation | macro-F1 0.816 / 0.798 / 0.816 vs 0.864 (CE); neutral F1 0.547 / 0.507 / 0.554 vs 0.661 | below the encoder on UIT-VSFC |
| **H7 local arm** Qwen3-4B, zero-shot | NEU-ESC test (6,613) | macro-F1 0.475 vs 0.494 (CE s42), 0.462 (served s42): +0.014 [−0.001, +0.028]; gpt-4o-mini 0.604 | a local 4B model does not close the gap (`cycle3.yaml` v6) |
| **H7 API arm** gpt-4o-mini, zero-shot | challenge set (285 scored) | macro-F1 0.944 vs 0.774 (CE), 0.827 (augmented); neutral F1 0.955 | construction confound (v1 written in the prompt's convention) |
| **H7 API arm** gpt-4o-mini, zero-shot | NEU-ESC test (6,613) | macro-F1 0.604 vs 0.494 (CE s42), 0.462 (augmented s42); neutral F1 0.769 vs 0.526 / 0.474; USD 0.21 | LLM ahead on real off-domain posts, mostly neutral |

Challenge set accuracy per category (seed-42 encoders; Qwen3-1.7B zero-shot):

| Category | n | CE | Augmented (served) | Qwen3-1.7B |
|---|---:|---:|---:|---:|
| `code_switch` | 25 | 0.920 | 0.960 | 0.800 |
| `long_context` | 20 | 0.850 | 0.900 | 1.000 |
| `mixed_aspect` | 40 | 0.875 | 0.925 | 0.175 |
| `negation_pair` | 30 | 0.800 | 0.833 | 0.700 |
| `objective_neutral` | 30 | 0.767 | 0.633 | 0.967 |
| `suggestion_cue` | 25 | 1.000 | 1.000 | 0.600 |
| `suggestion_implicit` | 25 | 0.880 | 0.840 | 0.440 |
| `teencode_typed` | 40 | 0.975 | 0.875 | 0.825 |
| `unaccented_typed` | 50 | 0.320 | 0.740 | 0.380 |

Outputs: `results/studies/{topic_stacking,challenge,llm_reference}/`.

### 5.9 Research Cycle 3 — the served model on real input

Declared in `configs/experiments/cycle3.yaml` (v1 before the first run; v2 external evaluation; v3 S2a,
ADR-029; v4 S2b′; v5 confirmation on NEU-ESC, ADR-030). Challenge v1 is development data.

| Step | Data | Result | Decision |
|---|---|---|---|
| V1 CE vs augmented, 5 seeds (all reproduce exactly) | challenge v1 | teencode −0.035 (3/5, p = 0.09); short factual −0.087 (3/5, p = 0.02); contrast −0.075 (4/5, p = 0.004); unaccented +0.33 (5/5) | contrast drop confirmed (exploratory); others not |
| Case invariance | validation, first letter capitalized | served model 1.07% flips; lowercased 0% | service lowercases (rule met) |
| ViLexNorm invariance | 1,045 real comment pairs | 17% flips, CE ≈ augmented (p = 0.93) | measured |
| S2a lexicon from ViLexNorm train, 5 seeds | ViLexNorm test, validation, v1 | flips 0.169 → 0.177 (p = 0.13); validation +0.003 | gate not passed |
| S2b diacritic restoration | validation stripped, v1 | 0.686 → 0.857; v1 unaccented 0.74 → 0.86; no clean change | development passed |
| S3 out-of-scope score | validation vs 551 off-topic | Mahalanobis AUROC 0.949; energy 0.889; max-prob 0.862 | development choice: Mahalanobis |
| v5 S2b confirmation | NEU-ESC test, diacritics stripped (6,613) | served model 0.270 → 0.374 (+0.104 [+0.091, +0.118]); as written −0.0006 | **passed → served** (ADR-031) |
| v5 S2b′ confirmation | NEU-ESC contrast posts (414), 5 seeds | CE + R 0.440 vs augmented + R 0.437 (p = 0.68); overall +0.028 | **not passed** |
| v5 S3 confirmation | NEU-ESC off-topic posts (563) vs validation | Mahalanobis AUROC 0.977; energy 0.936; max-prob 0.920 | passed → served (ADR-031); confounded by institution (ADR-032) |
| S5 careful INT8 | train fidelity; validation | 178.5 MB; drop +0.0004, upper 0.0095 | not passed |

### 5.10 Research Cycle 4 — student text from other institutions

Declared in `configs/experiments/cycle4.yaml` (v1 before any run; v2 reporting-only, ADR-032; v3 B4′;
v4 S5′, ADR-035). Selection on NEU-ESC validation; NEU-ESC test once per rule, logged in
`results/studies/cycle4/neu_esc_test_uses.log`.

| Step | Data | Result | Decision |
|---|---|---|---|
| A1 served-pipeline latency | raw test text, 3 sessions | p95 26.9 ms (range 23.4–39.5); additions +2.2 ms; unaccented −4.2 ms | met (≤ 30 ms, additions ≤ 5 ms) |
| A2 `in_scope` within NEU-ESC | validation, in-scope (3,026) vs off-topic (279) | flagged 77.1% vs 84.2%; AUROC 0.573 | measures resemblance to UIT-VSFC, not topic (ADR-032) |
| A3 label policy | NEU-ESC validation, gold neutral in scope (2,054) | 67% called polar, 1,172 negative | input to owner decision 2 |
| **H8** selection (seed 42) | UIT-VSFC / NEU-ESC validation | control 0.864 / 0.459; mixed 0.832 / 0.750; sequential 0.596 / 0.741; two heads 0.872 / 0.533 (NEU-ESC head 0.761) | two heads chosen (cycle4.yaml) |
| **H8** confirmation (5 seeds) | NEU-ESC test in scope (6,050); UIT-VSFC validation | +0.111 [+0.102, +0.119]; UIT-VSFC −0.0045, neutral −0.010, stripped −0.0107; refitted out-of-scope AUROC 0.886, 31% flagged | **not passed** (ADR-033) |
| **B4′** scope detector | NEU-ESC validation (selection), test (rule) | TF-IDF logistic 0.921 vs feature logistic 0.869 (validation); test AUROC 0.922, in-scope flagged 7.3%, off-topic caught 72.3%, UIT-VSFC 0.2%; U4 0.918 | **passed → served** (ADR-034) |
| Latency with the detector | raw test text, 3 sessions (busy machine) | +0.5–0.8 ms over S0 within session (Mahalanobis: +1.3–3.1 ms) | cheaper; absolute p95 to re-measure idle |
| **E1** INT8 power check | held-out estimation; simulated acceptance | pooled power 92% (median bound −0.0008), UIT-VSFC alone 75%; calibrated (observed bound at the 82nd percentile) | S5′ declared (ADR-035) |
| **S5′** careful INT8 | UIT-VSFC + NEU-ESC validation (4,888) | pooled bound 0.0006; NEU-ESC drop −0.0044; neutral −0.0052; 178.5 MB; agreement 99.2% / 88.4% | **passed**; release gate: agreement 91.4% < 99%, batch-dependent → **not released** (ADR-036) |
| F1 runtime image | Docker build | 1,023 → 750 → **519 MB**; token ids identical on 49,141 texts | S8 met |
| F2 clean-clone reproduction | CI (`reproduce.yml`), Hub download | 2.3 min; validation 0.8644 reproduced | S10 met |

### 5.11 Research Cycle 5 — declared, to fill

Declared in `configs/experiments/cycle5.yaml` v1 (2026-09-29, before any run; H10) and v2 (2026-09-30,
before any H11 run; H11). Order chosen by the owner: H10 (real typing), H11 (distilled student), H12
(other institutions, declared once its data exists). Rows are filled from `results/studies/cycle5/`.

| Step | Data | Result | Decision |
|---|---|---|---|
| H10 selection (seed 42): one-sided vs symmetric consistency | ViLexNorm dev (837 pairs); UIT-VSFC validation | control 0.8644 / flips 0.135; one-sided 0.8541 (−0.0103) / 0.007; symmetric 0.8689 / 0.002 | symmetric chosen (one-sided not eligible) |
| **H10** confirmation (5 seeds) | ViLexNorm test (1,045 pairs, logged); UIT-VSFC, stripped, NEU-ESC validation | flips 16.6% → 0.3% (−0.164 [−0.180, −0.149]); UIT-VSFC −0.0016, neutral −0.0029, stripped −0.0065; **NEU-ESC −0.087**. Degenerate: 99.4% of ViLexNorm comments labelled negative in both forms; NEU-ESC predicted negative 58% → 76% | **not passed** (ADR-038) |
| H11 selection (seed 42) | UIT-VSFC / NEU-ESC validation | teacher-alternate 0.8614 / 0.4160; pretrained-first6 0.8730 / 0.4360 | pretrained-first6 chosen |
| **H11** confirmation (5 seeds each, teacher: the served recipe) | UIT-VSFC + NEU-ESC validation (4,888) | 185.1 MB; pooled drop 0.0024, bound 0.0078; UIT-VSFC drop 0.00001; neutral +0.0045; stripped 0.0032; parity 1.1e-5, batch-independent | **passed** (ADR-039) |
| H11 latency (idle condition waived) | raw test text, 3 sessions | S1 p95 median 12.3 ms vs 22.1 ms for the 12-layer model | **served** (ADR-040) |
| H12 other institutions | a new labelled sample (≥ 600 in-scope posts) | — | to declare; needs owner decision 2 and the data |
| H10b selection (seed 42) | ViLexNorm dev (837); UIT-VSFC validation | control agreement 0.828; anchored_orig 0.8626 (UIT −0.0027); anchored_both 0.8626 (UIT +0.0065); tie, first in declared order kept | anchored_orig |
| **H10b** confirmation (5 seeds each) | 1,500 fresh ViLexNorm pairs; validation guards | agreement +0.062 [+0.048, +0.075]; flips −0.061 [−0.073, −0.049]; label_tv 0.013; UIT +0.0014, neutral +0.003, stripped +0.0028, NEU-ESC −0.0039 | **passed** (ADR-042) |
| H10b closing gate | UIT-VSFC test, once (logged) | 0.8237 (5 seeds) vs 0.8296: −0.0059 [−0.019, +0.007]; seed 42 0.8208, neutral 0.550, no diacritics 0.623 | reported |
| H11 closing gate | UIT-VSFC test, once (logged) | student 0.8168 (5 seeds) vs 0.8296: −0.0129 [−0.0204, −0.0054], 0/5 seeds higher; seed 42 0.8175, neutral 0.545, no diacritics 0.609 | reported (ADR-040) |

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
