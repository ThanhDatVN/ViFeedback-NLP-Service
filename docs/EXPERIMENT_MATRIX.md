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

## 5. Result tables to fill

Copy these into the final report. Leave `—` where a run was not done; never leave a blank.

### 5.1 Baseline ladder — task: `sentiment` / `topic` *(one table each)*

| Run ID | Model | Features | Acc | Weighted F1 | **Macro-F1** | F1 neg | F1 neu | F1 pos | Fit time |
|---|---|---|---|---|---|---|---|---|---|
| | B0 majority | — | | | | | | | |
| | B1 | word 1-2g | | | | | | | |
| | B2 | char_wb 3-5g | | | | | | | |
| | B3 | union | | | | | | | |
| | B4 LinearSVC | best | | | | | | | |
| | B5 + class weight | best | | | | | | | |
| | B5 + threshold-tuned | best | | | | | | | |

### 5.2 PhoBERT reproduction — 5 seeds

| Run ID | Task | Model | Preproc | Macro-F1 mean | std | 95% CI | Weighted F1 | Acc | Δ vs best baseline | p (paired bootstrap) |
|---|---|---|---|---|---|---|---|---|---|---|
| | sentiment | phobert-base | P1 | | | | | | | |
| | topic | phobert-base | P1 | | | | | | | |

**Literature reconciliation row** — state the published number, its metric definition, and your matching
metric side by side. A 10 pp "gap" that is really a macro-vs-weighted difference must never be presented
as a failure to reproduce.

### 5.3 Preprocessing ablation — sentiment, 5 seeds each

| ID | Preprocessing | Macro-F1 mean ± std | Δ vs P1 | p vs P1 | Preproc latency p95 (ms) | End-to-end p95 (ms) |
|---|---|---|---|---|---|---|
| P0 | raw | | | | ~0 | |
| P1 | RDRSegmenter | | *ref* | *ref* | | |
| P2 | underthesea | | | | | |
| P3 | P1 + NFC/cleanup | | | | | |
| P4 | P3 + teencode | | | | | |
| P5 | P3 + lowercase | | | | | |
| P6 | best + emoji | | | | | |

**Conclusion line to write explicitly:** *"Segmentation contributes `[Δ]` macro-F1 (p = `[p]`) at a cost of
`[ms]` ms p95, i.e. `[%]` of end-to-end latency. Decision for serving: `[keep/drop]`."*

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
| E | | phobert-base-v2 | | | | |
| E | | ViSoBERT | | | | |
| E | | XLM-R base | | | | |
| E | | phobert-large | | | | |
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
