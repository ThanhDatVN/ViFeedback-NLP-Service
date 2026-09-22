# Evaluation Protocol

The contract every number in this project is held to. Written before the first experiment so that the
rules cannot be adjusted to fit a result.

---

## Table of contents

1. [Metrics](#1-metrics)
2. [Seed and variance policy](#2-seed-and-variance-policy)
3. [Statistical significance](#3-statistical-significance)
4. [Model selection protocol](#4-model-selection-protocol)
5. [Latency harness](#latency-harness)
6. [Error taxonomy](#error-taxonomy)
7. [Perturbation suites](#7-perturbation-suites)
8. [Software testing strategy](#software-testing-strategy)

---

## 1. Metrics

### Reported for every run, without exception

| Metric | Why it is in the list |
|---|---|
| **Macro-F1** | **Headline.** Unweighted mean of per-class F1 — the only metric here that the 4.3% neutral class can move |
| Weighted F1 | Comparability with published UIT-VSFC results, which are weighted |
| Accuracy | Universally understood; individually uninformative on this data |
| Per-class P / R / F1 / support | Where the actual behaviour lives |
| Balanced accuracy | Macro-recall; separates recall failures from precision failures |
| MCC | Robust single number for imbalanced multi-class; a useful sanity cross-check |
| Confusion matrix (counts + row-normalized) | Both forms — counts show volume, normalized shows rates |

### Why macro-F1 is the headline

On the real UIT-VSFC test split (1,409 / 167 / 1,590), a model that **never predicts neutral** but is
otherwise perfect scores:

| | Value |
|---|---|
| Accuracy | 0.947 |
| Weighted F1 | 0.922 |
| **Macro-F1** | **0.649** |

*(measured, not estimated; pinned by `tests/unit/test_metrics.py`)*

Three metrics, one model, and only one of them notices that an entire class is missing. Every headline in
this project is macro-F1, and the weighted number is always printed beside it so nobody can accuse the
project of picking the flattering metric in either direction.

### Reporting format (numbers below are illustrative, not results)

```
Sentiment · phobert-base · P1 · test
  Macro-F1      0.821 ± 0.009   [95% CI 0.804–0.838]
  Weighted F1   0.934 ± 0.004
  Accuracy      0.936 ± 0.004
  Per class     neg  P .95 R .94 F1 .945  (n=1409)
                neu  P .52 R .48 F1 .499  (n=  167)
                pos  P .96 R .97 F1 .965  (n=1590)
```

Supports shown for every class — a per-class F1 without its support invites the reader to over-interpret a
number computed on 167 examples.

---

## 2. Seed and variance policy

**Seeds:** `42, 1337, 2024, 7, 31337`. Fixed, listed, and never cherry-picked.

| Rule | |
|---|---|
| Exploration | 3 seeds acceptable |
| Any number in a final table | **5 seeds, mandatory** |
| Reporting | mean ± std, always; a bare number is not a result |
| Cherry-picking | Reporting a best-of-N seed as *the* result is falsification. If best-of-N is shown, it is labelled as such and the mean is shown beside it |

Seeding must cover Python `random`, NumPy, `torch` (CPU and CUDA), and the DataLoader worker seed. Set
`transformers.set_seed()` plus explicit `torch.backends.cudnn.deterministic`. Full determinism on GPU is
not always attainable; when it is not, say so rather than implying it.

**Why this matters here specifically:** fine-tuning BERT-family models on small datasets is a known source
of run-to-run instability, and the neutral class has ~458 training examples. A seed std of ±0.01–0.02
macro-F1 is entirely expected — which means any reported improvement smaller than that is indistinguishable
from noise, and the `> seed std?` column in
[EXPERIMENT_MATRIX § 5.4](EXPERIMENT_MATRIX.md#54-improvement-ladder) exists to enforce it.

---

## 3. Statistical significance

### Paired bootstrap (primary test)
For model A vs model B on the same test set:

1. Resample test indices with replacement, 10,000 times.
2. On each resample compute `macro_F1(A) − macro_F1(B)` using **the same indices for both models** — the
   pairing is what makes the test powerful on a 3,166-example set.
3. Report the mean difference and the 95% percentile interval.
4. Significant at α = 0.05 iff the interval excludes 0.

### McNemar's test (secondary)
On the paired correct/incorrect contingency table, for accuracy-level comparisons. Reported alongside the
bootstrap when the two disagree — and the disagreement itself is then worth a sentence.

### Applied to
- best baseline vs PhoBERT (objective O1 — the CV snippet's central claim);
- P0 vs P1 in the segmentation ablation (objective O3);
- champion vs each Phase 4 tier increment;
- FP32 vs quantized artifact (here the *desired* outcome is no significant difference).

### Multiple comparisons
Phase 4 runs ~20 comparisons. Two safeguards: all selection happens on **dev**, with test used only to
confirm the single champion; and where several test comparisons are reported together,
Benjamini–Hochberg FDR correction is applied and labelled. Running twenty tests and reporting the one that
reached p < 0.05 is not a finding.

---

## 4. Model selection protocol

```
train split   →  fit parameters
dev split     →  ALL decisions: hyperparameters, early stopping, checkpoint selection,
                 thresholds, preprocessing variant, model architecture, champion choice
test split    →  evaluated ONLY at phase gates, logged every time
```

**Test-set discipline.** Every test evaluation appends to `results/test_evaluations.log`:
`date, run_id, phase, reason`. Expected total for the whole project: **~8 evaluations** (one per gate,
plus the final tables). If that log reaches 30 entries, the test set has become a second dev set and the
final numbers are optimistically biased — and the visible log is what lets you notice this while there is
still time to say so.

**Checkpoint selection is on dev macro-F1**, never on dev loss and never on dev accuracy. With a 4%
minority class, selecting on loss or accuracy selects the checkpoint that has learned to ignore neutral.

---

## Latency harness

`src/vifeedback/inference/benchmark.py`. These rules are what separate a benchmark from a `time.time()`
call in a notebook.

### Protocol

1. **Isolation** — a dedicated process, no training running, no browser, AC power, power plan recorded.
2. **Warmup** — 200 iterations, discarded. First-call overhead (lazy init, JIT, allocator warmup, ORT
   arena growth) is large enough to dominate an unwarmed measurement.
3. **Timed** — 1,000 iterations with `time.perf_counter_ns()`.
4. **Realistic inputs** — sampled from the **actual test-set length distribution**, not a fixed-length
   dummy string. A benchmark on 256-token padded dummies measures a workload the service will never see,
   and it is exactly the configuration that flatters quantization.
5. **Repetitions** — the whole thing 5 times; report the **median of the five p95 values**. If the five
   disagree by more than 10%, the machine was throttling: re-run, and note it.
6. **Reported** — p50, p95, p99 at batch = 1; throughput at batch ∈ {1, 8, 32}; artifact size on disk;
   peak RSS.
7. **Split the measurement in two, and report both:**
   - **model-only** — forward pass alone;
   - **end-to-end** — normalization + segmentation + tokenization + forward + post-processing.

   Reporting only model-only latency is the standard way this benchmark gets quietly inflated, and in this
   project end-to-end is where the interesting finding lives: VnCoreNLP's JVM segmentation call may well
   exceed the transformer's own inference time.
8. **Thread configuration is recorded and swept** — `OMP_NUM_THREADS`, ORT `intra_op_num_threads` and
   `inter_op_num_threads`. On a laptop, more threads is frequently *slower* at batch = 1.
9. **`env.json` accompanies every benchmark**: CPU model, cores/threads, AVX2 and **AVX512-VNNI** flags,
   RAM, OS build, Python/torch/ORT versions, git SHA. A latency number without this file is not quotable.

### Accuracy budget
After every optimization step, macro-F1 is re-evaluated on the full test set. The pre-registered budget is
**≤ 0.5 pp macro-F1 loss** versus FP32. A step that exceeds it is reported in the table and **not shipped**.

---

## Error taxonomy

Used in Phase 5 to code ≥ 30 (target 60) errors. Categories are mutually exclusive by *primary* cause; a
secondary tag is allowed and recorded separately.

| Tag | Phenomenon | Vietnamese example | Why models fail |
|---|---|---|---|
| `NEG-SIMPLE` | Direct negation — 20.4% of the corpus; 84.3% of such sentences are gold-`negative`, so a unigram already captures it | *không tốt*, *chưa hay*, *chẳng hiểu gì* | Negation scope is a classic bag-of-words failure; TF-IDF cannot represent it at all |
| `NEG-DOUBLE` | Double / rhetorical negation flipping polarity | *không có gì để chê* ("nothing to criticize" → **positive**) | Surface negation cue points the wrong way |
| `NEG-COLLOQ` | Colloquial negation | *đâu có hay*, *có hay đâu* | Discontinuous negation; rare in pretraining corpora |
| `CONTRAST` | Contrastive discourse — polarity flips after the connective | *Thầy dạy nhiệt tình **nhưng** phòng học quá nóng* | Both polarities present; gold picks one |
| `MIXED` | Genuinely mixed polarity, one gold label | *Môn hay, giảng viên tốt, nhưng bài tập quá nhiều* | Structural annotation limitation (see [DATA_CARD L4](DATA_CARD.md#8-known-limitations)) |
| `TEENCODE`&nbsp;⚠ | Internet slang / abbreviation — **0.16% of this corpus**, so a *robustness* target, not an error-analysis one (ADR-007) | *k*, *ko*, *hok* (không) · *j* (gì) · *z*, *dz* (vậy) · *dc*, *đc* (được) · *bt* (bình thường) · *mn* (mọi người) · *ntn* (như thế nào) · *gv*, *sv* | Out-of-vocabulary for word-level features; fragments badly under BPE |
| `NODIACRITIC`&nbsp;⚠ | Diacritics omitted — **0.14% of this corpus**; measured via perturbation, not observation (ADR-007) | *thay day hay* → *thầy dạy hay* | Massive ambiguity: *ma* ↔ *mà/má/mã/mả/mạ*; PhoBERT's pretraining is fully diacritized |
| `TYPO` | Typos, elongation, repeated characters | *hayyyy*, *chán quáaaa*, *tôtt* | Rare subwords; elongation also carries *intensity*, which is lost when normalized away |
| `SARCASM` | Irony, sarcasm | *Hay quá, học xong chẳng nhớ gì* | Requires pragmatics; no lexical cue |
| `IMPLICIT` | Sentiment with no polarity-bearing words | *Học xong không biết áp dụng vào đâu* | Needs world knowledge |
| `SUGGESTION` | Imperative / request for change | *Nên tăng thời gian thực hành* | **Measured: 91.1% of sentences with `nên`/`cần`/`mong` are gold-`negative`** (ADR-007) — the guideline reads a request for change as implicit criticism. A model that calls these `neutral` has learned the intuitive rule, not the annotated one |
| `CODESWITCH` | Vietnamese–English mixing | *slide của thầy rất hay*, *deadline nhiều quá* | Mixed-script subwords |
| `LENGTH` | Very short (1–3 tokens) or very long inputs | *ok*, *tốt* | Too little signal / truncation |
| `GOLD-ARGUABLE` | The gold label is defensible-but-contestable | — | Not a model error. **Count it separately and honestly** — it bounds the achievable ceiling |
| `TOPIC-MULTI` | Multiple topics in one sentence | *Thầy dạy tốt nhưng phòng máy hỏng* (lecturer + facility) | Single-label task on multi-label content |
| `TOPIC-OTHERS` | `others` catch-all confusion | — | Fuzzy class boundary; consistent with the 71% topic IAA |

### Coding procedure
1. Sample **stratified by confusion-matrix cell**, over-sampling the informative cells
   (neutral→positive, neutral→negative, others→lecturer), not uniformly over errors.
2. Pass 1: code all cases.
3. Wait **≥ 24 hours**. Pass 2: re-code blind to pass 1.
4. Report your own **pass-1/pass-2 disagreement rate** as a self-consistency figure. It is a single number,
   it costs nothing, and it demonstrates annotation literacy more convincingly than the taxonomy itself.
5. Tally per category; select the top 3; design one mitigation for each; measure.

### Output schema — `results/error_analysis.csv`
`id, split, text, gold, pred, prob_gold, prob_pred, tag_primary, tag_secondary, pass1_tag, pass2_tag, note, mitigation_target`

---

## 7. Perturbation suites

Programmatic transformations of the **full test set**, converting "the model struggles with teencode" from
an impression into a number.

| Suite | Transformation | Reported |
|---|---|---|
| `nodiacritic` | Strip all diacritics from every sentence | Macro-F1 drop |
| `teencode` | Replace whole words with dictionary teencode variants, p = 0.3 | Macro-F1 drop |
| `charnoise-5` / `charnoise-10` | Random char swap / drop / duplicate at 5% / 10% | Macro-F1 drop |
| `negation-probe` | ~50 hand-built minimal pairs differing only by *không* | Pair accuracy: fraction where the model flips label correctly |

**Rules.** Perturbations apply to inputs only, never to gold labels. Each suite is seeded and
deterministic. The clean baseline is reported in the same table so the drop is unambiguous. The negation
probe set is handwritten, committed, and small enough to inspect by eye.

This suite is the highest-value-per-hour item in the project: it takes an afternoon, no GPU, and produces
a robustness table that essentially no other public UIT-VSFC project has.

---

## Software testing strategy

ML correctness and software correctness are different problems. Both are in scope, both run in CI.

### `tests/unit/` — pure functions
- teencode dictionary expansion, diacritic stripping, Unicode NFC normalization, whitespace cleanup;
- segmenter wrappers (mocked backends — no JVM in unit tests);
- **metric implementations cross-checked against `sklearn`** on synthetic cases, including the degenerate
  ones: a class with zero predictions, a class with zero support, a single-class input;
- bootstrap CI reproducibility under a fixed seed;
- config parsing and validation, including rejection of unknown keys.

### `tests/data/` — dataset integrity *(Phase 0; see [DATA_CARD § 7](DATA_CARD.md#9-validation-checks--all-passing-at-g0))*
Split sizes, label ranges, encoding, duplicates, cross-split leakage, class-distribution snapshot.

### `tests/contract/` — model behaviour
- **Golden predictions:** ~20 fixed Vietnamese inputs with expected labels, committed. Catches silent
  breakage in preprocessing, tokenizer, label mapping, or export.
- **Label-mapping test:** `id2label` round-trips correctly. A transposed label map produces a plausible-looking
  confusion matrix and wrong everything — worth one cheap test.
- **ONNX parity:** logits `np.allclose(torch, onnx, atol=1e-4)` for FP32.
- **Quantization parity:** label agreement ≥ 99.5% between FP32 and INT8 on the full test set, plus the
  macro-F1 budget assertion (≤ 0.5 pp).

### `tests/integration/` — the service
- FastAPI `TestClient`: happy path, batch path, empty string, very long input, wrong schema, oversized
  payload, malformed JSON;
- response schema stability (the contract external callers depend on);
- `/healthz` and `/readyz` behave differently before and after the model loads — a readiness probe that is
  green before the model exists is worse than no probe;
- container smoke test: `docker run` → `curl` → assert a correct label.

### `tests/perf/` — regression guard
A loose p95 threshold on the CI runner (shared, noisy hardware — the threshold catches 10× regressions,
not 10% ones). Real numbers come from the reference laptop only.

### CI pipeline

```
ruff check → ruff format --check → mypy → pytest (unit, data, contract, integration)
          → docker build → container smoke test → perf smoke test
```

Runs on push and PR. Coverage target **≥ 80%** on `src/vifeedback/`, excluding notebooks. Heavy tests
(full test-set contract checks) run on a `nightly` or manual trigger so the PR loop stays fast.

### What is deliberately *not* tested
Training convergence (non-deterministic and slow) and exact metric values (they change legitimately as the
work progresses). The registry and the gate reviews cover those; CI covers the code.
