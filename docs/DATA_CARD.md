# Data Card — UIT-VSFC

**Status:** measured 2026-09-22 at Gate G0. Every number below was computed from the downloaded
files by `vifeedback data report` and is reproducible from `results/data_report.json`. Nothing here
is copied from a paper abstract.

---

## 1. Identity

| Field | Value |
|---|---|
| Name | UIT-VSFC — Vietnamese Students' Feedback Corpus |
| Primary source | <https://nlp.uit.edu.vn/datasets/> · mirror: <https://github.com/kietnv/uit-vsfc> |
| HF Hub | `uitnlp/vietnamese_students_feedback` |
| Retrieved from | `refs/convert/parquet` branch (see [§ 2](#2-provenance-and-pinning)) |
| Paper | Nguyen, Nguyen, Nguyen, Truong, Nguyen — *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis*, KSE 2018 |
| Domain | University course feedback, collected 2013–2016, Vietnam |
| Language | Vietnamese, pre-lowercased and pre-tokenized (see [§ 6](#6-surface-and-linguistic-profile--measured)) |
| Unit | One sentence per example |
| Tasks | Sentiment (3-class) · Topic (4-class), both annotated on the same sentences |

---

## 2. Provenance and pinning

The Hub repo ships a legacy loading **script** that pulls nine plain-text files from Google Drive.
`datasets>=3.0` refuses to execute dataset scripts, and Drive is not a dependable programmatic
source, so `vifeedback data fetch` reads the Hub's own auto-converted parquet branch instead — a
byte-faithful conversion of that script's output.

Every file is pinned by SHA256 in `data/raw/manifest.json`:

| Split | Rows | SHA256 (first 16) |
|---|---|---|
| train | 11,426 | `568f1221ae4846a6` |
| validation | 1,583 | `8d87f472229762cb` |
| test | 3,166 | `dada9e9e4f8dd3ed` |

Raw data is **not committed** (`.gitignore`); the manifest is what pins the bytes. Split sizes are
asserted in `tests/data/test_integrity.py`, so a substituted or re-ordered upstream fails the build
rather than silently producing numbers against different data.

---

## 3. Splits — VERIFIED

| Split | Examples | Share |
|---|---|---|
| train | 11,426 | 70.64% |
| validation (dev) | 1,583 | 9.79% |
| test | 3,166 | 19.57% |
| **total** | **16,175** | 100% |

The official split is used **unchanged** (ADR-002). No re-splitting, no folding dev into train, no
cross-validation over the union.

---

## 4. Label schemes and class distribution — MEASURED

### Sentiment · `0=negative, 1=neutral, 2=positive`

| Split | n | negative | neutral | positive |
|---|---|---|---|---|
| train | 11,426 | 5,325 (46.60%) | **458 (4.01%)** | 5,643 (49.39%) |
| validation | 1,583 | 705 (44.54%) | **73 (4.61%)** | 805 (50.85%) |
| test | 3,166 | 1,409 (44.50%) | **167 (5.27%)** | 1,590 (50.22%) |
| **ALL** | 16,175 | 7,439 (45.99%) | **698 (4.32%)** | 8,038 (49.69%) |

**The consequence that drives every downstream decision.** A classifier that never predicts
`neutral` and is otherwise near-perfect still scores ≈0.93 weighted F1 and ≈0.93 accuracy, while its
macro-F1 is capped near 0.63. Published UIT-VSFC results in the 92–94% range are weighted or
accuracy figures. This project's headline metric is macro-F1 for exactly that reason (ADR-001).

### Topic · `0=lecturer, 1=training_program, 2=facility, 3=others`

| Split | n | lecturer | training_program | facility | others |
|---|---|---|---|---|---|
| train | 11,426 | 8,166 (71.47%) | 2,201 (19.26%) | 497 (4.35%) | 562 (4.92%) |
| validation | 1,583 | 1,151 (72.71%) | 267 (16.87%) | 70 (4.42%) | 95 (6.00%) |
| test | 3,166 | 2,290 (72.33%) | 572 (18.07%) | 145 (4.58%) | 159 (5.02%) |
| **ALL** | 16,175 | 11,607 (71.76%) | 3,040 (18.79%) | 712 (4.40%) | 816 (5.04%) |

Topic is *more* skewed than sentiment, not less: `lecturer` alone is 71.8% of the corpus, and
`facility` + `others` together are under 10%. Majority-class topic accuracy is ≈0.72 against a macro-F1
of ≈0.21 — the same trap as sentiment, one class further along.

### Joint distribution — the multi-task evidence

Sentiment × topic over the full corpus, **Cramér's V = 0.344** (moderate association; the two tasks
are *not* independent):

| | lecturer | training_program | facility | others |
|---|---|---|---|---|
| negative | 4,105 (35.4%) | 2,328 (76.6%) | 681 (**95.6%**) | 325 (39.8%) |
| neutral | 292 (2.5%) | 162 (5.3%) | 13 (1.8%) | 231 (**28.3%**) |
| positive | 7,210 (**62.1%**) | 550 (18.1%) | 18 (2.5%) | 260 (31.9%) |

*(percentages are within-topic)*

Two facts worth carrying into Phase 4:

1. **Topic is strongly predictive of sentiment.** Feedback about `facility` is 95.6% negative;
   feedback about `lecturer` is 62.1% positive. A shared encoder has real signal to exploit, so the
   multi-task experiment (Tier E) is justified *before* spending GPU hours on it.
2. **Neutral concentrates in `others`** — 28.3% of `others` is neutral, versus 2.5% of `lecturer`.
   The minority class is not spread uniformly; it lives in the catch-all topic.

---

## 5. Annotation quality

| Task | Inter-annotator agreement (published) |
|---|---|
| Sentiment | 91.20% |
| Topic | 71.07% |

The 20-point gap is a first-class finding. A topic macro-F1 near 0.75 against 71% human agreement is
a very different result from the same number against 91% agreement, and the model card must report
the topic ceiling **relative to the IAA**.

### Measured evidence of label noise

The near-duplicate analysis ([§ 7](#7-duplication-and-leakage--measured)) surfaces 63 train↔test
pairs whose text is identical after normalization. Their labels disagree:

| | agree | disagree |
|---|---|---|
| sentiment | 60 / 63 | **3** |
| topic | 61 / 63 | **2** |

Concrete cases, quoted verbatim:

| Test | Train |
|---|---|
| `em xin cám ơn !` → **positive** | `em xin cám ơn .` → **neutral** |
| `đến lớp đúng giờ !` → **positive** | `đến lớp dúng giờ !` → **neutral** |
| `đến lớp đúng giờ !` → topic **training_program** | `đến lớp đúng giờ .` → topic **lecturer** |

Formulaic politeness (`cám ơn thầy`) is annotated inconsistently between `neutral` and `positive`,
and that same phrase family is a visible part of the neutral class. This is a direct, measured bound
on the achievable ceiling — roughly 4.8% disagreement on identical text — and it is exactly the
`GOLD-ARGUABLE` category of the [error taxonomy](EVALUATION_PROTOCOL.md#error-taxonomy).

---

## 6. Surface and linguistic profile — MEASURED

### Orthography: the corpus is already normalized

| Property | Value | Implication |
|---|---|---|
| Sentences containing uppercase | **0 (0.00%)** | The corpus is fully lowercased. A lowercasing ablation condition is a **no-op** (ADR-006) |
| Unicode form | 100% NFC (or pure ASCII) | No NFC/NFD mixing; composed once on ingest |
| Sentences containing `_` | 1 | The corpus is **not** word-segmented — the P0/P1 ablation measures a real difference |
| Sentences ending in `" ."` | 15,601 (96.45%) | Punctuation is pre-separated; the text is syllable-tokenized |
| Sentences without diacritics | 23 (**0.14%**) | Diacritics are essentially always present |
| Sentences with teencode probes | 26 (**0.16%**) | Informal orthography is essentially absent (`ok` accounts for 19 of 26) |
| Sentences containing digits | 766 (4.74%) | |

**This overturns a planning assumption and must be stated plainly.** The project scope named
teencode and missing diacritics as error-analysis targets. They are not present in this benchmark:
together they touch under 0.3% of sentences. The work does not disappear — it changes register. The
`nodiacritic` and `teencode` perturbation suites ([EVALUATION_PROTOCOL § 7](EVALUATION_PROTOCOL.md#7-perturbation-suites))
now measure **deployment robustness to a distribution shift the benchmark cannot exhibit**, which is
a stronger and more honest claim than "the model struggles with teencode on the test set". Recorded
as ADR-007.

### Length

| Unit | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| Characters (train) | 47 | 110 | 141 | 218 | 660 |
| Syllables (train) | 11 | 26 | 33 | 52 | 159 |
| **PhoBERT subwords (train)** | **14** | 29 | 36 | 55 | 163 |
| PhoBERT subwords (test) | 13 | 29 | 37 | 55 | 102 |

**`max_length = 96` — decided, with justification.** It truncates 8 train sentences (0.07%) and
2 test sentences (0.06%), the smallest candidate under a 0.1% tolerance. That is **2.7× shorter than
PhoBERT's 256-token default**, and since the median input is 14 subwords, dynamic padding at batch=1
is a further ~7× reduction on top. Both are pure latency wins available before any quantization work
(Phase 6, steps L0–L1).

### Linguistic markers

Measured over the full corpus by whitespace-token matching:

| Marker family | Sentences | Share |
|---|---|---|
| Negation (`không`, `chưa`, `chẳng`, `chả`, …) | 3,300 | **20.40%** |
| Intensifier (`rất`, `quá`, `lắm`, `khá`, …) | 4,420 | 27.33% |
| Suggestion (`nên`, `cần`, `mong`, `đề nghị`, …) | 3,235 | 20.00% |
| Contrastive (`nhưng`, `tuy nhiên`, `mặc dù`, …) | 735 | 4.54% |

**Sentiment conditioned on the marker:**

| | negative | neutral | positive |
|---|---|---|---|
| has negation | **84.3%** | 4.5% | 11.1% |
| no negation | 36.3% | 4.3% | 59.5% |
| has `nên`/`cần`/`mong` | **91.1%** | 1.1% | 7.9% |
| no suggestion marker | 37.1% | 5.0% | 58.0% |

Two consequences:

1. **Negation is real here (20.4%) and highly predictive.** A bag-of-words model will capture
   `không` → negative as a unigram, so the interesting question is not whether a model sees negation
   but whether it handles negation *scope* — `không có gì để chê` ("nothing to criticize" → positive)
   is where linear models must fail. The negation probe set targets scope, not presence.
2. **In this corpus a suggestion is annotated `negative`, not `neutral`** (91.1%). The planned
   `SUGGESTION` error-taxonomy entry said the opposite and has been corrected (ADR-007). Requesting
   a change is treated as implicit criticism under the annotation guideline.

### What the neutral class actually contains

Reading a random sample of neutral sentences shows it is **not** "balanced opinion" — it is
heterogeneous, which explains why it is hard beyond merely being small:

| Sub-type | Example |
|---|---|
| Topic fragment, no opinion | `phương thức giảng dạy .` · `tất cả các hoạt động .` |
| Politeness formula | `cảm ơn thầy .` · `cám ơn thầy !` |
| Explicit non-opinion | `không có ý kiến khác .` |
| Lukewarm / hedged | `cô dạy được .` · `thầy dạy lý thuyết tạm ổn .` · `bình thường với giảng viên .` |
| Off-topic | `cho em xin ít con chip nfc .` |
| Double negation | `không có gì để không hài lòng .` |

A single `neutral` label spans six unrelated phenomena across 458 training examples. That is a
better explanation of the class's difficulty than class imbalance alone, and it argues that Tier A
(reweighting) will help less than Tier D (data-centric work) — a prediction to test in Phase 4.

---

## 7. Duplication and leakage — MEASURED

Headline: **share of test rows whose text also appears in train.**

| Mode | Test rows affected | Share of test |
|---|---|---|
| Exact string match | **0** | **0.00%** |
| Normalized (lowercase + diacritics stripped + punctuation removed) | **55** | **1.74%** |

| Mode | train↔dev | train↔test | dev↔test |
|---|---|---|---|
| exact | 0 | 0 | 0 |
| normalized | 35 (2.21% of dev) | 55 (1.74% of test) | 13 (0.41% of test) |

Within-split near-duplicates: train 119 rows (1.04%), dev 3 (0.19%), test 10 (0.32%).

**Verdict: immaterial, and disclosed.** Zero exact leakage and 1.74% normalized overlap will not
meaningfully inflate a macro-F1. The normalized-only overlap is mostly typo variants that the dedup
key deliberately collapses (`truyền đạt` / `truyền dạt`, `đúng giờ` / `dúng giờ`), plus punctuation
differences — evidence of a formulaic corpus rather than of a broken split. The pair analysis is
reused in [§ 5](#5-annotation-quality) as the measurement of label noise, which is the more
interesting finding to come out of this check.

`tests/data/test_integrity.py` pins both numbers as regression guards.

---

## 8. Known limitations

| # | Limitation | Status | Consequence |
|---|---|---|---|
| L1 | Severe sentiment imbalance — 4.01% neutral in train (458 examples) | **Confirmed** | Macro-F1 is dominated by neutral; mandates class weighting, threshold tuning, 5-seed reporting |
| L2 | Severe topic imbalance — `lecturer` 71.5%, `facility` 4.4% | **Confirmed** | Topic macro-F1 has the same trap one class further along |
| L3 | Low topic IAA (71.07%) | Published | Caps achievable topic macro-F1; `others` is a catch-all |
| L4 | Measurable gold-label noise on identical text (4.8% of matched pairs) | **Confirmed** | Bounds the ceiling; politeness formulae are annotated inconsistently |
| L5 | Neutral is a heterogeneous grab-bag of ≥6 phenomena | **Confirmed** | Explains difficulty beyond imbalance; predicts Tier D > Tier A |
| L6 | Single-domain, single-institution, 2013–2016 | Structural | No evidence of generalization to product reviews or social media. Phase 8 quantifies it |
| L7 | Sentence-level labels on mixed-polarity sentences | Structural | `Thầy dạy hay nhưng phòng học nóng` has one gold label and two polarities |
| L8 | Corpus is pre-lowercased, pre-tokenized, ~fully diacritized, ~teencode-free | **Confirmed** | Not raw user text. Robustness to informal orthography must be *induced*, not observed (ADR-007) |
| L9 | Near-duplicate leakage 1.74% of test | **Confirmed, immaterial** | Disclosed; pinned by test |
| L10 | Random rather than temporal split | Structural | Reported numbers are in-distribution and optimistic for deployment |
| L11 | No demographic metadata | Structural | No subgroup fairness analysis is possible; do not imply one was done |

---

## 9. Validation checks — all passing at G0

Implemented in `tests/data/test_integrity.py`, run by `pytest -m needs_data`.

- [x] Split sizes exactly 11,426 / 1,583 / 3,166
- [x] Label ids in `{0,1,2}` / `{0,1,2,3}`; no nulls; every class present in every split
- [x] Label maps contiguous from zero (guards against a transposed confusion matrix)
- [x] No empty or whitespace-only sentences; no nulls
- [x] Text is NFC-composed on ingest; no NFD or mixed forms downstream
- [x] Exact duplicates within and across splits
- [x] Near-duplicate cross-split overlap, reported and pinned < 5%
- [x] Neutral is the minority sentiment class in every split
- [x] `facility` is the minority topic class in every split
- [x] Sentiment and topic are not independent (Cramér's V > 0.2)
- [x] Corpus is lowercased and not word-segmented
- [x] Length percentiles in characters, syllables and PhoBERT subwords
- [x] SHA256 manifest of the raw files

---

## 10. Preprocessing variants

Materialized once into `data/processed/<variant>/` and reused by every run, so preprocessing cost
never contaminates training-time measurements and every model sees byte-identical input.

| Variant | Transformation | Status |
|---|---|---|
| `raw` | As distributed; whitespace trimmed only | Phase 0 |
| `seg_vncorenlp` | RDRSegmenter via `py_vncorenlp` — PhoBERT's canonical pipeline | Phase 3 · **needs a JVM, absent on this machine** (R1) |
| `seg_underthesea` | `underthesea.word_tokenize` | Phase 3 |
| `norm` | NFC + whitespace/punctuation cleanup | Phase 3 |
| `norm_teencode` | `norm` + teencode dictionary expansion | Phase 3 |
| ~~`norm_lower`~~ | ~~lowercasing~~ | **Dropped — the corpus is already 100% lowercase (ADR-006)** |
| `perturb_nodiacritic` | Test-only: diacritics stripped | Phase 5 (robustness) |
| `perturb_teencode` | Test-only: teencode substituted in | Phase 5 (robustness) |
| `perturb_charnoise_{5,10}` | Test-only: character swap/drop/duplicate | Phase 5 (robustness) |

---

## 11. Licensing and citation

Check the UIT NLP Group's terms at <https://nlp.uit.edu.vn/datasets/> before redistributing.
Raw data is not committed; `vifeedback data fetch` retrieves it and the SHA256 manifest pins it.

```bibtex
@inproceedings{van2018uit,
  title={UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis},
  author={Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu Xuan-Vinh
          and Truong, Tham Thi-Hong and Nguyen, Ngan Luu-Thuy},
  booktitle={2018 10th International Conference on Knowledge and Systems Engineering (KSE)},
  pages={19--24},
  year={2018}
}
```
