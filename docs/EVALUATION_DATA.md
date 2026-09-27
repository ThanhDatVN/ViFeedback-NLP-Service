# Evaluation Data and the Evaluation Matrix — what counts as credible evidence

**Written 2026-09-27, Cycle 3.** Cycle 2 confirmed its decisions on a challenge set that turned out to be
weaker evidence than it was presented as (ADR-028): constructed text, one labeller, 20–50 rows per
category. This document sets what each claim needs instead. It covers the evaluation matrix, the
metrics per cell, the minimum sample sizes, and the test data ranked by credibility, with the sources
behind each choice.

---

## 1. The evaluation matrix

**Rows are capabilities, columns are test types** (CheckList: Ribeiro et al., ACL 2020). Each cell
also names its data source and credibility tier (§ 3) and its metric (§ 2), which is the multi-metric
idea of HELM (Liang et al., 2022): accuracy alone never decides.

| Capability | MFT: labelled examples | INV: label-preserving change, prediction must not move | DIR: change with a known effect | Data (tier) |
|---|---|---|---|---|
| Clean in-domain feedback | UIT-VSFC validation / test | — | — | A |
| Real informal typing (teencode, slang) | NEU-ESC test, real forum text | **ViLexNorm: original vs human-normalized sentence** | — | B, B′ |
| Missing diacritics | challenge v2 unaccented rows | UIT-VSFC with diacritics stripped (suite v1); ViLexNorm pairs that differ only in diacritics | — | C, A (synthetic) |
| Negation | challenge v2 negation pairs | — | negation flips positive ↔ negative (minimal pairs) | C |
| Contrast / mixed aspects | challenge v2 `mixed_aspect` | — | the clause after *nhưng* sets the label | C |
| Suggestions (the corpus convention) | challenge v2 suggestions | — | adding *nên / mong* to a neutral fact moves it to negative | C |
| Factual or no-opinion text (neutral) | challenge v2 short factual rows; UIT-VSFC neutral | — | — | C, A |
| Code-switching | challenge v2 | replacing an English word with its Vietnamese equivalent | — | C |
| Domain shift (forum vs survey) | NEU-ESC test (3-class mapping) | — | — | B |
| Out-of-scope input | challenge v2 off-topic rows; NEU-ESC `Spam` / `News` topics | — | — | C, B |

The INV cells with ViLexNorm need **no sentiment labels**: the model is asked whether it gives the same
answer to real noisy text and to the same text normalized by a person. That turns an existing,
peer-reviewed, human-annotated corpus into a label-free robustness test on real typing.

## 2. Metrics per cell

| Cell type | Primary metric | Uncertainty | Also reported |
|---|---|---|---|
| MFT, whole dataset | Macro-F1 (neutral carries a third of it) | Paired bootstrap over examples; seed-paired t over 5 seeds | Per-class P/R/F1, confusion matrix |
| MFT, one category | Accuracy | Wilson interval; exact McNemar when two models are paired | Errors listed for inspection |
| INV | Flip rate: share of pairs whose label changes | Wilson interval; McNemar between models | Mean change in the probability of the original label |
| DIR | Share moving in the expected direction | Wilson interval | Mean probability shift |
| Confidence | NLL, ECE (15 equal-width bins and equal-mass), Brier | Bootstrap | Reliability diagram; temperature from validation |
| Abstention | AURC; coverage per class at a fixed risk | Bootstrap | Risk–coverage curve |
| Out-of-scope | AUROC; share of in-domain flagged at 95% out-of-scope recall | Bootstrap | Score histograms |
| Cost | p50/p95 latency (reference CPU), texts/s, artifact size; USD per 1k for API models | Two passes, rotated order | Peak RSS |

A cell that decides something states its test and its margin in `configs/experiments/cycle3.yaml`
before it is evaluated. Several cells deciding together are corrected (Holm).

## 3. Test data ranked by credibility

| Tier | What makes it credible | Sets |
|---|---|---|
| **A** | Real text, human labels, published, agreement reported | UIT-VSFC (agreement 91.2% sentiment, 71.1% topic; Nguyen et al., KSE 2018). Its **test split has been evaluated 28 times**, so it is for historical comparison only |
| **B** | Real text, human labels, peer-reviewed or public paper, agreement *not* reported | **NEU-ESC** test split (6,613 of 32,966 posts from Vietnamese university forums and Facebook groups, 2025) |
| **B′** | Real text, human annotation of a different task, peer-reviewed | **ViLexNorm** test split (about 1,050 social-media comments with human normalizations, EACL 2024) |
| **C** | Text typed by people for this project, two labellers, agreement measured | **Challenge v2** (to be written; § 5) |
| **D** | Constructed text, one labeller | Challenge v1: development only (ADR-028) |
| — | Real course evaluations, human labels, Fleiss κ "substantial" | **DUIT** (EduPulse, EACL 2026 industry track, UIT course surveys): internal, not released. Worth a request to the UIT NLP group |

### NEU-ESC in detail

- **Content:** 32,966 posts (train 23,048 / validation 3,305 / test 6,613), sentiment `Neutral`
  (69.1%), `Positive` (12.6%), `Negative` (15.8%), `Toxic` (2.6%), and 10 topics (`Academic`,
  `Service`, `Other`, `Spam`, `News`, …). Average 25 words per post against 14 for UIT-VSFC, and
  much more slang: the authors call UIT-VSFC "relatively formal".
- **How it fits:** it is the only public, human-labelled set of **real student writing in Vietnamese
  with informal typing**. That is exactly what challenge v1 lacked.
- **Limits:**
  - No agreement figure is reported.
  - `neutral` means "no emotion on a topic" in forum posts, a different label mix and policy from
    course feedback.
  - The labels are not UIT-VSFC's: `Toxic` is mapped to `negative` (declared before evaluation, and
    reported with and without those rows).
  - Access is gated on Hugging Face (accept the conditions with the owner's account).
  - The licence is stated as CC BY 4.0 in the paper and Apache-2.0 on the dataset card; either allows
    research evaluation, and the text is never committed here.
- **Use:** a domain-shift and real-noise test. Two views, both declared before evaluation: all posts,
  and the `Academic` + `Service` topics, which are closest to course feedback. Expect lower scores than
  on UIT-VSFC, partly from label policy. The comparison between models on the same posts is the claim,
  not the absolute number.

### ViLexNorm in detail

- 10,467 comment pairs, split 8:1:1, `original` → `normalized` by human annotators; CC BY-NC-SA 4.0,
  research use only, citation required.
- **Use:** the INV test above, on the test split: flip rate between original and normalized text, for
  CE, the served augmented model, and any Cycle 3 candidate. It is also the licensed source the
  teencode lexicon v2 can draw from (Step 2a), from the **train** split only, so the test split stays
  unseen.

## 4. What the literature already shows, and how it matches this project

EduPulse (Nguyen et al., EACL 2026 industry track), from the UIT group behind UIT-VSFC, reports:

| Data | PhoBERT-base (fine-tuned) | GPT-4o, zero-shot |
|---|---|---|
| UIT-VSFC sentiment | macro-F1 **0.828** | macro-F1 0.689 |
| DUIT: real course evaluations with teencode (5 × 300) | accuracy 0.717, macro-F1 0.508 | accuracy **0.920**, macro-F1 0.786 |

This is the same pattern as H7 here: the LLM loses on UIT-VSFC, where the labels follow the corpus's
own policy, and wins on noisy real text. That supports the pilot's reading. It also shows why the
gpt-4o-mini result on challenge v1 (+0.24 neutral F1) is not enough on its own: v1 was written in the
prompt's convention. The decision needs Tier B and C data.

## 5. Sample size: how big each cell must be

Card et al. (EMNLP 2020) show that underpowered test sets are common in NLP and that small sets make
most comparisons inconclusive. For an accuracy near 0.8:

| Rows per cell | 95% Wilson half-width | Smallest paired difference detectable (80% power, 15% discordant pairs) |
|---:|---:|---:|
| 30 | ± 0.14 | 0.20 |
| 50 | ± 0.11 | 0.15 |
| 100 | ± 0.08 | 0.11 |
| 200 | ± 0.055 | 0.08 |
| 400 | ± 0.04 | 0.05 |

Challenge v1's cells (20–50 rows) could only detect differences of 0.15–0.20, which is why its two
drops were inconclusive. **A cell that decides something needs ≥ 100 rows; 200 for effects near
0.05.** Smaller cells are descriptive.

## 6. Challenge v2: the protocol for Tier C

1. **Who writes.** At least three people, the owner plus two others, typing on their own phones as
   they would message, with no editing afterwards. Different writers cover different habits, which is
   the point of "real typing".
2. **What.** Decision cells of ≥ 100 rows each: unaccented (≥ 100), teencode or abbreviated (≥ 100),
   short factual (≥ 100). Descriptive cells: 50 negation pairs, off-topic (≥ 50), mixed aspects (≥ 50).
3. **Labels.** Two labellers per row, blind to each other and to any model output, under
   ANNOTATION_GUIDE § 2. Report Cohen's κ per category. Disagreements are adjudicated; unresolved rows
   are marked `ambiguous` and excluded from accuracy (reported separately). Target κ ≥ 0.80, against
   0.91 for UIT-VSFC sentiment.
4. **Isolation.** No row is shown to any model, and no model output is shown to any writer, before
   the hash is frozen in `cycle3.yaml` v2. No LLM writes or edits a row.
5. **Checks before freezing.** Deduplication against UIT-VSFC, NEU-ESC and ViLexNorm (normalized
   key); per-category counts; κ.

## 7. Changes to the plan

| Item | Change |
|---|---|
| Cycle 3 confirmation | Every Step 2–3 rule is confirmed on **challenge v2 (Tier C) and NEU-ESC (Tier B)**; ViLexNorm INV flip rate is reported for each candidate |
| H7 | The LLM–encoder comparison is repeated on NEU-ESC test. The data egress rule applies: NEU-ESC's licence allows research use, but sending its text to the OpenAI API waits for the owner's confirmation |
| Challenge v2 | Written by ≥ 3 people, ≥ 100 rows per decision cell, two labellers, κ reported (§ 6) |
| Teencode lexicon v2 | Built from ViLexNorm **train** pairs plus owner-checked candidates, never from any test data |
| DUIT | The owner may ask the UIT NLP group for evaluation access; if granted, it becomes the best test set available (Tier A-) |

## Sources

- Ribeiro, Wu, Guestrin, Singh. *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.* ACL 2020. <https://aclanthology.org/2020.acl-main.442/>
- Liang et al. *Holistic Evaluation of Language Models.* 2022. <https://arxiv.org/abs/2211.09110>
- Card, Henderson, Khandelwal, Jia, Mahowald, Jurafsky. *With Little Power Comes Great Responsibility.* EMNLP 2020. <https://aclanthology.org/2020.emnlp-main.745/>
- Nguyen et al. *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis.* KSE 2018. <https://ieeexplore.ieee.org/document/8573337/>
- Mai et al. *NEU-ESC: A Comprehensive Vietnamese Dataset for Educational Sentiment Analysis and Topic Classification toward Multitask Learning.* 2025. <https://arxiv.org/abs/2506.23524>; data: <https://huggingface.co/datasets/hung20gg/NEU-ESC>
- Nguyen et al. *ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text.* EACL 2024. <https://aclanthology.org/2024.eacl-long.85/>; data: <https://github.com/ngxtnhi/ViLexNorm>
- Nguyen Xuan Phuc et al. *EduPulse: A Practical LLM-Enhanced Opinion Mining System for Vietnamese Student Feedback in Educational Platforms.* EACL 2026, Industry Track. <https://aclanthology.org/2026.eacl-industry.25.pdf>
- Northcutt, Athalye, Mueller. *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks.* NeurIPS 2021 Datasets and Benchmarks.
