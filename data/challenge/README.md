# Challenge set v1

305 constructed Vietnamese feedback sentences, written by the project author for Cycle 2
(`configs/experiments/cycle2.yaml`). None is drawn from UIT-VSFC. Every row was checked against the
corpus (train, validation, test) and the negation probe with `dedup_key` (lowercase, no diacritics,
no punctuation), and none matches.

**Frozen.** The file's SHA-256 is recorded in `cycle2.yaml` version 2, committed before any model is
evaluated on it. No row is edited after that. Corrections go to `challenge_v2.csv`.

| Column | Meaning |
|---|---|
| `id` | `c001` … `c305` |
| `category` | one of the ten below |
| `pair_id` | `n01` … `n15` for `negation_pair` rows; empty otherwise |
| `text` | lowercase, as a student would type it; no word segmentation |
| `sentiment` | `positive` / `negative` / `neutral`; `none` for `out_of_scope` |

Read it with `keep_default_na=False`: pandas would otherwise parse an empty `pair_id` as missing.

## Categories

| Category | Rows | neg / neu / pos | What it tests |
|---|---:|---|---|
| `negation_pair` | 30 | 15 / 0 / 15 | 15 minimal pairs whose negation is harder than the 44-pair probe's: *đâu có*, *có … đâu*, *chẳng … chút nào*, *chưa bao giờ*, *không hề*, negated negatives |
| `mixed_aspect` | 40 | 20 / 0 / 20 | Two evaluations joined by *nhưng* / *tuy* / *mặc dù*, often across aspects (lecturer vs room) |
| `suggestion_cue` | 25 | 25 / 0 / 0 | Requests for change with *nên* / *cần* / *mong* |
| `suggestion_implicit` | 25 | 18 / 7 / 0 | Requests without a cue word (18), and questions or information requests (7) |
| `unaccented_typed` | 50 | 20 / 10 / 20 | Written without diacritics by hand, not by `strip_diacritics` |
| `teencode_typed` | 40 | 16 / 8 / 16 | Abbreviations and informal spelling as typed (*ko*, *wá*, *hỉu*, *bt*, *tks*) |
| `code_switch` | 25 | 9 / 4 / 12 | English words inside Vietnamese sentences (*deadline*, *slide*, *support*) |
| `long_context` | 20 | 8 / 4 / 8 | 23–38 words, no punctuation: the evaluation comes late or is spread out |
| `objective_neutral` | 30 | 0 / 30 / 0 | Facts about the course, and non-answers |
| `out_of_scope` | 20 | label `none` | Not course feedback. Excluded from accuracy; used to describe confidence on off-topic input |

## Labelling policy

Labels follow [docs/ANNOTATION_GUIDE.md](../../docs/ANNOTATION_GUIDE.md) § 2, which follows the
corpus's own conventions:

- **Suggestions.** A request for change is `negative`, as 91.1% of the corpus's *nên / cần / mong*
  sentences are (ADR-007). A question or information request is `neutral`.
- **Contrast.** In *A nhưng B*, the clause after the contrast word sets the label. This matches the
  corpus: of the 464 train sentences containing *nhưng*, *tuy* or *mặc dù*, 5.0% are gold `neutral`,
  about the corpus base rate (4.0%). The corpus does not reserve `neutral` for mixed sentences.
  `mixed_aspect` results are always reported separately, because this is the least certain policy
  in the set.
- **Negated negatives.** *thầy chưa bao giờ đến lớp trễ* is `positive`: it is praise.

## Limits

- One author wrote and labelled every row, so there is no agreement estimate. The set tests
  phenomena, not the label distribution of real feedback.
- Typed noise reflects one writer's habits, so it is wider than the synthetic suites but not a
  sample of real users.
- `sentiment` only. Topic is not labelled: `mixed_aspect` rows have two topics by design.

The CSV is the artifact. The script that wrote it is not part of the pipeline.
