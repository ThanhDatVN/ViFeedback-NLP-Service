# Probe sets

Small, hand-written evaluation sets. **Constructed for this project, not drawn from UIT-VSFC**, so they
can be committed (the corpus itself is not redistributed; see [DATA_CARD § 11](../../docs/DATA_CARD.md)).

## `constructed_course_forum_v1.csv` — exploratory authored examples

Sixty authored sentences test four course-feedback groups (12 each) and 12 off-topic inputs. Labels
follow the UIT-VSFC convention that a request for change is negative. These sentences were written
by an AI assistant on 2026-10-25, before running the service on them. They are **not** posts from a
third university, independently human-labelled data, H12 confirmation data, or a representative
sample. The file is frozen for this run; a revision gets a new filename.

`python scripts/evaluate_constructed_probe.py --out results/studies/constructed_probe/served.json`
scores the served pipeline, records the input file's SHA-256 and model hash, and saves each prediction
by ID, with the manifest, restorer and scope hashes for provenance. The initial H10b run gave
macro-F1 0.958 on the 48 labelled in-scope examples and correctly
flagged only 2 of 12 off-topic examples. This is a diagnostic of these authored examples only.

## `negation_v1.csv` — negation minimal pairs

Each row is a pair that differs only by a negation cue (`không`, `chưa`, `chẳng`, or the informal `ko`).

| Set | Pairs | What it tests | How firm the expected labels are |
|---|---:|---|---|
| `pos_to_neg` | 36 | Negating a positive predicate yields a negative sentence | Firm: *không nhiệt tình* is a complaint under any reading |
| `neg_to_pos` | 8 | Negating a negative predicate yields a positive one | **Arguable.** *thầy không khó tính* may be read as mild praise or as neutral. Reported separately and never pooled with `pos_to_neg` |

Metrics (from `vifeedback study robustness`, `robustness.negation_probe`):

- **pair accuracy** — both sentences of a pair classified as expected. The strict metric: a model that
  calls everything negative gets half of every pair right and zero pairs right.
- **flip rate** — the prediction changes between base and negated, regardless of direction.
- **base accuracy** — the un-negated sentences alone. A pair whose base is already misclassified says
  nothing about negation, so pair accuracy is also reported *conditional on a correct base*.

The labels are the author's judgement under [ANNOTATION_GUIDE.md](../../docs/ANNOTATION_GUIDE.md). With 36
pairs, one pair is ~2.8 points of pair accuracy: read differences between models accordingly.
Changing any row means a new file (`negation_v2.csv`), never an edit in place.
