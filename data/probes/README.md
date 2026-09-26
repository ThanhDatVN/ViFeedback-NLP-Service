# Probe sets

Small, hand-written evaluation sets. **Constructed for this project, not drawn from UIT-VSFC**, so they
can be committed (the corpus itself is not redistributed; see [DATA_CARD § 11](../../docs/DATA_CARD.md)).

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
