# `configs/experiments/` — declared before running

One file per research cycle. Each one lists its hypotheses, the exact conditions, seeds, the success
criterion, and the **run budget**, written *before* any of its runs start. That is what makes the
difference between "we tried things until one worked" and a test: the comparison, the metric and the
threshold cannot move after the numbers are seen
([REVIEW_AND_RESEARCH_PLAN.md §§ 9–10](../../docs/REVIEW_AND_RESEARCH_PLAN.md)).

| File | Cycle | State |
|---|---|---|
| [`ledger.csv`](ledger.csv) | all | Every run that updated model weights, including pilots and diagnostics |
| [`cycle1.yaml`](cycle1.yaml) | 1 | Declared 2026-09-27: H1 imbalance (logit adjustment, cRT), H2 augmentation, H3 XLM-R input confound |
| [`cycle2.yaml`](cycle2.yaml) | 2 | Declared 2026-09-27: H5 topic stacking, H6 serving model, H7 LLM reference (track A), audit decision tree |
| [`cycle3.yaml`](cycle3.yaml) | 3 | Declared 2026-09-27, v1–v6: reliability on real input (V1 seeds, external evaluation, S2a lexicon, S2b restoration, S2b′, S3 out-of-scope score, S5 INT8), confirmation moved to NEU-ESC (v5, ADR-030), the H7 decision (v6) |
| [`cycle4.yaml`](cycle4.yaml) | 4 | Declared 2026-09-28, v1–v4: H8 in-domain data (three recipes), B4′ topic-aware scope detector, S5′ INT8 on a larger acceptance set |
| [`cycle5.yaml`](cycle5.yaml) | 5 | Declared 2026-09-29, v1: H10 consistency training on real typing (ViLexNorm pairs); H11 distilled student and H12 other institutions planned, declared later before their first run |

## Rules

- **Every weight update counts.** Pilots, failed runs and diagnostic retraining (the Study A OOF folds)
  go in the ledger. A budget that only counts the runs that worked hides the search cost.
- **Budgets are in runs *and* GPU-minutes.** A multi-task run and a single-task run have different
  costs.
- **An existing run substitutes for a new one only when** the data, preprocessing, software version
  and configuration match, and its artifact exists (review § 10).
- **Changing a declared criterion after seeing results** requires a new file version and a note in
  `docs/DECISIONS.md` saying what changed and why.
