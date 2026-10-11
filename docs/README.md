# Documentation

Where to find what. The documents are in English. The one exception is the owner's step-by-step
guide to the manual tasks, which is in Vietnamese.

## Start here

| Document | What it answers |
|---|---|
| [STATUS.md](STATUS.md) | Where the project stands: every gate and research cycle with its result, the open problems, the external review's items |
| [NEXT_PLAN.md](NEXT_PLAN.md) | Cycle 5 plan and updates through 2026-10-01; consult STATUS for the current state and remaining work |
| [RESEARCH_REPORT.md](RESEARCH_REPORT.md) | Technical report through Cycle 4, with a Cycle 5 update: questions Q1–Q9, methods, findings and limitations |
| [HUONG_DAN_THU_CONG.md](HUONG_DAN_THU_CONG.md) | *(Tiếng Việt)* The tasks only the owner can do: the neutral audit, label review, idle latency sessions, licence and label-policy decisions, new labelled data |

## The record

| Document | What it holds |
|---|---|
| [DECISIONS.md](DECISIONS.md) | Append-only decision log: decisions, scope cuts and failed gates, with context and consequences |
| [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) | Run-id scheme, matrix axes, and every result table from the baselines to Cycle 5 |
| [`configs/experiments/`](../configs/experiments/README.md) | The pre-registered cycles (`cycle1.yaml` … `cycle5.yaml`) and the compute ledger |
| [`results/`](../results/README.md) | Every committed result: registry, per-run metrics, study outputs |

## Contracts and protocols

| Document | What it fixes |
|---|---|
| [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) | Metrics, seeds, significance tests, test-set discipline, the serving contract |
| [EVALUATION_DATA.md](EVALUATION_DATA.md) | What counts as evidence: the evaluation matrix, test data ranked by credibility, sample sizes |
| [DATA_CARD.md](DATA_CARD.md) | UIT-VSFC measured from the files: splits, labels, overlap, licence, what is never committed |
| [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) | Labels, neutral subtypes and gold assessment for the neutral audit (v1.1) |
| [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) | Running the three Kaggle notebooks for cloud experiments |

## Foundations

Written before or at the start of the research cycles. They still define rules that later documents
cite (ROADMAP § 9's reporting rules, the review's study designs), so they stay in place.

| Document | What it is |
|---|---|
| [ROADMAP.md](ROADMAP.md) | The original plan (2026-09-22): phases, gates, scope boundaries, success criteria S1–S10 (revised in ADR-015) |
| [REVIEW_AND_RESEARCH_PLAN.md](REVIEW_AND_RESEARCH_PLAN.md) | The external technical review (2026-09-23/24) and the research plan the cycles follow |

## Archive

Dated snapshots, kept as the record and no longer updated.

| Document | Snapshot of |
|---|---|
| [archive/RESEARCH_NOTES.md](archive/RESEARCH_NOTES.md) | The landscape survey before planning (2026-09-22) |
| [archive/PROPOSALS.md](archive/PROPOSALS.md) | Techniques and models proposed after Gate G3 (2026-09-22) |
| [archive/BENCHMARK_COMPARISON.md](archive/BENCHMARK_COMPARISON.md) | The comparison with published UIT-VSFC results at Gate G3 (2026-09-22) |
| [archive/REVIEW_COMPLIANCE.md](archive/REVIEW_COMPLIANCE.md) | The review-compliance audit during Cycles 1–2 (2026-09-27); current item status is STATUS § 8 |
