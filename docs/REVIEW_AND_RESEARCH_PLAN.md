# ViFeedback: Technical Review and Experimental Research Plan

Initial code review: **23 September 2026**. Research plan completed and scope updated: **24 September 2026**. Target roles: **Junior ML Engineer, AI Engineer, and Data Scientist**.

**Current scope:** model experiments, data quality, evaluation methodology, robustness, uncertainty, learning efficiency, and offline inference optimization. Backend/frontend development, dashboards, web deployment, and product workflows are deferred. The existing service is reviewed as part of the repository, but extending it is not a prerequisite for this research plan.

This assessment is based on the source code, experiment registry, local artifacts, and primary research sources. It is not a hiring outcome or a promise of higher scores. The experiments proposed below **were not run during this review**. The project owner's existing changes were preserved.

**Recommended starting point:** complete the validity fixes, investigate neutral errors, evaluate calibration/robustness, and test one shared sentiment/topic encoder. Then choose one specialization: label efficiency for Data Science, compression/PEFT for ML Engineering, or an encoder-versus-LLM study for AI Engineering. Do not run all 22 experiment families in one cycle.

Reading guide:

- [Current evidence and review findings](#1-assessment-and-current-evidence)
- [Research questions and dataset extensions](#3-expand-the-research-problem)
- [Models and prioritized experiment catalog](#5-model-ladder-each-model-should-answer-a-question)
- [Detailed study designs](#7-five-detailed-study-designs-worth-prioritizing)
- [Evaluation protocol](#8-evaluation-protocol-for-the-next-research-cycle)
- [Success criteria and compute budget](#9-research-success-criteria-and-stopping-rules)
- [Junior-role evidence and roadmap](#11-evidence-that-matters-for-junior-ml-ai-and-data-science-roles)

## 1. Assessment and Current Evidence

The project has a stronger foundation than a standalone fine-tuning exercise: tuned baselines, multiple seeds, minority-class evaluation, a decision log, a CLI, an API, and CI. Its main weakness is the gap between **claims**, **experimental evidence**, and **artifacts that another researcher can reproduce and evaluate**.

Recommended positioning: **a reproducible study of reliable, data-efficient Vietnamese feedback understanding under class imbalance and distribution shift**. UIT-VSFC is the initial benchmark. New evaluation data, controlled interventions, and one task extension should establish what improves quality, under which conditions, and at what computational cost.

The following results were recalculated from the current `results/registry.csv`, using the mean and sample standard deviation (`ddof=1`):

| Configuration | Split | Seeds | Macro-F1 | Mean minority-class F1 |
|---|---|---:|---:|---:|
| TF-IDF B3 + priors fitted on dev, sentiment | test | 1 | 0.7450 | See baseline artifact |
| PhoBERT-base + VnCoreNLP, sentiment | test | 5 | **0.8373 ± 0.0031** | neutral: 0.5955 |
| PhoBERT-base + pyvi, sentiment | test | 5 | **0.8288 ± 0.0108** | neutral: 0.5714 |
| PhoBERT-base + pyvi, topic | test | 5 | **0.8038 ± 0.0045** | others: 0.5533 |
| PhoBERT-large + pyvi, sentiment | dev | 5 | Approximately 0.8560 | Not a test result |
| XLM-R-base + pyvi, sentiment | dev | 5 | Approximately 0.8403 | Not a test result |

The registry contains **94 evaluation rows**: 73 validation and 21 test rows. These should not be described as 94 independent training runs, because one training run can produce two rows. Ten rows belong to the two models run on Kaggle. No duplicate IDs were found in the inspected snapshot.

Checks completed during the initial review on 23 September; the registry counts and headline scores were rechecked on 24 September:

- `python -m pytest tests/unit tests/contract tests/data -q -m 'not slow'`: passed, with one skipped test and dependency/scheduler warnings.
- `python -m ruff check src tests`: passed.
- Independently reproduced the gradient-scaling bug in the FGM + AMP branch using small tensors, without training a model.
- No `.onnx` files were found under `models/` at review time. Docker, full training integration tests, and new deployment benchmarks were not run.

## 2. Issues to Resolve Before Expanding the Experiments

For the current research scope, prioritize R1, R2, R7–R12. R3 matters before reporting ONNX/INT8 experiments. R4–R6 remain valid observations about the existing service but are deferred; do not spend the research budget building API or UI functionality.

### R1 — High: the segmentation paper is being misinterpreted

`README.md:61–72` and ADR-012 present the project as refuting a conclusion that segmentation is unnecessary. However, the abstract of the [original paper](https://arxiv.org/abs/2301.00418) distinguishes two cases: some traditional classifiers on social-domain text may not require segmentation, whereas deep learning with BPE does. The [official PhoBERT instructions](https://github.com/VinAIResearch/PhoBERT#notes) also require word-segmented input.

A defensible description would be: “Quantified the effect of segmentation on macro-F1 and neutral F1 in a PhoBERT pipeline, while measuring the cost of different segmenters.” There is insufficient evidence to claim that the project “refutes the paper” or that “a field-level conclusion is an artifact of the metric.” This corrects the interpretation without invalidating the repository's ablation measurements.

### R2 — High: FGM with mixed precision has a gradient bug

In `src/vifeedback/training/trainer.py:284–300`, the clean gradient has already passed through `unscale_`, after which the adversarial gradient is added through `scaler.scale(adv_loss).backward()`. The two gradients have different scales. The optimizer step does not automatically unscale the newly added component because the optimizer has already been marked as unscaled. With `grad_accum > 1`, the next microbatch can call `unscale_` a second time before `update()`, raising a RuntimeError.

Reproducing the mechanism with `GradScaler('cpu', init_scale=128)`: a clean gradient of 1 and an adversarial gradient of 2 should sum to 3, but the current sequence produces 257; the second unscale call raises an error. Keep accumulated gradients on the same scale and unscale only once before clipping/stepping. Compute the perturbation from a correctly handled gradient copy. Add AMP/FP32 equivalence and accumulation checks for this branch before using FGM results. CE runs with `fgm_epsilon=0` are unaffected by this bug.

### R3 — High: export does not enforce the documented quality contract

`src/vifeedback/cli.py:269–314`:

- Static INT8 calibration and parity checks use raw text, while serving defaults to pyvi. They must use the checkpoint's actual preprocessing.
- Parity checks cover only the first 64 sentences and label agreement. `logits_close` is computed but does not block an FP32 export with divergent logits.
- There is no macro-F1/per-class F1 check of the INT8 artifact on an independent acceptance set before release.
- `OnnxClassifier` automatically selects the quantized file if it exists (`onnx_export.py:133`). Re-exporting with `quantize=none` into an existing directory can still select a stale quantized file.
- Artifacts are written into the serving directory before parity verification finishes. A failure does not enforce a staging/release boundary.

Add a manifest specifying the exact file, SHA256, tokenizer revision, preprocessing, label map, and maximum length. Export into a new version directory, verify it, and only then update the served version. FP32 needs logit parity; INT8 needs a quality budget and measured latency. If a graph was optimized for a CPU in another environment, verify compatibility on the target machine rather than assuming every optimized artifact is hardware-independent.

### R4 — High: API contract tests do not yet demonstrate successful inference

`tests/contract/test_api.py` primarily checks the model-free state, schemas, and invalid inputs. Docker CI also runs its smoke test without an artifact. These checks are useful, but they do not demonstrate that `POST /v1/classify` returns correct labels from a real ONNX model. The file's reference to golden-prediction tests was not matched by a corresponding inference suite in the current test tree.

Add a smoke test with a small artifact to verify the integration, plus a release/manual integration test using the real model, golden cases, label mapping, preprocessing, and a checkpoint → ONNX → API round trip. Distinguish “infrastructure implemented” from “deployed and verified.”

### R5 — Medium: readiness and preprocessing can overstate service health

`serving/app.py:72–80` catches segmenter failures and serves raw text instead, while `readyz` requires only that any model be loaded. Readiness should depend on the configured tasks and required preprocessing. Missing artifacts or segmenters for required tasks should produce HTTP 503 for readiness. The current Docker health check reads the JSON body, but a conventional HTTP readiness probe would see status 200.

The `-0.023 macro-F1` degradation in the fallback log comes from comparing models trained with two different pipelines. It does not directly measure the damage from feeding raw text into a model trained on segmented text, so it should not be treated as a guaranteed fallback degradation.

### R6 — Medium: metrics accumulate memory without a bound

`serving/app.py:53,107,212`: `_latencies` appends an entry for every request, even though percentile calculations read only the last 10,000 entries. The list still grows indefinitely. Use a histogram or a bounded buffer with a separate counter, and verify correctness under concurrent requests. Multiple workers also require an explicit metrics aggregation design.

### R7 — High for interpretation: the statistical protocol makes some overly strong claims

- A CI for **one model's score** is not a CI for **the difference between two models**. The width of one model's CI does not establish that “every improvement below 0.027 is noise”; a paired difference also depends on error correlation.
- A seed-level t-test on the same dev set describes training variation conditional on that set. It does not address evaluation-sample uncertainty. Report both instead of choosing one to support a general superiority claim.
- The same seed value does not imply the same initialization across architectures. Even within one architecture, RNG consumption order and the sampler matter.
- A lack of statistical significance does not establish equivalence. For quantization, specify an acceptable degradation margin and assess non-inferiority using the quality difference.
- Twenty-one test rows do not themselves prove leakage. Conversely, inspecting the test set once and changing the model accordingly can introduce test-set adaptation. Track **decisions influenced by test results** rather than applying a mechanical rule that “30 rows turn test into dev.”

Methodological reference: [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/). The specific applications above are this review's analysis.

### R8 — Medium: model comparisons do not justify closing the architecture search

ADR-016 draws broad conclusions from one recipe, four epochs, and XLM-R evaluated only with `seg_pyvi`. Add an XLM-R **raw-text** control, tokenizer-specific length profiles, and comparable tuning budgets. PhoBERT and XLM-R differ in tokenizer, corpus, vocabulary, and architecture; this is not an experiment that isolates parameter count.

The current conclusion should be: “PhoBERT-base performs best among the configurations tested under this budget.” Stopping larger-model experiments for cost reasons is reasonable, but the evidence does not establish that encoder capacity has no remaining value or that CafeBERT cannot help.

### R9 — Medium: separate the headline result from the deployment configuration

The 0.8373 headline belongs to **VnCoreNLP**. The **pyvi** configuration intended for serving has test macro-F1 of 0.8288 ± 0.0108 and neutral F1 of 0.5714. Both results deserve reporting, with each score attached to the correct model/pipeline. Do not attribute the best research score to a different deployment artifact.

### R10 — Medium: documentation and benchmarks are out of sync

- README/STATUS still report 69 runs and zero external GPU use, while the registry already includes Kaggle results.
- STATUS reports G4 as complete while also saying the test set has not been inspected. The current test-row count is 21.
- In `results/bench_phase6_torch.json`, the full protocol reports dynamic-padding model p95 of **70.452 ms** and end-to-end pipeline p95 of **63.175 ms**, with large variation and `throttling_suspected=true`. This benchmark cannot yet establish an SLA. Variation indicates unstable measurement; it does not, by itself, prove thermal throttling as the cause.
- “End-to-end” in the current harness means a Python call through preprocessing and the model. It excludes HTTP, queueing, and concurrent load.
- `throughput()` computes `n_batches * batch_size / elapsed`, which measures **texts/second**, not HTTP requests/second as the current naming suggests.
- The model-only benchmark receives raw texts, whereas the end-to-end pipeline receives segmented texts. Measure model-only performance on the same inputs with segmentation precomputed to avoid differences in workload/token length.
- The ratio of two separately measured p95 values is not an exact decomposition of per-request latency. For attribution, record each stage's timing within the same request.

### R11 — Medium: reproducibility is not fully pinned

Dependencies use broad lower bounds, pretrained model revisions are not pinned, and the dataset points to a moving branch. Stored SHA256 values are useful, but distinguish a fixed reference manifest from a new manifest generated on refetch. Local and Kaggle environments have different torch versions, and Kaggle lacks a git SHA. Store a source-snapshot hash in the archive.

`TrainConfig.run_id()` excludes learning rate, epochs, maximum length, and a configuration hash. Because `save_run()` writes to the same path, configuration changes that do not rename the recipe can overwrite artifacts. The current snapshot has no duplicate IDs, but the mechanism does not protect future runs. Fail if a run already exists, or generate IDs containing a hash and an attempt ID.

Only one file under `results/runs/` is tracked by Git. Locally available artifacts are not necessarily available to someone cloning the repository. Provide release artifacts or versioned downloads with checksums. Type checking in CI is currently advisory, and the defined jobs do not run training integration tests.

### R12 — Medium: narrow the benchmark comparison claims

[BamiBERT, Table 2](https://arxiv.org/html/2607.02259v1), reports UIT-VSFC sentiment F1 of 83.41 and **topic F1 of 79.90**. The categorical statement that the literature does not report topic F1 therefore needs correction. The inspected section labels the metric as F1 but does not provide enough detail to assume identical averaging, seed protocols, and model selection. Avoid calling it the “published best macro-F1” without sufficient verification.

Do not use the project's seed standard deviation alone to infer the statistical significance of a difference from a published point estimate. For a direct comparison, run the reference model in the same harness and state the budget.

## 3. Expand the Research Problem

Central question: **How can Vietnamese feedback classifiers improve minority-class performance, reliability under language variation, and learning efficiency within a limited compute budget?**

The project should answer a small number of questions convincingly. The catalog in Section 6 provides options, not a requirement to implement every method.

| Research question | Competing explanations | Evidence needed |
|---|---|---|
| Q1: What limits neutral performance? | Class imbalance, ambiguous annotations, weak representations, or decision boundaries | Label audit plus controlled loss/head/data interventions |
| Q2: What transfers beyond clean benchmark text? | Segmentation, lexical normalization, social-domain pretraining, or augmentation | Clean, natural-shift, and synthetic-challenge results reported separately |
| Q3: Which method uses labels most efficiently? | Sparse learning, frozen embeddings, full fine-tuning, or active selection | Repeated learning curves at equal annotation budgets |
| Q4: Does shared learning help sentiment and topic? | Useful task structure versus negative transfer | Separate-model controls, both task metrics, joint metrics, and training cost |
| Q5: Can uncertainty identify difficult examples? | Calibration error, model disagreement, or genuinely missing context | Reliability diagrams, selective-risk curves, per-class coverage, and OOD evaluation |
| Q6: What quality survives a tighter compute budget? | Better pretrained models, PEFT, smaller students, or quantization | Matched-budget comparisons and a quality/latency/memory Pareto frontier |

An optional task extension is **multi-aspect sentiment analysis**. For example, translated into English:

> “The teacher explains things clearly, but the room is hot and the projector often breaks.”

The desired labels include teaching/positive and facilities/negative. This exposes a limitation of assigning one sentiment and one topic to an entire sentence. Evaluate aspect detection and end-to-end aspect/polarity prediction offline; evidence-span extraction is a separate extension requiring span annotations.

UIT-VSFC provides sentence-level labels for two tasks; **it cannot directly supply ground-truth aspect spans or multiple sentiments within a sentence**. Correlated sentiment/topic labels support a multi-task hypothesis, not an ABSA ground truth.

| Research stage | Deliverable | Completion criterion |
|---|---|---|
| A — Trustworthy benchmark | Corrected claims, experiment identity, training checks, reproducible baseline tables | Every headline traces to a configuration, split, and artifact |
| B — Core investigation | Minority-class analysis, robustness, calibration, and controlled improvements | At least three questions answered, including inconclusive or negative findings |
| C — One extension | Low-label adaptation, active learning, distillation, or multi-aspect sentiment | Dedicated controls, budget accounting, and independent evaluation |

Stages A and B plus **one** Stage C extension are the recommended portfolio scope. CSV/JSON artifacts, a CLI, executed notebooks, figures, and a technical report are sufficient deliverables.

## 4. Dataset Extensions and How to Avoid Inflating Results

### A. UIT-VSFC: retain the original benchmark and add challenging evaluations

- Keep the official split for historical comparisons. The data report identified **55 test rows (1.74%) that overlap with train after normalization**. Add a predefined evaluation slice excluding overlap; do not remove or modify test examples based on model errors.
- Manually audit 100–200 train/dev examples, prioritizing neutral, others, and model disagreements. Use out-of-fold predictions to rank suspected label issues. Model disagreement does not prove that the gold label is wrong. Methodological reference: [Confident Learning](https://arxiv.org/abs/1911.00068).
- Categorize neutral examples into objective facts, no opinion, requests/suggestions, mixed sentiment, and insufficient context. Treat the labels as ordinal only if the audit supports that assumption; symmetry in a confusion matrix does not establish it.
- Build a challenge set of approximately 300–500 sentences covering negation, opposing sentiments across two aspects, abbreviations, missing diacritics, typos, code-switching, long sentences, and out-of-domain content. Distinguish naturally occurring text, manually authored examples, and synthetic transformations.
- Also create an **independent, naturally sampled set** of approximately 500–1,000 sentences if suitable data is available. An intentionally difficult challenge set does not represent operational error rates.
- Freeze a portion before testing methods. If old test examples inform augmentation design, label that work exploratory and confirm it on a new set.

### B. Multi-aspect education: a small project-built dataset

Annotate approximately 800–1,200 multi-aspect responses, starting with a 100-sentence pilot. The schema should include aspect category, polarity, and an evidence span where explicit evidence exists, with implicit/unclear states. Have two annotators independently label at least a 20–30% subset, measure agreement by label type, and adjudicate disagreements.

Split by near-duplicate text groups, source, or time when the metadata genuinely exists. Do not describe a split as temporal when the dataset has no timestamps. This small dataset supports a prototype; report class support and uncertainty rather than treating it as representative of every educational institution.

### C. ABSA in another domain: choose the correct dataset

The [official UIT-ViSFD dataset](https://github.com/LuongPhan/UIT-ViSFD) is the **Vietnamese Smartphone Feedback Dataset**, not student feedback: 11,122 comments, 10 aspects, and 3 polarities. It is suitable as a second ABSA benchmark. Preserve its original schema, evaluate it independently, and do not mix smartphone labels into education topics. Its README states research-use conditions.

For sentence-level sentiment transfer alone, select a corpus with compatible polarity labels and inspect its annotation guidelines first. Do not map emotion or hate-speech labels into sentiment based only on their names. Publicly fine-tuned sentiment checkpoints also require a training-data audit before use as controls, to avoid models that have already seen the benchmark.

### D. Separate three kinds of shift

- **Input variation with the same task:** missing diacritics, spelling errors, informal vocabulary, or a new source of student feedback. Check that transformations preserve the intended label.
- **Domain transfer with compatible labels:** education to another sentence-sentiment corpus. Report source-only transfer, target-only training, and source-to-target adaptation separately. A change in annotation policy can confound domain effects.
- **Out-of-scope input:** text for which the education classifier's labels are not meaningful. Evaluate detection/rejection separately; this is not simply another negative-sentiment class.

For data collection, record source, collection period when available, sampling rules, annotation instructions, duplicate groups, and dataset version. Split original examples before generating variants, and keep all paraphrases/perturbations of an example in the same partition. Use a fixed label-ID mapping and an explicit schema compatibility check for every dataset.

## 5. Model Ladder: Each Model Should Answer a Question

| Family | Candidate | Experimental purpose | Priority |
|---|---|---|---|
| Sparse | Existing word/character TF-IDF + logistic regression/LinearSVC | Low-cost, noise-tolerant control and lexical-error comparison | Required |
| Frozen embedding | `intfloat/multilingual-e5-small` + logistic regression | Separate representation quality from expensive end-to-end training | High for low-label work |
| Few-shot embedding | SetFit with a documented multilingual sentence encoder | Compare label efficiency against frozen embeddings and full fine-tuning | High for Q3 |
| Existing encoder | PhoBERT-base with VnCoreNLP and pyvi | Preserve established controls; distinguish segmentation choices | Required |
| Alternative pretraining | PhoBERT-base-v2 | Test broader pretraining while retaining a similar model scale | Optional |
| Social-domain encoder | `uitnlp/visobert` on its native input pipeline | Test informal/noisy Vietnamese and domain-specific pretraining | High once shift sets exist |
| Raw-text encoder | `Qualcomm-AI-Research/BamiBERT` | Test raw-input modeling and quality at a comparable encoder scale | High |
| Multilingual control | XLM-R-base on raw text | Revisit the current segmented-only comparison | Required before broad architecture claims |
| Larger encoder | Existing PhoBERT-large | Diagnose tuning/budget sensitivity; reuse existing results first | Low until cheaper hypotheses are tested |
| LLM reference | `Qwen/Qwen3-4B`, zero-shot and few-shot | Compare in-context learning on low-label and multi-aspect cases | One bounded comparison |
| Smaller student | A six-layer PhoBERT student, with/without distillation | Measure whether teacher knowledge improves the speed/quality frontier | Optional Q6 extension |

Source-backed motivation:

- [PhoBERT's official repository](https://github.com/VinAIResearch/PhoBERT) specifies segmentation requirements and different pretraining data/licensing for base-v2. Do not treat base and base-v2 as identical except for their names.
- [ViSoBERT](https://aclanthology.org/2023.emnlp-main.315/) targets Vietnamese social-media text. Its value here is a testable domain-fit hypothesis, not an assumed clean-benchmark advantage.
- The [BamiBERT model card](https://huggingface.co/Qualcomm-AI-Research/BamiBERT) specifies raw input and a transformers-version compatibility note. Use an isolated, pinned environment when needed.
- The [multilingual E5 model card](https://huggingface.co/intfloat/multilingual-e5-small?inference_provider=hf-inference) provides the encoding recipe. Fix pooling, normalization, and the input-prefix convention before comparison; changing them silently changes the baseline.
- [SetFit](https://arxiv.org/abs/2209.11055) motivates a few-shot comparison. It should compete on the same labeled examples, with all pair generation restricted to training data.
- For [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), fix revision, prompt, label definitions, thinking mode, decoding settings, and output limit. Do not assume a 4B model will fit comfortably on the laptop's 4 GB GPU; measure a small pilot on suitable hardware.

Start with two new encoder controls rather than loading every model in the table. Treat model-specific preprocessing as part of each pipeline. Report training budget and truncation coverage instead of enforcing an inappropriate common token length. An architecture comparison under a fixed budget answers a different question from a heavily tuned comparison; label which one you ran.

## 6. Experiment Catalog and Priorities

**P0:** validity prerequisites. **P1:** recommended first research cycle. **P2:** choose according to findings and role emphasis. **P3:** optional extension. Expected improvements are hypotheses; no gain is promised.

| ID | Priority | Hypothesis / intervention | Control and minimum comparison | Main outputs / decision |
|---|---|---|---|---|
| E01 | P0 | Current rankings depend on preprocessing or tuning budget | XLM-R raw versus existing segmented pipeline; LR candidates 1e-5/2e-5/5e-5 under a stated budget | Macro/per-class F1, truncation, runtime; narrow claims to tested conditions |
| E02 | P1 | Class imbalance contributes to neutral errors | CE versus sqrt class weighting, focal gamma 1/2, or training-time logit adjustment tau 0.5/1; stage the search | Neutral precision/recall/F1 and macro-F1; reject a recall gain that causes unacceptable precision loss |
| E03 | P1 | A biased classifier head is part of the bottleneck | Frozen encoder + balanced head retraining versus the original head and full CE fine-tuning | Quality, head-training cost, confusion changes |
| E04 | P1 | Training-label ambiguity/noise limits performance | OOF-ranked audit with human confirmation; original versus corrected training labels | Confirmed changes, inter-annotator agreement, per-class quality; preserve final-evaluation labels |
| E05 | P1 | Augmentation improves robustness without excessive clean-data loss | No augmentation versus controlled typo/diacritic/abbreviation augmentation at 10%/30% exposure | Clean and shifted scores; per-transformation failures and confidence intervals |
| E06 | P2 | Regularization improves stability | CE versus LLRD or R-Drop separately; FGM only after fixing R2 | Mean/std, compute, training behavior; do not combine methods before isolated ablations |
| E07 | P1 | Sentiment and topic benefit from shared representations | Two separate encoders versus one shared encoder with two heads; loss weights 0.3/1 | Both task scores, joint exact match, negative transfer, parameter count, runtime |
| E08 | P2 | Sparse and dense predictions contain complementary information | Each standalone model, simple probability averaging, then OOF stacking | Topic macro-F1 and facility/others F1; use calibrated probabilities where needed |
| E09 | P1 | Raw confidence is miscalibrated | Uncalibrated predictions versus temperature scaling on independent calibration data | NLL, Brier score, reliability plot, ECE with binning disclosed |
| E10 | P1 | Uncertainty identifies errors worth abstaining on | Maximum probability, margin, entropy, and optional ensemble disagreement | Risk–coverage, AURC, per-class coverage, and OOD false acceptance |
| E11 | P2 | Strong performance is possible with fewer labels | TF-IDF, frozen encoder, SetFit, and PhoBERT on identical subsamples | Learning curves, repeated subset uncertainty, class support, training cost |
| E12 | P2 | Active selection beats random labeling | Random versus uncertainty versus uncertainty+diversity at equal budgets | Quality versus label count; learning-curve area and labels needed for a fixed target |
| E13 | P2 | Unlabeled in-domain text improves adaptation | No continued pretraining versus bounded TAPT/DAPT using training-pool text only | In-domain/shifted scores, total GPU-hours, possible forgetting |
| E14 | P3 | Neutral has useful structural relationships with other labels | Softmax versus CORN or an objective/non-objective cascade, tested separately | Macro-F1, ordinal error metrics where justified; run only after annotation audit |
| E15 | P2 | Distillation improves a small student's quality | Same student trained on hard labels versus CE + teacher KL; teacher trained only on allowed data | Quality, minority degradation, size, RAM, CPU latency |
| E16 | P2 | Quantization and runtime choices improve offline efficiency | PyTorch FP32 → ORT FP32 → dynamic INT8; static INT8 only if justified | Logit/label parity, F1 degradation, p50/p95/p99, RSS, artifact size |
| E17 | P3 | Multi-aspect modeling captures mixed feedback better | Keyword/multi-label linear baseline → encoder per aspect → LLM reference | Aspect detection and end-to-end aspect/polarity F1; do not evaluate only on gold aspects |
| E18 | P3 | Class-aware contrastive learning helps representations | CE versus CE + supervised contrastive loss with sufficient same-class pairs | Weak-class quality, batch sensitivity, training cost; no claim based on embedding plots alone |
| E19 | P3 | Additional weak/synthetic labels are useful | Original data versus equal-sized random/repeated-data control versus human-audited synthetic/pseudo labels | Quality versus extra-label cost, acceptance rate, label preservation, duplication audit |
| E20 | P2 | Parameter-efficient tuning preserves useful quality | Full fine-tuning versus frozen encoder/head versus LoRA ranks 4/8/16 on the same backbone | F1, trainable parameters, peak VRAM, wall time, adapter size; not parameter count alone |
| E21 | P2 | Adaptation helps a second compatible domain | Source-only, target-only, pooled training, and source→target adaptation | Target-domain macro/per-class F1, source retention, label-budget accounting |
| E22 | P3 | Prediction sets offer useful uncertainty summaries | Calibrated classifier versus split-conformal prediction sets | Empirical coverage, set size, per-class coverage; assess shift sensitivity separately |

E02 is motivated by [logit adjustment](https://arxiv.org/abs/2007.07314); E03 by [decoupled representation/classifier training](https://arxiv.org/abs/1910.09217). The latter originated in long-tailed visual recognition, so transfer to this NLP task requires evidence. Distinguish training-time logit adjustment from post-hoc prior correction; verify the sign and inference rule against the chosen objective.

E06: [R-Drop](https://arxiv.org/abs/2106.14448) regularizes disagreement between dropout passes. Account for extra forward/backward computation. A gain that appears only after doubling compute should also be compared with a control given a similar compute budget.

E05: [ViLexNorm](https://aclanthology.org/2024.eacl-long.85/) and [ViSoLex](https://aclanthology.org/2025.coling-demos.18/) motivate lexical-normalization experiments. Compare no normalization, a frozen rule-based normalizer, and a learned normalizer only if resources permit. Measure sentiment preservation; better normalization scores do not automatically improve classification.

E09–E10 build on [calibration](https://proceedings.mlr.press/v70/guo17a.html). Scalar temperature scaling changes confidence but preserves argmax labels, so F1 improvement is not its success criterion. Abstention must report coverage by class: rejecting almost all neutral examples is not a satisfactory solution.

E12: [BADGE](https://arxiv.org/abs/1906.03671) motivates combining uncertainty and diversity. E13: [DAPT/TAPT](https://aclanthology.org/2020.acl-main.740/) motivates adaptation with unlabeled text. E14: [CORN](https://arxiv.org/abs/2111.08851) provides an ordinal alternative, but neutral may represent “no opinion” rather than intermediate polarity.

E15: [knowledge distillation](https://arxiv.org/abs/1503.02531) supplies the teacher/student framework. E20: [LoRA](https://arxiv.org/abs/2106.09685) supplies a parameter-efficient alternative. Neither method guarantees a better accuracy/compute trade-off on this dataset.

E22: [conformal prediction](https://arxiv.org/abs/2107.07511) provides coverage guarantees under stated assumptions such as exchangeability. Marginal coverage is not per-class coverage, a singleton prediction is not automatically correct, and an in-distribution guarantee does not automatically survive domain shift. This is optional after basic calibration is complete.

## 7. Five Detailed Study Designs Worth Prioritizing

### Study A — Diagnose neutral before optimizing it

1. Sample 100–200 train/dev examples, stratified across classes and disagreement patterns. Keep a random component so the audit does not only describe model-selected hard examples.
2. Write an annotation guide distinguishing neutral, mixed sentiment, suggestions, no opinion, and missing context. Label “ambiguous” separately from “incorrect gold.” Record original and adjudicated labels.
3. Generate OOF training predictions for suspected-label ranking. Report agreement and confirmed issue rates within the sampling strata; do not extrapolate a targeted audit directly to the whole corpus.
4. Compare CE, one selected imbalance method, and balanced head retraining. Hold tokenizer, preprocessing, training data, and checkpoint-selection rules constant.
5. Run data correction as a separate intervention, followed by a small 2×2 comparison only if warranted: original/corrected training labels × CE/selected loss.

Output: a neutral taxonomy, audit table, confusion matrices, and a clear answer about which intervention helped which error group. Finding that annotation ambiguity limits gains is a valid result. Do not redefine test labels to make the model appear better.

### Study B — Robustness to realistic Vietnamese variation

Use three independently reported evaluation layers: the original benchmark, a naturally sampled external set where available, and a deliberately constructed challenge set. Use [CheckList](https://aclanthology.org/2020.acl-main.442/) to distinguish invariance tests from meaning-changing tests.

| Phenomenon | Example test design | Evaluation caveat |
|---|---|---|
| Unicode/spacing | NFC/NFD and harmless whitespace variants | Confirm the pipeline normalizes consistently |
| Missing diacritics | Controlled partial/full removal | Removal can introduce lexical ambiguity; annotate uncertain cases |
| Informal spelling | Audited abbreviation and typo substitutions | Do not assume every replacement preserves polarity |
| Negation | Minimal pairs changing a positive statement to a negative one | Expected labels must change; this is not invariance |
| Mixed aspects | Positive teaching plus negative facilities | Sentence-level gold may be ambiguous; evaluate under a declared policy |
| Length/context | Add irrelevant context or move sentiment cues | Track truncation and evidence retention |
| Out-of-scope text | Non-feedback or unrelated-domain examples | Evaluate rejection, not forced sentiment correctness |

Compare a fixed PhoBERT control, one augmentation recipe, and one alternative encoder. Record delta from clean performance, transformation failure rate, minority-class effects, and the number of independent source sentences. Multiple variants of one sentence are correlated: bootstrap by original sentence/group, not by treating variants as independent examples.

### Study C — Learning efficiency and active learning

Use two separate learning-curve regimes:

- Balanced few-shot subsets, such as 8/16/32/64/128 examples per sentiment class, subject to available training support.
- Natural-prevalence subsets, such as 10/25/50/100% of the training set, reporting actual counts for every class.

Use at least three subset seeds and the same subsets for all compared methods. When reporting uncertainty, distinguish subset variation from model-training variation. A frozen encoder+LR baseline helps identify whether SetFit's benefit comes from adaptation rather than the pretrained embedding alone.

For an offline active-learning simulation, hide pool labels from acquisition. Start with a fixed labeled seed set, reveal labels in equal batches, and compare random selection with uncertainty and uncertainty+diversity. Keep the evaluation set outside the acquisition pool. Record whether every round starts from the same pretrained weights or continues training; match that policy across methods. Teacher/model initialization must not have been fine-tuned on the hidden pool labels.

Output: learning curves, class acquisition counts, and label counts required to reach a predeclared target. A target such as 25% fewer labels is a hypothesis to test. Offline oracle-label savings do not establish actual human time savings without an annotation-time study.

### Study D — Multi-task learning and task interference

Train a shared encoder with separate sentiment and topic heads using `L = L_sentiment + lambda * L_topic`. Start with lambda values 0.3 and 1. Compare against independently trained single-task models with the same input pipeline and document both per-task and total compute budgets.

Report sentiment macro-F1, topic macro-F1, both minority-class F1 scores, and joint exact match, where both predictions must be correct. Inspect conditional errors, such as neutral within the “others” topic, but use gold topic only for analysis—not as an input unavailable at inference. If conditioning sentiment on predicted topic, measure error propagation and compare with an unconditioned head.

If one task improves while the other declines, report negative transfer and consider weighting or separate models. Advanced gradient-balancing methods are justified only after a simple shared model demonstrates interference. The joint model is a research artifact and can be evaluated entirely from batch predictions.

### Study E — Compact encoders versus an LLM reference

Compare supervised encoders and an LLM under clearly distinguished supervision: full labeled training versus zero-shot/few-shot prompting. Use identical evaluation examples and label definitions. Few-shot demonstrations and retrieval candidates come only from the allowed training pool, with near-duplicate filtering.

For the LLM, freeze a small prompt-development budget, report sensitivity to at least two demonstration selections where feasible, and count invalid labels/format failures. Prefer a classification-only structured response for the baseline; evaluate explanations separately if added. Do not treat fluent reasoning as evidence of correct sentiment or faithful attribution.

Record input/output tokens, runtime, hardware/VRAM, and monetary cost if paid inference is used. State that public benchmark contamination may be unknown for a pretrained LLM. A new independent set strengthens the comparison but does not erase unknown pretraining provenance.

Optional follow-up: use an encoder as teacher for a smaller student, or use audited LLM labels on training-only unlabeled text. Compare against equal-size hard-label/repeated-data controls. Do not let LLM-generated labels define the final evaluation truth.

## 8. Evaluation Protocol for the Next Research Cycle

### 8.1 Split discipline and experiment provenance

1. Freeze the current benchmark as v1. Its test set has already been inspected; future work must not describe it as untouched.
2. Separate training, selection, calibration, and final confirmation. Use nested/OOF procedures when data is limited, ensuring that preprocessing learned from data, checkpoint selection, and hyperparameter selection respect outer holdouts.
3. A new partition drawn from old training data is not an untouched holdout for checkpoints previously fitted on that data. Retrain the complete procedure without outer-fold data when performing cross-validation. For stronger confirmation after extensive benchmark exploration, prefer newly collected independent data.
4. Keep official-split results for comparison and report any grouped/deduplicated evaluation separately. Do not replace the official test with a favorable custom split.
5. Register hypotheses, controls, primary metrics, search budget, and practical acceptance criteria before a sweep. Distinguish exploratory from confirmatory comparisons.
6. Save full config, code/source hash, dependency versions, model/tokenizer revision, data/split hashes, seed, checkpoint hash, predictions, and metrics. Use config hashes and unique attempt IDs to prevent artifact overwrites.
7. Log failed/OOM/diverged runs and discarded configurations. Search cost is part of the experiment cost, not just the winning model's final training time.

### 8.2 Metrics matched to the question

| Question | Primary measurement | Required supporting information |
|---|---|---|
| Classification quality | Macro-F1 | Per-class precision/recall/F1/support, accuracy, confusion matrix |
| Minority improvement | Neutral/others F1 together with macro-F1 | Precision–recall trade-off, confidence intervals, affected error categories |
| Model comparison | Paired difference on common evaluation examples | Seed variation, evaluation uncertainty, effect size, search budget |
| Calibration | NLL and Brier score | Reliability plot, ECE with bins/sample size stated, per-class behavior |
| Selective prediction | Risk at fixed coverage and coverage at fixed risk | AURC, class-wise coverage, accepted-example counts, uncertainty |
| OOD detection | AUROC/AUPRC with a declared positive class | Threshold-specific false acceptance/rejection, OOD prevalence and source |
| Robustness | Clean-to-shift performance change | Per-slice support, paired/grouped uncertainty, meaning-preservation audit |
| Label efficiency | Quality at a fixed label budget | Repeated-subset learning curves, actual class counts, total training cost |
| Multi-task learning | Both task macro-F1 scores | Joint exact match, negative transfer, total parameters and compute |
| Multi-aspect analysis | End-to-end aspect/polarity F1 | Aspect detection F1, micro/macro convention, rare-aspect support |
| Compression/efficiency | Quality versus latency/memory/size | Minority-class loss, parity, CPU configuration, repeated measurements |

For selective prediction, define **coverage** as the accepted fraction and **risk** as the error fraction among accepted predictions. Report class-wise coverage using gold labels during evaluation. A model that abstains mostly on the minority class can look good globally while failing the research objective.

ECE depends on binning and sample size; a lower ECE alone does not establish better uncertainty estimation. For ABSA, evaluating polarity only on gold aspects measures an easier subtask than end-to-end detection and classification—report both if used.

### 8.3 Uncertainty and inference

- Use a single-seed pilot to check correctness and feasibility. Use three seeds for exploration and five for finalists. Small one-seed score differences are not a reliable elimination rule.
- Report mean and sample standard deviation across seeds. Pair comparisons by experimental replicate where meaningful, without claiming identical initialization across architectures.
- Estimate paired score-difference intervals by resampling the same evaluation examples for both systems. For repeated variants, duplicated sources, or grouped data, resample independent groups.
- Seed-level tests and evaluation bootstrap answer different questions. Neither replaces the other. Hierarchical bootstrap may summarize both sources when its resampling assumptions match the experimental design; five seeds still provide limited information about training variability.
- Apply multiple-comparison correction to families of inferential comparisons, and limit adaptive searches. A corrected p-value does not undo extensive selection on the same dev set.
- A non-significant difference is not equivalence. For compression, predeclare a non-inferiority margin and assess the uncertainty of the difference against that margin.
- Avoid causal claims from uncontrolled model comparisons. Changing the encoder often changes tokenizer, pretraining data, and optimization behavior simultaneously.

### 8.4 Offline efficiency measurement

Use the reference CPU for comparable inference experiments. Precompute the exact inputs for model-only timing, then separately measure preprocessing + tokenization + model. Record warmup, sequence-length distribution, batch size, thread settings, precision, runtime version, and repeated-run variation.

Report single-example latency separately from batched throughput, with throughput labeled **texts/second**. Include peak process memory and total artifact size, accounting for external ONNX data files if present. Randomize or rotate configuration order to reduce confounding from temperature and background load. Measurements with unstable repeats should be flagged and rerun before a speedup claim.

Quantization calibration uses a designated training/calibration subset with the checkpoint's actual preprocessing. It must not reuse the final evaluation set to select a quantization recipe. Check FP32 numerical parity and INT8 quality on independent evaluation examples, including minority-class examples and variable batch/sequence lengths. No HTTP server is needed for this study.

## 9. Research Success Criteria and Stopping Rules

These are proposed decision rules, not achieved results or universal hiring requirements. Freeze or revise them explicitly before the relevant experiment, not after seeing the final test result.

| Area | Minimum defensible result | Stretch target to investigate |
|---|---|---|
| Reproducibility | Recreate the selected baseline from a pinned config and regenerate tables from artifacts | A clean-environment reproduction plus documented tolerance across hardware |
| Scientific contribution | Answer at least three research questions with controls and limitations | One result confirmed on a second dataset or independent natural sample |
| Minority quality | Explain dominant neutral/others errors and evaluate a targeted intervention | Neutral F1 +0.03 absolute with no more than 0.005 macro-F1 loss, supported by uncertainty analysis |
| Robustness | Publish clean and challenging-slice results with adequate support | Reduce a predefined slice's error rate by 20% relative while losing no more than 0.005 clean macro-F1 |
| Uncertainty | Compare calibrated and uncalibrated predictions on independent data | Accepted-set error ≤5% at ≥80% coverage, with acceptable class-wise coverage and intervals |
| Label efficiency | Repeated learning curves with fair baselines | Reach a predeclared quality target with ≥25% fewer labeled examples than random sampling |
| Efficiency | A reproducible quality/latency/memory comparison | ≥1.5× CPU speedup with macro-F1 loss ≤0.005 and neutral F1 loss ≤0.02 |
| Task expansion | One independently evaluated extension | Shared learning improves one task without material harm to the other, or an ABSA model beats its dedicated baseline |

Practical improvement thresholds are separate from statistical significance. If an interval remains too wide to assess the target, report insufficient evidence and consider collecting more independent examples. Do not increase the seed count as a substitute for additional evaluation examples.

Stopping rules:

- Stop a method immediately if its implementation fails correctness checks; fix the implementation before interpreting scores.
- Stop expanding a model family when matched-budget comparisons do not improve quality, robustness, or efficiency beyond the current frontier. Report the tested scope of that conclusion.
- Reject an intervention whose apparent gain is driven by leakage, invalid label-preservation assumptions, or reduced evaluation coverage that was not disclosed.
- Promote at most three finalists per cycle. Do not keep searching until a preferred method wins.
- Treat a well-supported negative result as a completed investigation. Explain which hypothesis it weakens and which alternatives remain untested.

## 10. Compute Budget and Experiment Sequencing

The laptop GPU and existing checkpoints are sufficient for substantial work, but the full catalog is much larger than one research cycle. Use staged budgets.

### Cycle 0 — Correctness and diagnostics

Correct the citation/statistical claims, verify FGM/AMP and accumulation if that method remains enabled, protect experiment IDs, and regenerate the current result tables. Audit errors, plot calibration, and analyze disagreement from saved predictions. These activities mostly require CPU time; OOF prediction generation can require retraining and must be budgeted explicitly.

### Cycle 1 — At most 30 new full fine-tuning runs

An example allocation is **six configurations × three seeds = 18 runs**, then **two additional seeds for three finalists = 6 runs**, with **six reserved runs** for replication, failed pilots, or necessary controls.

The six configurations might cover the current PhoBERT control, a correctly preprocessed XLM-R control, one new encoder, one imbalance method, one augmentation recipe, and one multi-task model. Existing runs can substitute only when data, software, configuration, and artifact provenance make the comparison valid.

This is not enough to exhaust every listed learning rate, loss, and model combination. Any pilot that updates model weights counts toward the run ledger. If model-specific tuning consumes more runs, reduce the number of families instead of hiding the search cost. A multi-task run and a two-model control have different costs; track GPU-hours as well as run counts.

### Cycle 2 — Choose one specialization

- **Data/label efficiency:** learning curves plus a bounded active-learning simulation.
- **Efficient modeling:** LoRA or distillation plus offline inference benchmarking.
- **Task expansion:** multi-aspect sentiment with a separate annotation/evaluation protocol.
- **Transfer:** one compatible second domain with source-only, target-only, and adaptation controls.

Each branch needs its own pilot-derived budget. Active-learning rounds, nested evaluations, and per-budget learning curves can require many more fits than a normal sweep. Estimate their costs before selecting the branch.

Track wall time, GPU-hours, peak VRAM, CPU inference latency, and storage. Do not assign an assumed fixed duration to every run: model size, sequence length, R-Drop, continued pretraining, and LLM inference have different cost profiles. External compute may be used for training; keep comparable CPU benchmarks on the documented reference machine.

## 11. Evidence That Matters for Junior ML, AI, and Data Science Roles

The following is a reviewer-proposed portfolio rubric, not a survey of all employers. A higher benchmark score alone is not the completion criterion. Strong evidence shows that the author can formulate a question, build a correct comparison, interpret uncertainty, and explain a technical decision.

| Role | Highest-value evidence from this project | Recommended specialization |
|---|---|---|
| Junior ML Engineer | Correct training loop, reproducible configs/artifacts, controlled ablations, profiling, quality/cost trade-offs | Multi-task learning, LoRA/full fine-tuning comparison, or distillation |
| Junior AI Engineer | Encoder-versus-LLM comparison, prompt evaluation, constrained outputs, data provenance, cost and failure analysis | Bounded LLM reference plus an audited weak-label or teacher/student experiment |
| Junior Data Scientist | Sampling design, annotation audit, uncertainty, paired comparisons, class imbalance, and data-efficiency curves | Neutral-label investigation, calibration, and active learning |

Shared completion checklist:

- A concise problem statement with 3–5 research questions and clearly bounded scope.
- Strong sparse and neural baselines, with exact configurations and a reproducible evaluation harness.
- At least three controlled investigations, including a documented negative or inconclusive finding where that is what the evidence shows.
- A manually inspected error taxonomy covering approximately 60–100 cases, with sampling explained.
- One robustness/generalization evaluation beyond the original aggregate benchmark score.
- Seed variation and evaluation uncertainty reported without treating them as interchangeable.
- One extension completed deeply enough to explain design choices and failure cases.
- Executed notebooks or CLI-generated figures, concise results tables, and accessible artifact metadata/checksums.
- A short technical report that separates observations, hypotheses, decisions, and limitations.

No backend, frontend, dashboard, or online deployment is required for this milestone. A reproducible notebook walkthrough or a short terminal/notebook recording can demonstrate the research workflow.

Suggested figures:

1. Per-class F1 and confusion matrices for the baseline and finalist.
2. A paired-difference plot showing uncertainty for the main ablations.
3. Learning curves by labeled-data budget, if Q3 is selected.
4. Clean-versus-shift performance by phenomenon, including support counts.
5. Reliability diagrams and risk–coverage curves.
6. A quality-versus-CPU-latency plot, with memory/size reported alongside it if Q6 is selected.

Potential current CV statement, supported by existing results after correcting the documentation:

> Built a reproducible Vietnamese feedback classification benchmark on UIT-VSFC; PhoBERT + VnCoreNLP achieved sentiment test macro-F1 of 0.837 ± 0.003 across five seeds versus 0.745 for TF-IDF, with minority-class analysis and preprocessing ablations.

Future statements should describe measured improvements and their conditions: for example, a demonstrated reduction in labels at a fixed target or a measured CPU speedup within a stated F1-loss budget. Do not put proposed targets on a CV as completed results. If citing the pyvi pipeline, use its own 0.829 ± 0.011 score.

Questions the finished project should prepare the author to answer:

- Why was macro-F1 selected, and what does it still fail to capture?
- How do you know a reported gain is not selection bias or label leakage?
- What distinguishes ambiguous annotation from a model error?
- Why did a larger or newer model fail under the tested setup?
- What did you learn from a negative result, and what experiment followed?
- How does the best method change under a label, compute, or robustness constraint?
- Which conclusions would change if the deployment population differed from UIT-VSFC?

## 12. Six-Week Research Roadmap

| Time | Main work | Exit criterion |
|---|---|---|
| Week 1 | Correct claims, validate relevant training branches, freeze baseline configs, protect run identities | Trustworthy baseline table and a written experiment protocol |
| Week 2 | Audit neutral/others, categorize errors, define challenge sets, inspect existing predictions | Error taxonomy, versioned evaluation slices, and 3–5 selected hypotheses |
| Week 3 | Run the bounded three-seed exploration cycle | Comparable ablation table, failure log, and at most three finalists |
| Week 4 | Complete five-seed finalists; evaluate calibration and robustness | Independent quality/uncertainty analysis and a decision on each hypothesis |
| Week 5 | Complete one chosen specialization from Cycle 2 | Dedicated controls and a substantive result for the extension |
| Week 6 | Perform locked confirmation, regenerate figures/tables, write the report and artifact guide | A reproducible research package and evidence aligned with the target roles |

This is a planning estimate. New data collection, double annotation, nested evaluations, or active-learning loops may extend the schedule. A four-week version should complete Weeks 1–4 and publish that evidence before attempting an extension. A two-week version should prioritize correctness, error analysis, calibration, and one carefully controlled intervention using existing checkpoints where appropriate.

Suggested implementation backlog, not files already implemented:

| Proposed module/artifact | Purpose | Priority |
|---|---|---|
| `evaluation/compare_runs.py` | Regenerate result tables and paired comparisons from registry/predictions | P0 |
| `evaluation/error_analysis.py` | Stratified error sampling, OOF disagreement, annotation export | P1 |
| `evaluation/calibration.py` | Temperature fitting and independent uncertainty evaluation | P1 |
| `evaluation/robustness.py` | Versioned transformations, original-example grouping, slice reports | P1 |
| `training/multitask.py` | Shared encoder, two heads, explicit task weighting | P1 if Q4 is selected |
| `training/low_resource.py` | Reusable subset manifests and label-budget comparisons | P2 |
| `training/distillation.py` | Teacher/student objectives and matched student controls | P2 |
| `evaluation/llm_reference.py` | Frozen prompts, output validation, cost/error accounting | P2 |
| `configs/experiments/` | Full experiment definitions including search and compute budgets | P0 |
| `results/studies/` | Generated tables, figures, predictions, and provenance for each research question | P1 |
| `docs/ANNOTATION_GUIDE.md` | Label definitions, ambiguity policy, agreement/adjudication rules | P1 |
| `docs/RESEARCH_REPORT.md` | Methods, findings, negative results, limitations, reproducibility | Final deliverable |

First five concrete tasks:

1. Correct the segmentation-paper interpretation and distinguish VnCoreNLP/pyvi results in the headline table.
2. Validate or disable the flawed FGM branch before any FGM sweep; ensure config changes cannot silently overwrite run artifacts.
3. Export a stratified train/dev error sample with predictions and uncertainty, and write the neutral-label audit guide.
4. Freeze one baseline, the calibration protocol, and the initial robustness-slice definitions.
5. Select three primary hypotheses and launch only the controls needed to answer them, within the declared run budget.

The central research workflow is **observe an error → propose competing explanations → design a controlled intervention → evaluate independently → record a decision and its limits**. This provides a coherent technical portfolio across ML engineering, AI engineering, and data science without requiring product development.

