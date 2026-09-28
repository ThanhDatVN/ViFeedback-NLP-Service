# Next Plan — v5: one work list after H7

**v5, 2026-09-28.** Since v4:
- **H7 is decided.** Qwen3-4B ran on Kaggle, and `study h7-decide` applied the rule.
- **Publishing is settled.** The licences were checked on the Hub, and the owner chose to publish the
  model as a non-commercial research artifact.
- **Every open problem was researched.** The sources are in § 6.

v5 merges four things into **one prioritized work list**: v4's register of open items, Cycle 4, the
earlier development plan (the ROADMAP targets, track B, engineering), and the research findings.
The Cycle 3 outcome is kept as
[Appendix A](#appendix-a--cycle-3-outcome-v3-closed-2026-09-28), and the v2 register as
[Appendix B](#appendix-b--v2-register-closed-2026-09-27).

Status key: ✅ done · ⏳ next, no owner input needed · 👤 needs the owner · ⏸ deferred, with the reason · ✖ dropped, with the reason.

---

## 0. State after Cycle 4 (2026-09-28, commit cfc571e)

**Done since v5 was written.**
- A1: served p95 is 26.9 ms.
- A2 (ADR-032): `in_scope` tracks the institution, not the topic.
- A3, A5, and B1 (`cycle4.yaml`).
- B2 H8 not passed (ADR-033). Two heads lift NEU-ESC test by +0.111, but the stripped-text check
  and the refitted out-of-scope score fail, so the service is unchanged.
- F1: the image is 519 MB, so S8 is met.

**Targets still not met:**
- S1 0.830 (target 0.84)
- S2 0.804 (target 0.83)
- S4 neutral 0.576 (target 0.65)
- S7 540 MB (target 200)
- S9 0 coded errors (target 30)
- S10 not timed

**Problems found this cycle:**
- `in_scope` cannot filter other institutions' text (within NEU-ESC AUROC 0.573).
- Two-head training costs 0.0107 on stripped UIT-VSFC text.
- NEU-ESC test has now served H8 (two calls, both logged), so a new claim about other institutions
  needs new labelled data.
- The two-head model is exploratory and is neither served nor published.

**Next, in order.** No step needs Kaggle; everything fits the laptop.

| # | Step | Needs | Rule |
|---|---|---|---|
| 1 | **A4 upload** of the served model (bundle built, card with the ADR-032 caveat) | 👤 one command | — |
| 2 | **F2** S10: a CI job fetches the published model and reproduces validation macro-F1 on CPU | step 1 | < 15 min; same numbers as the manifest |
| 3 | **B4′** topic-aware scope detector: a small head on the encoder feature, trained on NEU-ESC train topics (in-scope vs off-topic) with UIT-VSFC as in scope | declare first | within-source AUROC ≥ 0.85; UIT-VSFC validation flagged ≤ 5% |
| 4 | **E1** INT8 power check on UIT-VSFC + NEU-ESC validation; then S5′ or distillation (E2) | — | as in Track E |
| 5 | Cycle 5 specialization: real typing (C1 consistency training on ViLexNorm pairs) or size (E2) | owner's choice | declared in `cycle5.yaml` |
| 6 | Neutral audit → D2 | 👤 6–8 h | the frozen tree |
| 7 | A multi-institution service (the NEU-ESC head reaches 0.76) | 👤 decision 2 and new labelled data from another institution | a new declared rule |

---

## 1. Where things stand

**The service.** It runs the H2-augmented PhoBERT-base with three additions:
- it lowercases input;
- it restores diacritics on unaccented input (ADR-031);
- it returns an `in_scope` score next to the label (ADR-031).

Every part is SHA-256-checked, `docker-e2e` passes, and CI is green. On UIT-VSFC the service scores
0.830 test macro-F1 (5 seeds). On real posts from another university (NEU-ESC) it scores 0.462.

**What H7 settled.** Both LLM arms beat the encoder on neutral, on constructed text only (Holm
p 0.0004 each). On real text:
- gpt-4o-mini, zero-shot, reaches 0.604 on NEU-ESC. It is not deployable, because student text cannot
  be sent to a third party without consent.
- Qwen3-4B, which could run locally, reaches only 0.475 on NEU-ESC (vs 0.462 for the served model,
  p 0.06) and 0.816 on UIT-VSFC validation (vs 0.86 for the encoder).
- So no LLM route closes the gap cheaply. The measured lever that is left is in-domain training
  data: the NEU-ESC authors' PhoBERT-base-v2, trained on NEU-ESC, reaches 77.7 macro-F1 on its four
  classes [S1].

**Targets not met** (ROADMAP):

| Target | Now | Work item |
|---|---|---|
| S1 sentiment test ≥ 0.84 | 0.830 served, 0.837 best | Not chased (§ 5) |
| S2 topic test ≥ 0.83 | 0.804 | Not chased (§ 5) |
| S4 neutral F1 ≥ 0.65 | 0.576 (UIT-VSFC test), 0.474 (NEU-ESC) | D1–D2 👤 |
| S7 artifact ≤ 200 MB | 540 MB | E1, E2 |
| S8 image ≤ 700 MB | ✅ 519 MB (was 1.02 GB) | F1 done |
| S9 ≥ 30 coded errors | 0 | D1 👤 |
| S10 clean-clone evaluation < 15 min | not timed | F2 |

**Holes found in the v4 review** (the IDs used below):

| ID | Hole | Work item |
|---|---|---|
| G1 | Served-pipeline latency unmeasured since ADR-031 | A1 |
| G2 | `in_scope` never checked on another institution's in-scope text | A2, B4 |
| G3 | NEU-ESC test reused (external evaluation, three v5 rules, H7) | B1: validation for selection, test once per rule, logged |
| G4 | H7 never applied | ✅ decided 2026-09-28 |
| G5 | `kaggle_train.ipynb` saves corpus files with the Kaggle version | A5 |
| G6 | NEU-ESC's label policy differs from UIT-VSFC's | A3, B2 (b), decision 2 |
| G7 | Training on NEU-ESC off-topic posts would disable the out-of-scope score | B1, B4 |
| G8 | Real typing flips 17% of labels (ViLexNorm) | Track C (Cycle 5) |
| G9 | The contrast-sentence drop is confirmed on v1 only | Challenge v2 (optional) |
| G10 | The rollback copy has no features output; the CE copy is gone | Documented; re-export on demand |

---

## 2. The work list

Each item gives the gap it closes (v4's register IDs: G = hole, S = target), the method and its
source, and the rule fixed before the run. Items marked **declare** get their rule frozen in
`cycle4.yaml` before any run.

### Track A — close now *(CPU or none; no training)*

| ID | Task | Closes | Method | Rule | Status |
|---|---|---|---|---|---|
| **A1** | Latency of the pipeline that is served | G1, C4 | `serve bench` times lowercase → restorer → pyvi → ONNX logits + features → Mahalanobis, on clean test and stripped test, in three sessions | p95 ≤ 30 ms (unchanged); ADR-031's additions ≤ 5 ms at p95 | ✅ met: served p95 26.9 ms (median of 3 sessions, range 23.4–39.5); additions +2.2 ms; unaccented input 4 ms faster (ADR-032) |
| **A2** | `in_scope` on other institutions' in-scope text | G2 | Flag rate by NEU-ESC topic on validation. Fine-tuned Mahalanobis separates *semantic* shift (off-topic) well and *background* shift (another institution's in-scope text) poorly [S3, S4, S5], which is what the service wants | Descriptive. If Academic or Service posts are flagged at > 10% (twice validation's 5%), the limit goes into the card and the API docs, and B4 refits the score | ✅ **triggered**: Academic 75.5%, Service 64.6% flagged; within NEU-ESC AUROC 0.573: the score tracks the institution, not the topic (ADR-032). Card and README corrected |
| **A3** | Label-policy description | G6 | The served model's confusion on NEU-ESC validation, by gold label and topic | Descriptive; input to owner decision 2 | ✅ 67% of gold-neutral in-scope posts are called polar, mostly negative (1,172 of 2,054) |
| **A4** | Publish the model | S-R11 | Card: CC BY-NC 4.0 weights, intended use, limits; bundle dry run built (`models/publish/vifeedback-sentiment-phobert`, 15 files, 1.09 GB) | The owner reviews the card and uploads | ✅ bundle · 👤 upload |
| **A5** | `kaggle_train.ipynb` clones into `/kaggle/working` | G5 | Clone into `/tmp`, as the H7 notebook does, with the same guard test | The guard test passes | ✅ `/tmp/repo`; the zip leaves out test predictions and `local/` |

### Track B — Cycle 4: student text from other institutions *(the specialization)*

| ID | Task | Closes | Method | Rule | Status |
|---|---|---|---|---|---|
| **B1** | Declare `cycle4.yaml` | G3, G7 | Data roles as in v4 § 3: in-scope NEU-ESC train only (21,113 posts, four off-topic topics excluded); NEU-ESC validation for every selection; NEU-ESC test once per rule, logged; UIT-VSFC validation for non-inferiority; UIT-VSFC test unused | Committed before the first B2 run | ✅ `cycle4.yaml` v1 (c085091), v2 reporting-only (ADR-032) |
| **B2** | **H8: in-domain training data** | the NEU-ESC gap; G6 | Three candidate recipes, each at one seed on NEU-ESC validation; the best goes to 5 seeds against the served recipe (control checkpoints exist). **(a) mixed**: UIT-VSFC + NEU-ESC train, one head, NEU-ESC labels mapped as in v5. **(b) two heads**: one shared encoder with a UIT-VSFC head and a NEU-ESC head, so each dataset keeps its own label policy [S9, S1's multitask setup]; the service keeps the UIT-VSFC head. **(c) sequential**: the served checkpoint fine-tuned on NEU-ESC at a lower learning rate. Reference ceiling: 77.7 mF1 on four classes [S1] | **declare**. NEU-ESC test macro-F1 up with a paired CI above 0; UIT-VSFC validation macro-F1 not lower by more than 0.005 and neutral F1 not lower by more than 0.02 (5-seed means); stripped validation with the restorer not lower by more than 0.01; out-of-scope AUROC on NEU-ESC off-topic ≥ 0.90 | ✅ **not passed** (ADR-033). Two heads chosen; at 5 seeds NEU-ESC test +0.111 [+0.102, +0.119] (5/5 seeds), UIT-VSFC −0.0045, neutral −0.010, but stripped −0.0107 (limit 0.01) and the refitted out-of-scope score AUROC 0.886 with 31% of in-scope posts flagged. One head for both label policies hurts UIT-VSFC; two heads do not |
| **B3** | Backbone for informal text *(conditional)* | NEU-ESC gap, G8 | PhoBERT-base-v2 or ViSoBERT under B2's winning recipe. ViSoBERT is pretrained on social media; on NEU-ESC it reaches 75.8 mF1 alone and 77.2 in the authors' best multitask setup, against PhoBERT-base-v2's 77.7 and 77.5 [S1]. One axis: only the backbone changes | **declare**; only if B2 passes. Same rule as B2, against B2's model | ✖ not triggered (B2 not passed) |
| **B4** | Refit the out-of-scope score | G2, G7 | Mahalanobis means and covariance from UIT-VSFC train plus in-scope NEU-ESC train features, so another institution's in-scope text counts as in-domain [S3, S4] | Off-topic AUROC ≥ 0.90; in-scope NEU-ESC flag rate ≤ 10% | ⏳ with B2's release, or alone if A2 fails |
| **B4′** | A topic-aware scope detector | G2 | The Mahalanobis score measures resemblance to the training text (ADR-032). Candidates: a small head on the encoder feature trained to tell in-scope NEU-ESC topics from off-topic ones (NEU-ESC train has topic labels), UIT-VSFC counted as in scope | **declare**. Within-source AUROC (NEU-ESC test, in-scope vs off-topic) ≥ 0.85; UIT-VSFC validation flagged ≤ 5% | ⏸ after B2 |
| **B5** | Release | — | Through the gate; the restorer and the score are re-accepted; ADR; card | Release gate | ✖ no release (ADR-033) |
| ~~H9~~ | Local LLM router | — | — | — | ✖ H7: Qwen3-4B is not ahead on real text (+0.014 [−0.001, +0.028]) |

### Track C — robustness to real typing *(G8; the next specialization, Cycle 5)*

| ID | Task | Method | Rule | Status |
|---|---|---|---|---|
| **C1** | Consistency training on real typing | The instability is between a real comment and its normalized form (17% flips on ViLexNorm), not in spelling alone (S2a failed). Train with an invariance loss between each ViLexNorm train pair's two forms (stability training, as in R-Drop-style consistency) | **declare** (Cycle 5). ViLexNorm test flips down with a paired CI above 0; UIT-VSFC validation non-inferior at 0.005 | ⏸ Cycle 5 |
| **C2** | A normalization front-end | A BARTpho normalizer trained on ViLexNorm reaches 57.7% error reduction and helps downstream tasks [S6]; ViSoLex packages one [S7]. Measure the flip reduction and the CPU latency. ViLexNorm is CC BY-NC-SA: a shipped normalizer must carry that licence | Flips down with CI; p95 ≤ 30 ms in total | ⏸ Cycle 5; latency is the likely blocker |

### Track D — neutral on UIT-VSFC *(S4, S9; owner-gated)*

| ID | Task | Method | Status |
|---|---|---|---|
| **D1** | Neutral audit | The 160-row sheet; `study audit-report` applies the tree frozen in `cycle2.yaml` | 👤 6–8 h |
| **D2** | The audit's branch | E04 label correction, **or** a soft-label run, **or** one representation experiment. For soft labels, training on soft and gold labels together is the robust choice when items have few annotations [S8]. The audit's second pass supplies the disagreement | ⏸ after D1 |

### Track E — size and speed *(S7, efficiency stretch)*

| ID | Task | Method | Rule | Status |
|---|---|---|---|---|
| **E1** | Power check for careful INT8 | From INT8's disagreement pattern (1.35% of labels, 4.1% of neutral on the held-out train subset), the upper bound reachable on UIT-VSFC validation + NEU-ESC validation (4,888 labelled) | Declare **S5′** (same 0.005 margin, larger set, ADR) only if it is demonstrable | ⏳ after B5 decides the model |
| **E2** | Distillation into a 6-layer student *(if E1 cannot pass)* | Task-specific distillation from the served model; 6-layer students keep > 99% of the teacher at about 2× speed on GLUE-type tasks [S10] | **declare**; the release gate plus ≤ 200 MB | ⏸ Cycle 5+ |

### Track F — engineering

| ID | Task | Method | Rule | Status |
|---|---|---|---|---|
| **F1** | Image ≤ 700 MB (S8) | Measured in the local environment: pyvi pulls scikit-learn (45 MB) and SciPy (118 MB) only through `sklearn-crfsuite`; the CRF itself needs `python-crfsuite` (1.5 MB). `transformers` (107 MB) is there only for PhoBERT's BPE tokenizer. Load pyvi's CRF with `python-crfsuite` directly, and read `bpe.codes`/`vocab.txt` with a small tokenizer | Byte-identical segmentation and token ids on all UIT-VSFC and NEU-ESC text; release parity; `docker-e2e` passes; image ≤ 700 MB | ✅ **519 MB** (1,023 → 750 without scikit-learn/SciPy → 519 without transformers). Segmentation and token ids identical on all 49,141 texts; `docker-e2e` passes with the same confidences |
| **F2** | S10 timing | A CI job times the clean-clone evaluation path on CPU | Recorded; < 15 min | ⏳ |
| **F3** | Model card after every serving change | `serve publish` dry run | — | ongoing |

### Not pursued

| Item | Why |
|---|---|
| Sweeps for S1/S2 on the old validation set | S1 is inside seed noise; S2's two interventions failed; the test split is spent |
| An LLM in the serving path (API or router) | Consent for API calls; a local 4B model is not ahead on real text (H7) |
| Challenge v2 | Replaced as the confirmation set by NEU-ESC (ADR-030); optional if the owner ever has the people |
| U3 (API teencode candidates) | Superseded by ViLexNorm |

---

## 3. Order

| Step | Without the owner | The owner |
|---|---|---|
| 1 | A1, A2, A3 (CPU); A5; B1 | A4 upload; decision 2 |
| 2 | B2: three recipes at one seed, then the winner at 5 seeds; B4 | D1 audit |
| 3 | B2 confirmation on NEU-ESC test; B5 release if it passes; F1 | U2 review |
| 4 | E1 on the released model; F2; the D2 branch once D1 is done | Review the new card; upload |
| later | B3 (if B2 passed); Cycle 5: C1/C2, or E2 | — |

## 4. Budget

| Resource | Estimate |
|---|---|
| Weight-updating runs | B2 ≤ 8 (3 selection + 5), B3 ≤ 6, D2 ≤ 12: ≤ 26 in Cycle 4 |
| Laptop GPU | B2 ≈ 3 h (34k training posts ≈ 3× UIT-VSFC per seed), one job at a time with cool-downs |
| Kaggle | not needed; optional for B2 if the laptop is busy |
| API | none |

## 5. What not to do

- No selection on NEU-ESC test, and no NEU-ESC off-topic post in any training set.
- No change to what the service's labels mean without owner decision 2. With two heads (B2 b), the
  UIT-VSFC head stays the served one.
- No UIT-VSFC text to an API; no LLM output as a gold label; no LLM in the CPU service.
- No sweep for S1 or S2; no decision on the official test.
- One specialization per cycle: real typing (C) and distillation (E2) wait for Cycle 5.

## 6. Research sources

| | Source | Used for |
|---|---|---|
| S1 | Mai et al., 2025. *NEU-ESC: A Comprehensive Vietnamese dataset for Educational Sentiment analysis and topic Classification toward multitask learning*. [arXiv:2506.23524](https://arxiv.org/abs/2506.23524). Table 6: PhoBERT-base-v2 81.75 accuracy / 77.70 mF1 (4 classes); ViSoBERT 75.79; GPT-4o zero-shot 49.70; Claude 4 few-shot 70.89; posts from Facebook forums, more slang than UIT-VSFC | B2 ceiling, B3, the two-head design |
| S2 | Gururangan et al., 2020. *Don't Stop Pretraining*. [ACL 2020](https://aclanthology.org/2020.acl-main.740/) | Domain/task-adaptive pretraining, a B3 alternative |
| S3 | Podolskiy et al., 2021. *Revisiting Mahalanobis Distance for Transformer-Based Out-of-Domain Detection*. [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17612) | Why Mahalanobis on fine-tuned features (ADR-031), B4 |
| S4 | Arora et al., 2021. *Types of Out-of-Distribution Texts and How to Detect Them*. [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.835/) | Background vs semantic shift: A2, B4 |
| S5 | *Fine-Tuning Deteriorates General Textual Out-of-Distribution Detection by Distorting Task-Agnostic Features*. [arXiv:2301.12715](https://arxiv.org/abs/2301.12715) | Fine-tuned features catch semantic shift, miss non-semantic shift: A2 |
| S6 | Nguyen et al., 2024. *ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text*. [EACL 2024](https://aclanthology.org/2024.eacl-long.85/). Best ERR 57.74% (BARTpho); normalization helps downstream tasks; research use only | C2 |
| S7 | *ViSoLex: An Open-Source Repository for Vietnamese Social Media Lexical Normalization*. [COLING 2025 demos](https://aclanthology.org/2025.coling-demos.18/) | C2 |
| S8 | Uma et al., 2021. *Learning from Disagreement: A Survey*. JAIR 72 | D2 soft labels |
| S9 | Dataset-specific heads over a shared encoder for inconsistent label spaces, e.g. [Plain-Det, 2024](https://arxiv.org/abs/2407.10083); multi-dataset hate-speech training, [arXiv:2208.10598](https://arxiv.org/abs/2208.10598) | B2 (b) |
| S10 | Wang et al., 2020. *MiniLM*. [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf): a 6-layer student about 2× faster with > 99% of the teacher | E2 |

## 7. Decisions needed from the owner

| # | Decision | Default until decided |
|---|---|---|
| 1 | Upload the model: `vifeedback serve publish --repo-id Datk4/vifeedback-sentiment-phobert --upload`, after reading `models/publish/vifeedback-sentiment-phobert/README.md`. The token must be allowed to create and write model repositories | Not published |
| 2 | Label policy for other institutions' text: keep UIT-VSFC's (two heads in B2 keep it exactly) or adopt NEU-ESC's | Keep UIT-VSFC's |
| 3 | Neutral audit (D1); a second annotator? | D2 waits |
| 4 | U2 label review (15 rows) | v1 stays as frozen |
| 5 | Challenge v2 | Not written |

Settled on 2026-09-28:
- NEU-ESC may be used for training and published with attribution. Its card is Apache-2.0, and its
  gate adds no condition.
- The owner publishes the model as a non-commercial research artifact, so the weights carry
  CC BY-NC 4.0 in line with UIT-VSFC's research-purpose release.

---

## 8. Kaggle

H7 is done. The Kaggle run took 19–28 s per 1,000 prompts on two T4s, and the prefix cache was used
in all seven configurations, with self-check differences of 0.05–0.10. The notebook stays as it is,
verified, for any rerun. Cycle 4 needs no Kaggle: PhoBERT-base training fits the laptop.
`kaggle_train.ipynb` gets the `/tmp` fix (A5) before any use.

---

## Appendix A — Cycle 3 outcome (v3, closed 2026-09-28)

The v3 progress table as it stood when Cycle 3 closed. The rules are in `cycle3.yaml` v1–v6.

Every step that needed no owner input has run. Development results are not confirmations. The
human-typed challenge v2 needs time and people the owner does not have now, so the confirmation moved
to **NEU-ESC test** (real forum posts from another university, human labels; ADR-030, `cycle3.yaml`
v5), with rules committed before the cells were computed. Challenge v2 stays optional.

| Step | Result | Status |
|---|---|---|
| V1: are the Cycle 2 drops real? (5 seeds, all reproduce exactly) | Teencode: noise (3/5 seeds, p = 0.09). Short factual: not confirmed (3/5, p = 0.02). **Contrast sentences: confirmed drop** 0.875 → 0.800 (4/5, p = 0.004, Holm-robust). Unaccented gain holds in 5/5 | ✅ |
| Evaluation-data research | [EVALUATION_DATA.md](EVALUATION_DATA.md): CheckList matrix, metrics, power, NEU-ESC, ViLexNorm, EduPulse cross-check, v2 protocol | ✅ |
| Case invariance (declared rule) | Served model flipped 1.07% of labels on capitalized input → **the service now lowercases** (0 flips, 0 validation changes) | ✅ |
| ViLexNorm invariance (real typing) | 17% of labels flip between a real comment and its human normalization, for CE and augmented alike | ✅ measured |
| U1 gpt-4o-mini on challenge v1 | neutral F1 0.955 vs 0.713 (CE); confounded by v1's construction; same pattern as EduPulse | ✅ |
| U2 label check | 15 rows for owner review (`results/studies/challenge/label_review_v1.csv`) | 👤 review |
| H7 gpt-4o-mini on NEU-ESC (zero-shot, frozen prompt) | macro-F1 **0.604** vs CE 0.494 (+0.109 [+0.094, +0.124]) and augmented 0.462 (+0.142) at seed 42; neutral F1 0.769 vs 0.526 / 0.474; over-calls negative (precision 0.43, recall 0.80). USD 0.21 for 6,613 posts | ✅ measured; reference only |
| **S2a** real-typing lexicon (ViLexNorm train) | Development gate **not passed**: ViLexNorm flips 0.169 → 0.177 (p = 0.13); validation +0.003 | ✅ negative |
| **S2b** diacritic restoration (train-only) | Stripped validation macro-F1 **0.686 → 0.857**; clean predictions unchanged; challenge unaccented 0.74 → 0.86; 0.2 ms. NEU-ESC confirmation: unaccented posts 0.270 → **0.374** (+0.104 [+0.091, +0.118]), posts as written −0.0006 | ✅ confirmed · **served** (ADR-031) |
| **S2b′** CE + restoration vs augmented + restoration (5 seeds, v4) | Development: unaccented 0.904 vs 0.880, contrast **0.875 vs 0.800** (p = 0.004). NEU-ESC confirmation: contrast 0.440 vs 0.437 (**p = 0.68**); the other checks favoured CE (overall +0.028, unaccented +0.036) | ✅ **not passed**: the augmented model stays |
| **S3** out-of-scope score | Development AUROC **0.949** (max-probability 0.862). NEU-ESC confirmation: 563 off-topic posts, AUROC **0.977** (max-probability 0.920), 86% flagged at 5% of validation | ✅ confirmed · **served** as `in_scope` (ADR-031) |
| **S5** careful INT8 | 178.5 MB, neutral agreement 0.959; macro-F1 drop +0.0004 but upper bound 0.0095 > 0.005: **not passed**. Latency indicative only (no steady pass): p50 6.4 ms vs 11.1 ms for FP32 | ✅ negative → Cycle 4 |
| NEU-ESC (6,613 real forum posts, 5 seeds per recipe) | macro-F1 CE **0.463**, augmented 0.434, S2a 0.433: all low (the model calls 30% of posts neutral, gold 69%); augmentation lower on real student text (s42: −0.033 [−0.042, −0.024]; 4/5 seeds) | ✅ measured |
| Serving release (ADR-031) | Re-exported with a `features` output (logit parity 1.7e-05, 100% label agreement); restorer and out-of-scope score attached, each SHA-256-checked by the service | ✅ |
| Neutral audit · Kaggle Qwen3-4B · HF upload · U2 label review | — | 👤 |
| Challenge v2 (human-typed) | Replaced as the confirmation set by NEU-ESC (ADR-030); still the only place to test typed teencode and the S2b′ question on this corpus's register | ⏸ optional |

---

## Appendix B — v2 register (closed 2026-09-27)

Everything v2 could close without the owner, with its evidence. Items marked 👤 carried over into v3, and those still open into v4 § 1.

### B1. Outstanding issues (engineering, documentation)

| ID | Issue | What closes it | Who | Status |
|---|---|---|---|---|
| I1 | `mypy` is advisory in CI (35 errors in 13 files), so type errors can merge | Fix them; make `mypy` a blocking CI step | me | ✅ cde2174 |
| I2 | The Docker image has never served the real model. CI smoke-tests it without one | `make docker-e2e`: build, mount the released artifact, `/readyz` 200, golden `/v1/classify` cases | me | ✅ passes; it found `/version` reporting "unversioned", now fixed (f7b456c) |
| I3 | Status lines went stale: STATUS R8 ("XLM-R control running"), R10 ("benchmark pending"), P7 ("revisions not pinned"); REVIEW_COMPLIANCE ("H3 pending", an outdated "what happens next") | One consistency sweep, then the compliance audit re-run at Cycle 2 close | me | ✅ swept; compliance re-audited |
| I4 | Kaggle runs carry no git SHA: the code is uploaded as a zip. They do carry `source_sha256` | Map each recorded source hash to the commit whose `src/` produces it; record it | me | ✅ `results/provenance.json`: Kaggle runs → 5b2858c; 2 cRT runs from an uncommitted tree |
| I5 | The served model is not downloadable (R11: artifacts another researcher can fetch) | A publish command with a dry run, and a model card (data, per-class metrics, limits, manifest hash). The upload stays with the owner | me → 👤 | ✅ `serve publish` (dry run) · 👤 `--upload` |
| I6 | Local leftovers: blocked INT8 staging directories, `kaggle_results/` | Delete them. Keep `.previous-sentiment` (rollback) and the p9 checkpoint (H6) | me | ✅ 270 MB removed after a dry-run merge showed nothing unmerged |
| I7 | No explicit research-question list (review § 11, 🔶) | RQ1–RQ5 in RESEARCH_REPORT, each linked to its evidence | me | ✅ the report had Q1, Q2, Q4–Q6; Q3 (deferred), Q7, Q8 added |

### B2. Experiments not yet run or not yet decided

| ID | Experiment | Status now | What closes it | Who | Status |
|---|---|---|---|---|---|
| H5 | Topic stacking, TF-IDF × PhoBERT (E08) | Declared; 5 OOF folds on the laptop | Declared rule applied; code, features and summary committed | me | ✅ not supported: −0.0102 [−0.0228, +0.0013] |
| H6 | Which model the service runs | Challenge set frozen | Rule applied on the challenge set; if it switches, release through the gate, then ADR plus EVALUATION_PROTOCOL § 4 | me | ✅ switched to the augmented model (ADR-027) |
| H7 | LLM reference on the hard cases (Study E) | Code, frozen prompt, pilot done | `evaluation/llm_reference.py`; pilot Qwen3-1.7B (laptop); declared Qwen3-4B (Kaggle cell); **gpt-4o-mini** (API, owner's key) | me, 👤 (Kaggle, key) | ✅ pilot · 👤 Kaggle cell 4f · 👤 `OPENAI_API_KEY` |
| A1 | Neutral audit (Step 1 of v1) | Sheet exported; decision tree frozen; analysis tool built | `study audit-report` (κ, per-stratum rates, Wilson intervals on the random stratum, applies the frozen tree), with tests. The owner annotates | me → 👤 | ✅ tooling · 👤 annotation |
| E04 | Label correction of train | Gated by A1's outcome | Runs only if A1 finds ≥ 30% incorrect gold | — | ⏸ gated |
| P5 | Diacritic-restoration front-end (alternative to H2) | Untested | Becomes the next robustness question **only if** H6 shows the augmented model failing on typed unaccented text | — | ⏸ not triggered: typed unaccented 0.74 |
| B | Distillation, careful INT8 (track B) | Designed in v1 | A later cycle. The review's rule is one specialization per cycle | — | ⏸ |
| R8′ | Per-model learning-rate tuning for XLM-R and PhoBERT-large; gradient balancing for multi-task | Proposed in RESEARCH_REPORT § 6 | Only with a named question (see "What not to do") | — | ⏸ |

### B3. Results not yet confirmed

| ID | Claim | Why it is not confirmed | What confirms or refutes it | Status |
|---|---|---|---|---|
| C1 | H2 augmentation makes the model robust to missing diacritics | Shown only on *synthetic* noise (`strip_diacritics`, scripted teencode) | H6: `unaccented_typed` and `teencode_typed` rows, constructed (not typed by users) | ✅ confirmed for unaccented (0.32 → 0.74, p < 0.001); teencode not demonstrated (0.975 → 0.875, p = 0.13 at one seed; ADR-028) |
| C2 | TF-IDF beats PhoBERT on `facility` (0.921 vs 0.905) | One model each on dev, no interval | H5 descriptive: paired bootstrap on the same validation rows, plus the 5-fold out-of-fold view | ✅ refuted: −0.021 [−0.060, +0.015] |
| C3 | The model handles negation | 44 templated pairs, simple *không* forms | `negation_pair` rows: *đâu có*, *chẳng … chút nào*, *chưa bao giờ*, negated negatives, both models | ✅ measured: both pair members right in 60% (CE) / 67% (augmented) of 15 pairs, against 36/36 on the simple probe; n is small |
| C4 | Served p95 latency meets the 30 ms target | Met in one session (23.9 ms), missed in the latest (33.6 ms) | Three sessions on AC power in the same Windows power mode; report the median and range. The claim stays the ratio | ✅ met: served p95 19.2 and 20.0 ms in the two reportable sessions (all six passes 18.7–20.7 ms; `latency/sessions/`). The 33.6 ms session had PyTorch 1.6× slower too: machine state |
| C5 | Out-of-scope input behaves sensibly | Never measured (RESEARCH_REPORT § 7) | Confidence on the 20 `out_of_scope` rows vs `objective_neutral`, descriptive | ✅ measured: it does not. Mean confidence 0.86–0.90 on off-topic text |
| C6 | "The neutral gap is not in the classifier" (H1) | Rules out the head, but cannot tell label ambiguity from representation | A1 (human audit) | 👤 |
| C7 | LLM results on UIT-VSFC are fair | Pretraining contamination unknown | The challenge set is the primary LLM comparison; validation is secondary | by design |
