# Next Plan — v6: Cycle 5 and the targets still open

**v6, 2026-09-29.** Since v5 (commit 3b8da19, in git history):
- **Cycle 4 is closed** (ADR-032 to ADR-036). The service now reports scope with a topic-aware
  detector. H8 was not passed. INT8 passed S5′ but was held back by the release gate.
- **The Hub release matches the service.** The Hub bundle was updated to ADR-034 on 2026-09-29, and
  `serve reproduce` from the Hub gives 0.8644.
- **The owner chose Cycle 5's order:** (b) real typing, then (a) a distilled student, then (c) other
  institutions.
- **The open targets were researched again** (sources S11–S21, § 10).

**Update, 2026-10-01.** H10 was not passed (ADR-038); H11 passed and is released (ADR-039, ADR-040);
H10b passed (ADR-042) and, by the owner's decision 9, **the service runs it** (ADR-043). Two
proposals wait for the owner: H12′, a substitute for H12 that needs no new labels (§ 4), and A1′,
an LLM-assisted neutral audit validated on a human subset (§ 5).

The manual work that only the owner can do is written up step by step, in Vietnamese, in
[HUONG_DAN_THU_CONG.md](HUONG_DAN_THU_CONG.md). Per-cycle results and open problems are in
[STATUS.md](STATUS.md).

Status key: ✅ done · ⏳ next, no owner input needed · 👤 needs the owner · ⏸ deferred, with the
reason · ✖ dropped, with the reason.

---

## 0. Where things stand

**The service.**
- **Model:** PhoBERT-base trained with H2's augmentation and H10b's anchored consistency (seed 42),
  FP32 ONNX, 540 MB, T = 1.349 (ADR-043).
- **Input handling:** it lowercases input and restores diacritics on unaccented input (ADR-031).
- **Scope:** it reports `in_scope` / `scope_score` from a TF-IDF topic detector (ADR-034).
- **Checks:**
  - the restorer and the detector are refused unless their SHA-256 matches the manifest;
  - `docker-e2e` checks the served model's hash inside the image;
  - CI and the Hub reproduction workflow are green.
- **Where to get it:** <https://huggingface.co/Datk4/vifeedback-sentiment-phobert> still holds the
  12-layer p9 model (CC BY-NC 4.0). The H10b bundle is built and waits for the owner's approval.

**Targets** (ROADMAP, revised in ADR-015). The served H10b model meets every committed minimum,
S4's by the smallest margin (neutral F1 0.550 at seed 42). S7 is met by the released 6-layer
student, not by the served model (ADR-043):

| Target | Minimum | Stretch | Now | Route in v6 |
|---|---|---|---|---|
| S1 sentiment test macro-F1 | ≥ 0.80 ✅ | ≥ 0.84 | Served H10b 0.824 (5 seeds); the p9 recipe 0.830; the student 0.817 | Not chased; measured once at each closing gate (ADR-040, ADR-042) |
| S2 topic test macro-F1 | ≥ 0.79 ✅ | ≥ 0.83 | 0.804 | Not pursued (§ 5) |
| S4 neutral F1 | ≥ 0.55 | ≥ 0.65 | Served H10b 0.550 at seed 42; the p9 recipe 0.576 (5 seeds, test); the student 0.545 | D1 audit (or A1′) → D2 branch (§ 5) |
| S7 serving artifact | — | ≤ 200 MB | ✅ **185 MB**, the released student; the served H10b is 540 MB | H11 passed and released (ADR-039, ADR-040); decision 9 (ADR-043) |
| S8 image | — | ≤ 700 MB | ✅ 519 MB | done |
| S9 coded errors | — | ≥ 30 | 0 | D1 audit (👤) |
| S10 clean-clone evaluation | — | < 15 min | ✅ 2.3 min | done |
| Latency, served p95 | ≤ 60 ms ✅ | ≤ 30 ms | ✅ the 12-layer architecture **22.1 ms** in the Cycle 5 sessions (the student 12.3 ms) | Measured 2026-09-30; the served H10b's own session waits for a free GPU |

**Open problems** (IDs from v4 where they still apply):

| ID | Problem | Evidence | Where handled |
|---|---|---|---|
| G8 | Real typing flips 17% of labels | A lexicon (S2a), augmentation and own-prediction consistency (H10, degenerate: ADR-038) did not help; **H10b cut flips to about 12% with no collapse** (ADR-042) | Reduced; the H10b model is served (ADR-043) |
| — | The student is 0.013 below the 12-layer model on test, while equal on validation | ADR-040: 5/5 seeds lower; 73 neutral validation sentences cannot resolve it | Released, not served (ADR-043); a larger acceptance set for any future size rule |
| — | Other institutions' text: 0.444 in-scope NEU-ESC test | Two heads reach +0.111, but failed two conditions (ADR-033); NEU-ESC test is spent | **H12**, Cycle 5 (c); **H12′** now (§ 4) |
| G6 | NEU-ESC's `neutral` differs from UIT-VSFC's | 67% of NEU-ESC gold-neutral posts are called polar (ADR-032) | Owner decision 2 (§ 8) |
| P2 | Neutral errors: label ambiguity or representation? | Confident errors; boundary and head already ruled out (Cycle 1) | D1 audit (👤) or A1′ → D2 |
| — | The scope detector judges topic from words | Short or unusual course comments can be flagged; off-topic text with course words passes (ADR-034) | Reported in the card; revisited with H12's data |
| G9 | The contrast-sentence drop is shown on challenge v1 only | V1, 4/5 seeds, p = 0.004 | Challenge v2, optional |

---

## 1. The order of work

| Step | Work | Needs | Compute | Gate |
|---|---|---|---|---|
| 1 | ✅ Declare H10 in [`cycle5.yaml`](../configs/experiments/cycle5.yaml) v1 | — | — | committed before any run |
| 2 | ✅ F5: consistency training code, the ViLexNorm dev split, the flip-rate evaluation, tests (2026-09-30) | — | CPU | unit tests; one smoke run |
| 3 | ✅ **H10 selection**: symmetric chosen (one-sided −0.0103 on UIT-VSFC, not eligible) | — | 2 GPU runs | `cycle5.yaml` recipe selection |
| 4 | ✅ **H10 confirmation**: **not passed** (ADR-038): flips 16.6% → 0.3%, but NEU-ESC −0.087; the model labels informal text negative | — | 4 GPU runs | the H10 rule |
| 5 | ✖ Release H10: not passed, nothing released | — | — | — |
| 6 | ✅ Declare H11 (`cycle5.yaml` v2, 2026-09-30): the teacher follows a written condition on H10's decision | — | — | committed before any H11 run and before H10 decided |
| 7 | ✅ F4: FP16-storage export. ✅ **H11 passed** (ADR-039): pretrained-first6, 185.1 MB, pooled bound 0.0078, UIT-VSFC drop 0.00001 on validation | — | 6 GPU runs | the H11 rule |
| 8 | ✅ Release H11 (ADR-040): candidate through the gate, restorer and detector re-accepted, three latency sessions (median p95 12.3 ms), served 2026-09-30 to 2026-10-01, now kept beside the served model. 👤 Hub upload to a separate repository (CC BY-NC-SA 4.0) | 👤 licence, upload | CPU | release gate, `cycle5.yaml` v3 |
| 9 | ✅ H11 closing gate (test, once): student 0.8168 (5 seeds) vs 0.8296, −0.0129 [−0.0204, −0.0054] | — | CPU | logged; reported, not a decision |
| 9b | ✅ H10b (ADR-042): passed; closing gate 0.8237 (−0.0059, n.s.); release candidate built (`cycle5.yaml` v5) | — | 12 GPU runs | the H10b rule |
| 9c | ✅ Decision 9 (ADR-043): **the service runs H10b** (2026-10-01); 40 API tests pass. 👤 Hub upload and licence | 👤 upload | CPU | release gate |
| 9d | Latency session of the served H10b. 2026-10-01: one session, S1 not reportable (no steady pass; model-only p95 34.4 ms against 20.5 ms for the same architecture in Cycle 5, so the machine was loaded); another project's GPU job ran before and after it | a quiet machine | CPU | reported next to the 22.1 ms of the same architecture |
| 10 | H12: declare, then run, when decision 2 and the new labelled sample exist | 👤 decision 2, 👤 data | ≈ 6 GPU runs | declared later |
| 10′ | H12′ (§ 4): ✅ declared in `cycle5.yaml` v6; run when the GPU is free | — | 5 GPU runs, about 3 h | the H12′ rule |
| any time | D1 neutral audit → D2 branch; or A1′ (§ 5), 70 rows by the owner plus a validated LLM | 👤 6–8 h, or 2–2.5 h with A1′ | D2: ≤ 12 runs | the tree frozen in `cycle2.yaml` |
| ✅ | Three latency sessions (owner waived the idle condition) | — | CPU | p95 ≤ 30 ms: 12.3 ms |

One heavy job at a time, with a cool-down and a temperature check between runs (GPU < 70 °C).

---

## 2. Cycle 5 (b): robustness to real typing — H10

**The problem.** A real comment and its human normalization receive different labels 17% of the time
(ViLexNorm test, CE and augmented alike, 5 seeds). Two cheaper fixes failed:
- A teencode lexicon learned from ViLexNorm train (S2a) left flips unchanged: 0.169 → 0.177,
  p = 0.13.
- H2's synthetic augmentation did not help (p = 0.93).

So the instability is not in spelling alone. Real comments differ from their normal form in word
choice, abbreviation, spacing and punctuation together.

**The method.** Consistency training teaches a model to give the same prediction to an input and a
perturbed copy of it:
- **UDA** [S11] adds a KL term between predictions on original and augmented unlabeled inputs, and
  shows the gain depends on the quality of the perturbation. ViLexNorm gives something better than
  any synthetic perturbation: 8,372 pairs of a *real* noisy comment and its human-written clean form.
- **R-Drop** [S12] and **stability training** [S13] use the same loss for robustness.

The pairs carry no sentiment label and need none. The loss asks only that both forms agree.

**Design** (declared in [`cycle5.yaml`](../configs/experiments/cycle5.yaml) v1):

| | |
|---|---|
| Control | The served recipe at 5 seeds (the existing p9/p10 checkpoints, re-evaluated with the same code) |
| Recipes | **one-sided**: CE on UIT-VSFC + 1.0 × KL(p(normalized) ‖ p(original)), no gradient through the clean form. **symmetric**: the bidirectional KL of R-Drop |
| Data | ViLexNorm train split once by a fixed permutation: 7,535 training pairs, 837 dev pairs. ViLexNorm test (1,045) for confirmation, with its two earlier uses disclosed. All text goes through the served transform |
| Selection | Seed 42, lowest dev flip rate among recipes within 0.01 of the control on UIT-VSFC validation |
| Rule | ViLexNorm test flips lower (paired-bootstrap CI below 0, 5 seeds); UIT-VSFC validation macro-F1 −0.005 at most; neutral F1 −0.02 at most; stripped text through the restorer −0.01 at most; NEU-ESC validation −0.01 at most |
| Power | With 1,045 pairs and about 10% discordant pairs, a paired difference of 2–3 points in flip rate is resolvable |
| Budget | 6 runs, about 2 GPU-hours on the laptop |

**Why the guards.** A model can lower its flip rate by predicting one class more often. Conditions
2–5 make that visible on labelled data. Condition 5 matters most: NEU-ESC is real student text.

**Licence.** ViLexNorm is CC BY-NC-SA 4.0. A released H10 model would carry CC BY-NC-SA 4.0 weights
(owner decision 3, § 8).

**Outcome (2026-09-30, ADR-038): not passed.**
- **Selection.** Symmetric KL was chosen. One-sided cost UIT-VSFC 0.0103 at seed 42, just past
  eligibility.
- **Confirmation at five seeds.** Flips fell from 16.6% to 0.3% (−0.164 [−0.180, −0.149]) with
  UIT-VSFC, neutral and stripped text all within their limits. NEU-ESC validation fell 0.087
  against a 0.01 limit.
- **Why.** The invariance is degenerate: the models label 99.4% of ViLexNorm comments *negative*,
  in both forms. On NEU-ESC, predicted negative rose from 58% to 76%. The unlabeled, off-domain
  pairs had one cheap solution, a single class, and nothing in the loss ruled it out. The guard
  declared for this case caught it.
- **What a next attempt needs** (not declared; ViLexNorm test has now served three rules):
  - a flip metric that a constant prediction cannot game (agreement with the control's label on the
    normalized form, next to each form's label distribution);
  - a guard on the predicted class distribution of the unlabeled text;
  - in-domain unlabeled pairs, or UDA's confidence mask;
  - new confirmation data.

**Not in H10, and why.**
- **A normalizer front-end (C2):** a BARTpho normalizer reaches 57.7% error reduction [S6, S7]. A
  sequence-to-sequence model on CPU for every request would, however, break the 30 ms budget several
  times over. It stays deferred unless H10 fails and a distilled normalizer is proposed.
- **ViSoBERT as the backbone** [S14]: pretrained on 1 GB of Vietnamese social-media text, with a
  15,004-token SentencePiece vocabulary that needs no word segmentation. The checkpoint is 390 MB,
  and the weights are for research use only. It is a second axis (backbone), so it is not mixed
  into H10. It is the declared follow-up if H10 fails *and* the audit points to representation (D2).

---

## 3. Cycle 5 (a): a smaller, faster student — H11 (S7)

**The problem.** The FP32 graph is 540 MB. INT8 passed its non-inferiority test (S5′) but was not
releasable (ADR-036), for two reasons:
- It agrees with PyTorch on only 91.4% of labels.
- Dynamic quantization scales activations per batch, so a sentence's label depended on what else
  arrived in the same request.

**Size arithmetic** (from the served model's config: vocabulary 64,001, hidden size 768, FFN 3,072):

| Layers | Vocabulary | Parameters | FP32 file | FP16 storage, FP32 compute |
|---|---|---:|---:|---:|
| 12 (served) | 64,001 | 135.0 M | 540 MB | 270 MB |
| 12 | trimmed to 20,000 | 101.2 M | 405 MB | 202 MB |
| **6** | 64,001 | 92.5 M | 370 MB | **185 MB** ✓ |
| 6 | trimmed to 20,000 | 58.7 M | 235 MB | 117 MB |
| 4 | 64,001 | 78.3 M | 313 MB | 157 MB |

So neither fewer layers nor FP16 storage alone reaches 200 MB; together they do.
- **FP16 storage** keeps weights in half precision in the file, and the graph casts them to FP32
  once, at load. Every multiplication is FP32, and nothing depends on the batch.
- **Vocabulary trimming** [S18] is not needed and is not chosen. It would map any token outside the
  kept set to `<unk>`, a new failure mode on exactly the real text H10 targets.

**The method.**
- **Student.** The student has 6 layers [S16]. Its initialization is the selection axis:
  - alternate layers of the fine-tuned teacher;
  - the first six layers of pretrained PhoBERT (the Patient-KD setup).
- **Teacher.** H10's model if it is released, otherwise the served model. The soft labels come from
  the teacher recipe's **5-seed ensemble**, averaged, at temperature 2 [S15, S17]. The checkpoints
  exist for the served recipe, and H10 creates them for its own.
  - An ensemble teacher is usually more accurate and better calibrated than one seed.
  - Its soft labels also carry the neutral class's ambiguity, which may help S4 (reported).
- **Loss.** 0.5 × KL to the teacher + 0.5 × CE on gold, plus the served recipe's augmentation and,
  if H10 passed, H10's consistency term.
- **Transfer set.** UIT-VSFC train, plus unlabeled text that the teacher labels: in-scope NEU-ESC
  train posts and the ViLexNorm training originals.
  - The student thereby sees other institutions' text and real typing through the teacher's eyes.
  - No gold label beyond UIT-VSFC train is used, and no NEU-ESC validation or test text.
- **Speed.** A 6-layer student should be about 2× faster than the teacher [S10, S16], which would
  also put the latency target out of reach of machine noise.

**Candidate rule** (declared in `cycle5.yaml` v2, before the first H11 run):
1. The artifact is ≤ 200 MB.
2. Pooled UIT-VSFC + NEU-ESC validation (4,888) macro-F1 is non-inferior to the teacher at a margin
   of **0.01** (one-sided 95% paired-bootstrap bound).
   - The margin is wider than S5′'s 0.005 because INT8 was meant to be lossless. A 6-layer student
     is expected to keep about 99% of the teacher [S10], and 1% of 0.864 is 0.009.
3. Per-source guards: UIT-VSFC validation −0.01 at most, neutral F1 −0.03 at most, stripped text
   through the restorer −0.015 at most.
4. The release gate:
   - logit parity between the ONNX graph and the PyTorch student with the same FP16-rounded weights
     (≤ 1e-4) and 100% label agreement;
   - batch-versus-single output difference ≤ 1e-4;
   - FP16 rounding against the FP32 student is reported.
5. The served pipeline's p95 is ≤ 30 ms in three idle sessions.

**Engineering (F4).** Windows Application Control blocks the `onnx` package's DLL on this machine
(ADR-036), so the usual FP16 converters cannot run here. Instead, the export wraps the student so
that its FP16 parameters are cast inside the traced graph (`torch.onnx.export`, no `onnx` import),
or runs on Kaggle. Either way the release gate verifies the result.

**Considered, not chosen for the first step.** ViDeBERTa-xsmall [S21] is smaller per layer
(12 × 384) but has a different tokenizer (SentencePiece, 128k vocabulary, 241 MB checkpoint). It
changes the tokenizer and the architecture at once, so it is kept for a later step if the PhoBERT
student fails.

**Budget.** About 7 runs:
- selection: 2 initializations at seed 42;
- confirmation: 4 more seeds;
- teacher soft labels: 5 inference passes.

Each run is shorter than a teacher run (half the layers).

---

## 4. Cycle 5 (c): other institutions — H12

**Where Cycle 4 left it** (ADR-033).
- The two-heads model lifts in-scope NEU-ESC test from 0.444 to 0.554 (+0.111, 5/5 seeds), and its
  NEU-ESC head reaches 0.76 on validation.
- It failed on stripped text (−0.0107 against −0.01).
- It also failed a scope condition that measured the retired Mahalanobis score. That condition is
  obsolete now that the topic detector serves (ADR-034).

**What H12 needs before it can be declared.**
1. **Owner decision 2** (§ 8): does the service keep UIT-VSFC's label policy for other institutions'
   text (two heads, the UIT-VSFC head served), or add a head that follows the source's policy?
2. **A new labelled sample.** NEU-ESC test is spent: it confirmed H8 and B4′, both logged. No other
   public, labelled Vietnamese student-feedback corpus from a third institution exists (searched
   2026-09-29; DUIT/EduPulse is internal to UIT, the same institution as UIT-VSFC).
   - The plan: ≥ 600 in-scope posts from another institution, labelled by two people, with
     disagreements adjudicated.
   - 600 posts resolve a +0.05 paired macro-F1 difference; Cycle 4's effect was +0.11.
   - The protocol for the owner is in [HUONG_DAN_THU_CONG.md § 7](HUONG_DAN_THU_CONG.md#7-dữ-liệu-có-nhãn-mới-cho-hướng-c).

**Planned design.**
- The two-heads recipe of ADR-033, trained with H10's consistency objective if H10 passed. H10 is
  aimed at exactly the stripped- and informal-text weakness that failed H8.
- B3's backbone question (PhoBERT-base-v2 or ViSoBERT) only after the recipe passes.
- The scope condition is restated for the served detector: its flag rate on the new sample's
  in-scope posts ≤ 10%.

### H12′: a substitute that needs no new labels (declared in `cycle5.yaml` v6, 2026-10-01)

The data for H12 does not exist yet. What can be tested now, honestly scoped, is whether the service
gets better on **NEU-ESC-style forum posts** without losing anything H8 lost. The held-out data comes
from NEU-ESC train, which no compared model has been scored on.

| | |
|---|---|
| Question | Does the two-heads recipe (ADR-033, UIT-VSFC head served) trained with H10b's anchored consistency raise the served head's macro-F1 on in-scope NEU-ESC posts no model trained on, with no loss on UIT-VSFC, neutral, stripped text or real typing? |
| Data | In-scope NEU-ESC train (21,113 posts) split once, stratified by label (`default_rng(44)`): **3,000 held-out posts** for confirmation (neutral 2,041, negative 583, positive 376), 18,113 for training. NEU-ESC validation (3,026 in-scope) for epochs and eligibility. NEU-ESC test stays spent |
| Control | The served H10b recipe, its five existing seeds (they never saw NEU-ESC) |
| Recipe | `two_heads_anchored`: H10b's loss with H8's two heads (the UIT-VSFC head served) |
| Eligibility | Seed 42: UIT-VSFC validation within 0.01 of the control's seed 42, and NEU-ESC validation higher; otherwise H12′ stops |
| Rule, five seeds each | (1) held-out in-scope macro-F1 higher, seed-paired bootstrap CI above 0; guards: (2) UIT-VSFC validation −0.005 at most; (3) neutral F1 −0.02 at most; (4) stripped text through the restorer −0.01 at most (H8's failure); (5) agreement with the teacher on H10b's 1,500 confirmation pairs −0.02 at most, label_tv ≤ 0.10 (keeps H10b's gain) |
| Budget | 5 GPU runs, about 3 GPU-hours on the laptop, one job at a time |

**What it cannot claim.**
- **Same institution.** The held-out posts come from the same forum and annotation as the training
  posts. H12′ measures in-distribution gain on NEU's forum, not transfer to a third institution; P8
  stays open for that.
- **No scope condition.** The served scope detector was trained on NEU-ESC train, including the
  held-out slice. Scope is reported only.
- **Label policy.** The served head keeps UIT-VSFC's policy (decision 2's default) and is scored
  against NEU-ESC's labels, as in H8. Part of the remaining gap is policy (G6), which no model fixes.
- **Unlabeled overlap.** H11's transfer set held NEU-ESC train text without labels. The student is not
  compared in H12′.

The owner approved it on 2026-10-01 (decision 10); `cycle5.yaml` v6 was committed before the split was drawn.

---

## 5. The other open targets

### S4 neutral F1 (0.576 → 0.65) and S9 coded errors (0 → 30)

The audit is the gate, and nothing fully automatic replaces it:
- Cycle 1 ruled out the decision boundary and the classifier head.
- What remains is **label ambiguity** or **representation**, and only a person reading the sentences
  can separate them.

The owner's guide is [HUONG_DAN_THU_CONG.md § 1](HUONG_DAN_THU_CONG.md#1-neutral-audit-gán-nhãn-kiểm-tra-lớp-neutral).
The audit codes ≥ 30 errors by type, which meets S9. `study audit-report` then applies the tree
frozen in `cycle2.yaml`:

| Branch | Trigger | D2 run | What the research says |
|---|---|---|---|
| `incorrect_gold` | ≥ 30% of the audited neutral-error rows have wrong gold | E04: train-label correction, 2 × 2 (labels × recipe), 3 seeds | Confident learning finds label errors, but a confident model disagreeing with gold is not proof [ANNOTATION_GUIDE § 4] |
| `policy_ceiling` | ≥ 40% ambiguous | One soft-label run, then stop optimizing neutral on this benchmark | Training on soft and gold labels together is the robust choice when items have few annotations [S8] |
| `representation` | otherwise | A raw-text or social-domain encoder (ViSoBERT [S14]), or task-adaptive pretraining [S2] | — |

**Other levers, and why they are not in the plan.**
- **Oversampling neutral with generated text.** A 2026 benchmark across imbalanced text-classification
  sets finds retrieval-based oversampling in embedding space (EmbSMOTE) equal or better than every
  LLM generator, with the gap growing with imbalance [S19]. LLM augmentation helped minority classes
  in educational text only with GPT-4o, not with open models [S20].
  - Here, UIT-VSFC text may not be sent to an API, so an LLM cannot be shown the corpus's neutral
    sentences.
  - If the audit points to representation, an embedding-space oversampling run is a cheap D2
    companion. It is not declared before the audit.
- **H11's ensemble soft labels** may raise the student's neutral F1. This is reported, not claimed in
  advance.

**A1′: an LLM-assisted audit, validated on a human subset (proposed 2026-10-01, not declared).**
A fully automatic audit cannot guarantee its own quality: the question is exactly where careful
readers disagree, and an LLM's agreement with gold says nothing about whether gold is right. Studies
of LLM annotation recommend the opposite order: validate the model against human labels on a subset,
and use it for the rest only if it agrees well enough [S22]. The protocol:

| Step | Who | Work |
|---|---|---|
| 1 | The owner | Blind-labels **70 of the 160 rows**: the 40 `random` rows and 30 error rows drawn by a declared seed (about 2–2.5 h, against 6–8 h). Codes ≥ 30 errors by hand, so S9 is met by a person |
| 2 | A local LLM on Kaggle | Labels all 160 rows blind, with the annotation guide as its prompt: `annotator_label`, `neutral_subtype`, then `gold_assessment` with gold shown. Open weights (Qwen3-14B or larger); **UIT-VSFC text is never sent to an API** |
| 3 | The code | Agreement on the 70 rows: Cohen's κ on `annotator_label` ≥ 0.6 with its bootstrap lower bound ≥ 0.4, and ≥ 70% agreement on `gold_assessment` over the 30 error rows |
| 4 | The code | If step 3 holds, the tree runs on the LLM's labels for the 90 rows the owner did not label, and on the owner's for the rest. The branch is accepted only if its proportion clears the threshold by its bootstrap interval; otherwise, or if step 3 fails, the owner audits the remaining 90 rows |

- **The LLM decides no training label.** Its output decides a branch of the frozen tree; E04's label
  correction keeps its own method. "No LLM output as a gold label" (§ 7) stands.
- **Honest expectation.** On NEU-ESC, Qwen3-4B scored 0.475 against 0.604 for gpt-4o-mini (H7), and
  neutral is where LLMs and people disagree most. The agreement gate may well fail. If it does, the
  owner's 70 rows still count, and the audit continues by hand from there.
- Changing the protocol needs `cycle2.yaml`'s audit section amended by a new version and an ADR before
  any LLM output is seen, and the owner's go-ahead (decision 11).

### S1 and S2 (stretch)

The test split is spent for selection: 28 logged test evaluations, and every Cycle 1–4 decision was
made off test. Tuning towards 0.84 on validation would chase noise, because 73 neutral examples
carry a third of the macro average and one example moves it by about 0.004.
- **S1** is re-measured once, at Cycle 5's closing gate, on whichever model Cycle 5 releases. It is
  not a selection criterion.
- **S2** (topic) is not served, and two topic interventions failed (H4, H5), so no topic work is
  planned. It stays as reported.

### Latency (≤ 30 ms, stretch)

Met. In the Cycle 5 sessions (idle condition waived by the owner) the 12-layer served pipeline
measured p95 22.1 ms and the student 12.3 ms. The served H10b model has the 12-layer architecture;
its own session runs when the GPU is free (`study latency` refuses while another process uses it).

---

## 6. Engineering

| ID | Task | Status |
|---|---|---|
| F3 | Model card after every serving change (`serve publish` dry run) | ongoing |
| F4 | FP16-storage export without the `onnx` package (H11) | ✅ `onnx_export.export_fp16_storage`: parametrized casts, constant folding off; on a tiny RoBERTa the file is half the size, logits equal the FP16-rounded model (< 1e-4), no batch dependence |
| F5 | `training/consistency.py` (paired KL, both variants); the ViLexNorm dev split; flip rate through the served pipeline; unit tests | ✅ `study h10-*`; the smoke run trains and scores end to end |
| F6 | U2 sensitivity table: challenge v1 on the owner-reviewed labels next to the frozen ones | ✅ `study challenge-review` (reproduces `study challenge` exactly when every row is kept); waits for the owner's review |
| F7 | ✅ `cli.py` (3,371 lines) split into `vifeedback/cli/` by group and by cycle; same 59 commands and options, checked command by command | done 2026-09-29 |
| F8 | ✅ `make export` rebuilt the retired Mahalanobis configuration; it now attaches the scope detector. `make publish` and `make reproduce` were added | done 2026-09-29 |
| F9 | ✅ Calibrated confidence in the service (ADR-041): `serve add-temperature`, T in the manifest; student T = 1.491, 12-layer T = 1.551; labels unchanged | done 2026-09-30 |
| H10b | ✅ **Passed** (ADR-042): consistency anchored on the frozen 5-seed teacher; flips −0.061, agreement +0.061, label_tv 0.013, all guards held. Closing gate: test 0.824 over 5 seeds (−0.006, n.s.). Release candidate built (`cycle5.yaml` v5) and **served** by decision 9 (ADR-043); card written (`publish.h10b_card`) | done 2026-10-01 |

## 7. Budget and what not to do

| Resource | Cycle 5 estimate |
|---|---|
| Weight-updating runs | H10 6, H11 about 7, H10b 12, H12′ 5, H12 about 6, D2 ≤ 12 |
| Laptop GPU (RTX 3050, 4.29 GB) | H10 about 2 h, H11 about 1.5 h; one job at a time |
| Kaggle | not needed; optional for F4 if the export cannot run locally |
| API | none; A1′ uses open weights on Kaggle, never an API, for UIT-VSFC text |

**What not to do.**
- No selection on any test split. ViLexNorm test and the Cycle 5 closing gate are used once per
  rule, and logged.
- No NEU-ESC off-topic post in any training set; NEU-ESC test is not used again.
- No UIT-VSFC text to an API; no LLM output as a gold label; no LLM in the CPU service.
- No change to what the service's labels mean without owner decision 2.
- One axis per hypothesis: objective (H10), size (H11), data (H12). Backbones come later, and only
  when conditioned.
- The release gate's 99% agreement rule stands (ADR-036).

## 8. Decisions needed from the owner

The step-by-step versions, in Vietnamese: [HUONG_DAN_THU_CONG.md](HUONG_DAN_THU_CONG.md).

| # | Decision or task | Default until decided | Blocks |
|---|---|---|---|
| 1 | Neutral audit, 160 rows, plus a second pass on ≥ 50 | D2 waits | S4, S9 |
| 2 | Label policy for other institutions' text | Keep UIT-VSFC's | H12 |
| 3 | Weights licence CC BY-NC-SA 4.0 (ViLexNorm text in training) for the served H10b (bundle built for the main repository) and the student (repository `Datk4/vifeedback-sentiment-phobert-6l`, bundle built) | Not uploaded; the Hub keeps p9 | uploads |
| 4 | Approve each Hub upload | No upload | every release |
| 5 | A new labelled sample from another institution | H12 not declared | H12 |
| 6 | ✅ Latency sessions: the owner waived the idle-machine condition (2026-09-30) | — | — |
| 9 | ✅ Which model the service runs: **H10b**, approved 2026-10-01 (ADR-043) | — | — |
| 10 | ✅ Go-ahead for H12′ (§ 4), 2026-10-01 | — | — |
| 11 | Go-ahead for A1′ (§ 5): label 70 rows, and let a validated open LLM label the rest | The full audit (decision 1) | A1′ |
| 7 | U2: 15 challenge v1 labels | v1 stays as frozen | F6 |
| 8 | Challenge v2 | Not written | optional |

Settled:
- Cycle 5's order is b → a → c (2026-09-29).
- NEU-ESC may be used for training and published with attribution (2026-09-28).
- The model is published as a non-commercial research artifact (2026-09-28).

## 9. Kaggle

Cycle 5 needs no Kaggle: H10 and H11 train PhoBERT-base-sized models, which fit the laptop. The two
notebooks stay verified for reruns: `kaggle_h7_llm.ipynb` (H7) and `kaggle_train.ipynb` (clones into
`/tmp`, a text-free results zip). F4 may use Kaggle for the export if torch's exporter cannot produce
the FP16-storage graph locally.

## 10. Research sources

| | Source | Used for |
|---|---|---|
| S1 | Mai et al., 2025. *NEU-ESC: A Comprehensive Vietnamese dataset for Educational Sentiment analysis and topic Classification toward multitask learning*. [arXiv:2506.23524](https://arxiv.org/abs/2506.23524). 32,966 forum posts; PhoBERT-base-v2 77.70 mF1 (4 classes); ViSoBERT 75.79 | H12 ceiling, the two-heads design |
| S2 | Gururangan et al., 2020. *Don't Stop Pretraining*. [ACL 2020](https://aclanthology.org/2020.acl-main.740/) | D2 representation branch (TAPT) |
| S3 | Podolskiy et al., 2021. *Revisiting Mahalanobis Distance for Transformer-Based Out-of-Domain Detection*. [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17612) | ADR-031, ADR-032 |
| S4 | Arora et al., 2021. *Types of Out-of-Distribution Texts and How to Detect Them*. [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.835/) | Background vs semantic shift (ADR-032) |
| S5 | *Fine-Tuning Deteriorates General Textual Out-of-Distribution Detection by Distorting Task-Agnostic Features*. [arXiv:2301.12715](https://arxiv.org/abs/2301.12715) | ADR-032, ADR-034 |
| S6 | Nguyen et al., 2024. *ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text*. [EACL 2024](https://aclanthology.org/2024.eacl-long.85/). CC BY-NC-SA; BARTpho normalizer ERR 57.74% | H10 data; C2 |
| S7 | *ViSoLex: An Open-Source Repository for Vietnamese Social Media Lexical Normalization*. [COLING 2025 demos](https://aclanthology.org/2025.coling-demos.18/) | C2 |
| S8 | Uma et al., 2021. *Learning from Disagreement: A Survey*. JAIR 72 | D2 soft labels |
| S9 | Dataset-specific heads for inconsistent label spaces, e.g. [Plain-Det, 2024](https://arxiv.org/abs/2407.10083); [arXiv:2208.10598](https://arxiv.org/abs/2208.10598) | H12 (two heads) |
| S10 | Wang et al., 2020. *MiniLM*. [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). 6-layer students keep > 99% of the teacher at about 2× speed | H11 margin and speed |
| S11 | Xie et al., 2020. *Unsupervised Data Augmentation for Consistency Training*. [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html). KL consistency between original and augmented unlabeled inputs; gains depend on the quality of the perturbation | H10 |
| S12 | Liang et al., 2021. *R-Drop: Regularized Dropout for Neural Networks*. [NeurIPS 2021](https://arxiv.org/abs/2106.14448). Bidirectional KL between two predictions of the same input | H10 symmetric recipe |
| S13 | Zheng et al., 2016. *Improving the Robustness of Deep Neural Networks via Stability Training*. [CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zheng_Improving_the_Robustness_CVPR_2016_paper.pdf) | H10 one-sided recipe (clean prediction as the target) |
| S14 | Nguyen et al., 2023. *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing*. [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.315/). XLM-R architecture, 12 × 768, 15,004-token SentencePiece vocabulary (config checked 2026-09-29), 390 MB checkpoint, research use only | Deferred backbone (H10 follow-up, D2, H12) |
| S15 | Hinton et al., 2015. *Distilling the Knowledge in a Neural Network*. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531). Soft targets at temperature; ensembles distilled into one model; transfer sets | H11 |
| S16 | Sun et al., 2019. *Patient Knowledge Distillation for BERT Model Compression*. [EMNLP 2019](https://arxiv.org/abs/1908.09355). 3- and 6-layer students initialized from BERT's first layers | H11 student initialization |
| S17 | Wu et al., 2021. *One Teacher is Enough? Pre-trained Language Model Distillation from Multiple Teachers*. [Findings of ACL 2021](https://arxiv.org/abs/2106.01023) | H11 ensemble teacher |
| S18 | Ushio et al., 2023. *Efficient Multilingual Language Model Compression through Vocabulary Trimming*. [Findings of EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.981/). About half the vocabulary suffices for one language | Considered for H11, not chosen (FP16 storage suffices; trimming adds `<unk>` on new text) |
| S19 | Inoshita, 2026. *Class-Structure Preservation Beats Diversity: A Comprehensive Benchmark of Text Augmentation Methods for Imbalanced Text Classification*. [arXiv:2608.12340](https://arxiv.org/abs/2608.12340). EmbSMOTE ≥ every LLM generator; gap up to 0.063 macro-F1 at high imbalance | S4: why not LLM oversampling |
| S20 | Neshaei et al., 2025. *Bridging the Data Gap: Using LLMs to Augment Datasets for Text Classification*. [EDM 2025](https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.long-papers.54/index.html). Minority-class gains with GPT-4o; open models (Llama 3.1 8B) subpar | S4: why not LLM oversampling here |
| S21 | Tran et al., 2023. *ViDeBERTa: A powerful pre-trained language model for Vietnamese*. [Findings of EACL 2023](https://aclanthology.org/2023.findings-eacl.79/). xsmall: 12 × 384, 128,000-token vocabulary, 241 MB checkpoint (checked 2026-09-29) | H11 alternative student, deferred |
| S22 | Pangakis, Wolken and Fasching, 2023. *Automated Annotation with Generative AI Requires Validation*. [arXiv:2306.00176](https://arxiv.org/abs/2306.00176). LLM annotation quality varies by task and dataset; validate against human labels on a subset before using it | A1′ |

---

## Appendix — Cycle 4 outcome (the v5 work list, closed 2026-09-29)

The full v5 text, with its Cycle 3 and v2 appendices, is in git history (`git show 3b8da19:docs/NEXT_PLAN.md`).

| Item | Result |
|---|---|
| A1 served latency | ✅ p95 26.9 ms (median of 3 sessions), additions +2.2 ms |
| A2 `in_scope` on other institutions | ✅ triggered: the Mahalanobis score tracked the institution (within NEU-ESC AUROC 0.573; ADR-032) |
| A3 label-policy description | ✅ 67% of NEU-ESC gold-neutral posts called polar |
| A4 publish | ✅ 2026-09-28; updated to ADR-034 on 2026-09-29 (Hub commit `ba58267`, reproduced 0.8644) |
| A5 Kaggle notebook clones into `/tmp` | ✅ |
| B1 `cycle4.yaml` | ✅ v1–v4 |
| B2 H8 in-domain data | ✅ not passed (ADR-033): two heads +0.111 on NEU-ESC test; stripped −0.0107, the refitted score failed |
| B3 backbone | ✖ not triggered |
| B4 refit the Mahalanobis score | ✖ superseded by B4′ |
| B4′ topic-aware scope detector | ✅ passed and served (ADR-034): AUROC 0.922 within NEU-ESC test, 7.3% of in-scope posts flagged |
| B5 release | ✖ no H8 release |
| E1 INT8 power check | ✅ 92% power on the pooled set (calibrated); S5′ declared (ADR-035) |
| S5′ careful INT8 | ✅ passed; ✖ release gate: 91.4% agreement, batch-dependent (ADR-036) |
| F1 image ≤ 700 MB | ✅ 519 MB |
| F2 clean-clone reproduction | ✅ 2.3 min, in CI (`reproduce.yml`) |
| H7 LLM reference | ✅ decided: `llm_better_on_neutral` on constructed text; Qwen3-4B 0.475 on NEU-ESC, no router |
