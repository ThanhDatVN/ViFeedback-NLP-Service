# Decision Log

One entry per decision that a future reader — or you in six weeks — would otherwise have to reverse-engineer
from the code. Append-only. Never edit a past entry; supersede it with a new one.

**Format:** `ADR-NNN · date · title · Status: Accepted | Superseded by ADR-MMM`
with **Context → Decision → Consequences**.

Per [ROADMAP § 5](ROADMAP.md#5-phase-plan), any gate that fails, any scope cut, and any addition to the
[§ 2 exclusion list](ROADMAP.md#2-scope-boundaries) requires an entry here before work continues.

---

## ADR-001 · 2026-09-22 · Macro-F1 is the headline metric · Accepted

**Context.** UIT-VSFC sentiment is ≈ 4.3% neutral. A classifier that never predicts neutral still reaches
≈ 0.93 weighted F1 and ≈ 0.93 accuracy. Most published results on this dataset report weighted F1 or
accuracy, which makes the minority class invisible.

**Decision.** Macro-F1 is the single headline metric for every claim, with weighted F1 and accuracy always
printed beside it.

**Consequences.** Absolute numbers will look lower than the "92–94%" commonly quoted for this dataset, and
Phase 2 must explicitly reconcile the metric definitions or the project will read as a failed reproduction.
Checkpoint selection, threshold tuning and the champion choice all optimize macro-F1, which makes Tier A of
Phase 4 (imbalance handling) the highest-value work in the project.

---

## ADR-002 · 2026-09-22 · Keep the official split unchanged · Accepted

**Context.** The official 11,426 / 1,583 / 3,166 split is a random ~70/10/20 partition. Cross-validation or
a temporal split would give a better-grounded estimate of generalization.

**Decision.** Use the official split verbatim. No re-splitting, no folding dev into train.

**Consequences.** Comparability with published work is preserved, which is what makes the reproduction
claim meaningful. The cost is that reported numbers are in-distribution and probably optimistic for a real
deployment; this is recorded as limitation L8 in the [data card](DATA_CARD.md#8-known-limitations)
and must appear in the model card.

---

## ADR-003 · 2026-09-22 · Single dataset; cross-domain deferred to Phase 8 · Accepted

**Context.** Several Vietnamese sentiment corpora exist (UIT-ViSFD, UIT-VSMEC, NEU-ESC, VLSP, AIVIVN).
Training on their union would likely raise absolute scores.

**Decision.** Train and evaluate on UIT-VSFC only. Other corpora are used solely for zero-shot cross-domain
evaluation in Phase 8, reported in a separate table.

**Consequences.** Lower absolute ceiling, preserved comparability. Any future change requires a new ADR.

---

## ADR-004 · 2026-09-22 · Word segmentation is treated as an open question, not a given · Accepted

**Context.** PhoBERT's model card mandates word-segmented input. Published work
([arXiv:2301.00418](https://arxiv.org/abs/2301.00418)) finds segmentation unnecessary for Vietnamese
sentiment classification. `py_vncorenlp` requires a JVM, which threatens both the Docker image size target
(S8) and the latency target (S5).

**Decision.** Treat it as hypothesis H2 and measure it on **both** axes — accuracy and latency — in
Phase 3. Segmentation runs offline during data preparation regardless, so no JVM appears in the training
loop or the serving path.

**Consequences.** Phase 3 becomes a joint accuracy-and-cost experiment rather than a checklist item, and it
produces the serving-pipeline decision. If H2 holds, the JVM leaves the deployment entirely and the
project's largest latency win arrives in Week 4.

---

## ADR-005 · 2026-09-22 · Both dynamic and static INT8 are pre-registered · Accepted

**Context.** ONNX Runtime dynamic INT8 quantization is widely reported as a speedup, but on CPUs without
AVX512-VNNI it can be *slower* than FP32. The reference machine's instruction-set support is unknown at
planning time.

**Decision.** Pre-register dynamic INT8 (L4), static INT8 (L5) and OpenVINO INT8 (L6) as separate ladder
steps in Phase 6, and record the reference CPU's instruction-set flags before benchmarking.

**Consequences.** A dynamic-quantization regression becomes a confirmed hypothesis (H3) rather than a dead
end, and the project still has two viable paths to the S5 latency target.

---

## ADR-006 · 2026-09-22 · Drop the lowercasing ablation condition · Accepted

**Context.** The Phase 3 preprocessing ablation included condition P5 (`norm` + lowercasing). Gate G0
measurement shows UIT-VSFC contains **0 sentences with any uppercase character** (0 of 16,175).

**Decision.** Remove P5 from the ablation matrix. Record the measurement in the data card instead.

**Consequences.** Five GPU runs saved, and one fewer flat line in the results table. The same
measurement also confirms the corpus is *not* word-segmented (1 sentence contains an underscore), so
the P0-vs-P1 segmentation contrast — the part of Phase 3 that matters — is measuring a real
difference. Generalizes to a rule: measure whether a preprocessing step is a no-op before spending
GPU hours proving it.

---

## ADR-007 · 2026-09-22 · Teencode and diacritics are robustness targets, not error-analysis targets · Accepted

**Context.** The project scope named negation, teencode and missing diacritics as the error-analysis
themes. Gate G0 measurement, over the full 16,175-sentence corpus:

| Phenomenon | Sentences | Share |
|---|---|---|
| Negation markers | 3,300 | 20.40% |
| Missing diacritics | 23 | 0.14% |
| Teencode probes | 26 | 0.16% (19 of which are `ok`) |

Negation is abundant. Teencode and undiacritized text are effectively **absent** — UIT-VSFC was
distributed pre-lowercased, pre-tokenized and orthographically clean. An error analysis of teencode
on this test set would be an analysis of roughly 26 sentences.

**Decision.** Split the theme in two rather than dropping it.

* **Negation** stays an error-analysis target, retargeted from *presence* to *scope*: a bag-of-words
  model already captures `không` → negative as a unigram (84.3% of negation-bearing sentences are
  negative), so the discriminating cases are scope inversions such as `không có gì để chê`.
* **Teencode and missing diacritics** become **robustness** targets, measured by the controlled
  perturbation suites as a macro-F1 drop under an induced distribution shift.

Also corrected: the `SUGGESTION` taxonomy entry claimed suggestions are usually `neutral`. Measured,
sentences containing `nên`/`cần`/`mong` are **91.1% negative** — the guideline treats a request for
change as implicit criticism. The taxonomy entry has been rewritten.

**Consequences.** The claim changes from "the model struggles with teencode" (unsupportable on this
benchmark) to "macro-F1 falls by X pp under de-diacritization, a shift real deployment traffic
exhibits and this benchmark does not" — narrower, measurable, and honest. The perturbation suites
gain importance rather than losing it, since they are now the *only* evidence on informal
orthography. Reported in its own section, never merged into the clean-test table.

---

<!-- Append new entries above this line. -->
