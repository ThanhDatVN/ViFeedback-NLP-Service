# Research Notes

Landscape survey conducted 2026-09-22, before planning. Recorded so that the roadmap's choices are
traceable to evidence rather than to habit. Sources are listed in [§ 7](#7-sources).

---

## 1. Dataset landscape

### Primary: UIT-VSFC
Chosen and fixed. Full details in [DATA_CARD.md](DATA_CARD.md). The decisive properties: it carries
**both** sentiment and topic labels on the same sentences (which makes the multi-task experiment free),
it has an official split (comparability), and it is genuinely informal student writing (which makes the
teencode/diacritic error analysis real rather than contrived).

### Alternatives surveyed and their role here

| Dataset | Size | Labels | Role in this project |
|---|---|---|---|
| **UIT-VSFC** | 16,175 | sentiment (3) + topic (4) | **Primary** |
| UIT-ViSFD | 11,122 | aspect-based, smartphone e-commerce | Phase 8 cross-domain evaluation. Out of scope as a training set — ABSA is a different task |
| UIT-VSMEC | ~6,927 | 6 emotions, social media | Not used. Different label space |
| NEU-ESC | — | educational sentiment **+ topic**, multitask | Closest analogue to UIT-VSFC; the natural Phase 8 cross-domain target, since its label scheme is nearest to ours |
| VLSP 2016 / 2018 | varies | SA and ABSA, restaurant/hotel | Not used. Domain mismatch |
| AIVIVN 2019 | 27,068 | binary | Not used. Binary labels would defeat the macro-F1 framing that the project is built on |
| ViHSD | — | hate speech | Not used |

**Decision: single-dataset, official split.** Adding datasets would improve absolute numbers and destroy
comparability with published UIT-VSFC results — and comparability is what makes the reproduction claim
(checklist item 3) meaningful. Cross-domain evaluation is kept as Phase 8, reported as a separate table.

---

## Prior results on UIT-VSFC

**Read this table before reporting any number, and put a version of it in the final report.**

| Source | Model | Sentiment | Topic | Metric reported |
|---|---|---|---|---|
| Nguyen et al. 2018 (original) | Maximum Entropy | ~0.88 F1 | >0.84 F1 | **Weighted** |
| Deep-learning comparison study | Bi-LSTM + Word2Vec | 0.92 F1 | 0.896 F1 | **Weighted** |
| Recent PhoBERT-based work | PhoBERT | ~0.94 F1 / 94.5% acc | — | **Weighted / accuracy** |
| BamiBERT (2026) | BamiBERT | 93.86 acc / **83.41 macro-F1** | — | Both |
| Independent reproduction | PhoBERT | **~0.83 macro-F1** | — | **Macro** |

### The critical reading

Almost every headline "92–94%" figure on this dataset is **weighted F1 or accuracy**. The two independent
sources that report macro-F1 both land near **0.83** — an 11-point gap, and the entire gap is the neutral
class.

Three consequences for this project:

1. **A macro-F1 of ~0.82–0.84 is a competitive result**, not a disappointing one. Anyone benchmarking
   against the 94% figure without checking the metric definition will conclude they have failed when they
   have not.
2. The [ROADMAP § 3](ROADMAP.md#3-success-criteria) targets (S1 ≥ 0.81, minimum 0.76) are calibrated
   against 0.83, not against 0.94.
3. Phase 2's "literature reconciliation" step exists precisely to make this explicit. Reporting both
   columns turns a potential embarrassment into the project's sharpest observation.

---

## 3. Model landscape

| Model | Params | Pretraining | Segmentation required | Notes |
|---|---|---|---|---|
| `vinai/phobert-base` | ~135M | 20 GB Vietnamese Wikipedia + News | **Yes** (RDRSegmenter) | RoBERTa recipe. The project's reference model |
| `vinai/phobert-base-v2` | ~135M | 20 GB + **120 GB OSCAR-2301** | **Yes** | Same size, far more data. Max length 256. Cheap upgrade to test |
| `vinai/phobert-large` | ~370M | as v1 | **Yes** | Colab-feasible but slow; poor fit for a CPU latency budget |
| `uitnlp/visobert` | — | Vietnamese **social media** text, XLM-R architecture | **No** (SentencePiece) | Purpose-built for informal text — strong prior fit for teencode-heavy feedback, and it removes the segmentation dependency entirely |
| CafeBERT | — | Vietnamese, XLM-R-large lineage | — | Larger; conflicts with the latency budget |
| `xlm-roberta-base` | ~270M | 100 languages | No | Multilingual control |
| BamiBERT | — | Vietnamese | — | Recent (2026); reports 83.41 macro-F1 on UIT-VSFC sentiment |

### Why ViSoBERT is in the plan and not an afterthought
It is pretrained on social-media Vietnamese, it uses SentencePiece so no word segmentation is needed, and
it therefore composes naturally with the P0 (raw text) condition. If H2 holds — segmentation is
unnecessary — ViSoBERT is simultaneously the accuracy play *and* the latency play, because it deletes the
JVM from the serving path. That makes it the single highest-leverage item in Phase 4 Tier E.

**Max sequence length note.** PhoBERT-base-v2 supports 256 tokens. UIT-VSFC sentences are short, so the
p99 token length from Phase 0 will almost certainly be far below that. Setting `max_length` from measured
p99 rather than from the model's maximum is a free latency win available before any optimization work
begins.

---

## H2 prior art

**The conflict.** PhoBERT's model card states, unambiguously: *"INPUT TEXT MUST BE ALREADY
WORD-SEGMENTED!"* — the model was pretrained on RDRSegmenter output, so the instruction follows from how
it was built.

**The counter-evidence.** *Is word segmentation necessary for Vietnamese sentiment classification?*
([arXiv:2301.00418](https://arxiv.org/abs/2301.00418)) evaluates on VLSP2016-SA and **VSFC** across
fastText, PhoBERT and other models, and concludes that word segmentation is **not** necessary for this
task, with differences typically under 1 percentage point.

**Why this is a good ablation rather than a settled question.** The published finding is an accuracy
claim; the operational question is a joint accuracy-and-cost question, and nobody has answered that one
for this pipeline on this class of hardware. `py_vncorenlp` spawns a JVM and calls into it per request.
If that call costs 5–50 ms per sentence while the quantized encoder costs ~15 ms, then segmentation is the
majority of p95 latency — and an accuracy difference of under 1 pp is not a tie, it is a clear win for
dropping it.

That reframing is what makes [Phase 3](ROADMAP.md#phase-3--preprocessing-and-word-segmentation-ablation)
worth a week: it converts a checklist item into the project's most quotable result, and it lands in
Week 4 rather than Week 7.

---

## Normalization and teencode tooling

| Tool | Use |
|---|---|
| `py_vncorenlp` (RDRSegmenter) | Canonical PhoBERT segmentation. **Requires Java** — the source of risk R1 |
| `underthesea` | Pure-Python segmentation and normalization; the P2 ablation condition and the Docker fallback |
| `pyvi` | Fastest of the common segmenters; a speed-oriented fallback |
| Public teencode/normalizer resources | Seed material for the project's own dictionary (Vietnamese normalizer and teencode lists; ViSoLex for NSW lookup and lexical normalization) |
| ViSoLex | Vietnamese social-media lexical normalization built on BARTpho/ViSoBERT — reference for the teencode category, and a possible Phase 8 preprocessing upgrade |

**Decision.** Build and version the project's **own** teencode dictionary in
`src/vifeedback/preprocess/`, seeded from public lists and extended from Phase 5's error analysis. Reasons:
it is unit-testable; it is versioned with the code; it has no heavyweight dependency; and it serves double
duty as the P4 ablation condition *and* the generator for the `teencode` perturbation suite. The same
asset used both to help the model and to attack it.

**R1 mitigation, restated.** Segmentation runs **offline** during data preparation, so the JVM never
appears in the training loop or the serving path. If Phase 3 shows segmentation is unnecessary, it is
dropped from serving entirely; if it is necessary, the service uses a pure-Python segmenter and the
resulting accuracy delta gets measured and reported.

---

## 6. Quantization and CPU inference evidence

Evidence gathered before Phase 6 so the ladder is designed around it rather than reacting to it.

| Finding | Source | Implication |
|---|---|---|
| ORT **dynamic** INT8 can be **~3.4× slower** than FP32 on some CPUs | ORT issue reports | H3 is real. Dynamic quantization is not a free win, and the ladder must not depend on it |
| **Static** INT8 gives ~1.8–2.95× over FP32 | ORT / OpenVINO benchmarks | Static is the likely real win; budget calibration time for it |
| Encoder benchmarks: onnx-int8 ≈ **3.2×**, openvino-int8 ≈ **5.3×**, at < 0.5% quality cost | sentence-transformers efficiency docs | OpenVINO via `optimum-intel` belongs in the ladder (L6), not as an afterthought |
| INT8 gains depend heavily on **AVX512-VNNI** | Multiple | **Record the reference CPU's instruction-set flags before benchmarking.** Without VNNI, the whole INT8 story changes, and knowing that in advance turns a confusing result into a predicted one |

**Consequences already reflected in the roadmap:**
- L4 (dynamic) and L5 (static) are **both** pre-registered, so a dynamic-quantization regression is a
  measured hypothesis rather than a failure;
- L6 (OpenVINO) is in the plan from the start;
- L1 (dynamic padding) sits *before* every quantization step, because on short Vietnamese sentences it may
  beat all of them — and it is nearly free;
- [EXPERIMENT_MATRIX § Expected ranges](EXPERIMENT_MATRIX.md#expected-ranges-pre-registered) records
  L4 as "0.5×–2.5×", i.e. explicitly two-sided.

**A regression reported well is still a strong CV item.** "I measured dynamic INT8, found it slower on a
non-VNNI CPU, diagnosed why, and shipped static quantization instead" is a better engineering story than
an unexamined 3× speedup quoted from a blog post.

---

## 7. Sources

**Dataset**
- [UIT-VSFC on Hugging Face](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) — split sizes, label maps
- [kietnv/uit-vsfc](https://github.com/kietnv/uit-vsfc) — official release and citation
- [UIT NLP Group datasets](https://nlp.uit.edu.vn/datasets/) — distribution terms
- [UIT-VSFC (KSE 2018)](https://ieeexplore.ieee.org/document/8573337/) — original paper, IAA figures
- [Deep Learning versus Traditional Classifiers on UIT-VSFC](https://arxiv.org/abs/1911.07223) — Bi-LSTM baselines
- [NEU-ESC](https://arxiv.org/abs/2506.23524) — educational sentiment + topic, multitask
- [NLP-Vietnamese-progress: sentiment analysis](https://github.com/undertheseanlp/NLP-Vietnamese-progress/blob/master/tasks/sentiment_analysis.md) — leaderboard

**Models**
- [PhoBERT (EMNLP Findings 2020)](https://arxiv.org/abs/2003.00744) · [repo](https://github.com/VinAIResearch/PhoBERT)
- [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2) — corpus, max length, segmentation requirement
- [ViSoBERT](https://arxiv.org/abs/2310.11166) · [uitnlp/visobert](https://huggingface.co/uitnlp/visobert)
- [BamiBERT](https://arxiv.org/abs/2607.02259) — recent Vietnamese BERT with macro-F1 on UIT-VSFC

**Preprocessing**
- [Is word segmentation necessary for Vietnamese sentiment classification?](https://arxiv.org/abs/2301.00418) — **the H2 counter-evidence**
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) — RDRSegmenter
- [underthesea](https://github.com/undertheseanlp/underthesea)
- [ViSoLex (COLING 2025 demos)](https://arxiv.org/abs/2501.07020) — Vietnamese lexical normalization

**Inference**
- [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ORT issue #12854](https://github.com/microsoft/onnxruntime/issues/12854) — dynamic INT8 slower than FP32
- [sentence-transformers efficiency guide](https://sbert.net/docs/sentence_transformer/usage/efficiency.html) — backend comparison
- [OpenVINO + ONNX Runtime BERT inference](https://opensource.microsoft.com/blog/2023/01/25/improve-bert-inference-speed-by-combining-the-power-of-optimum-openvino-onnx-runtime-and-azure/)

**Training technique**
- [On the Stability of Fine-tuning BERT](https://openreview.net/pdf?id=nzpLWnVAyah) — the basis for the 5-seed policy
- [How Does Adversarial Fine-Tuning Benefit BERT?](https://arxiv.org/abs/2108.13602) — FGM/adversarial training
- [PEFT for low-resource text classification: LoRA, IA3, ReFT](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1677331/full) — PEFT as a regularizer on small data
