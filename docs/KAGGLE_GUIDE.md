# Running on Kaggle — step by step

For the two or three models that do not fit the laptop's 4.29 GB GPU. Everything else runs locally
and faster (69 s/epoch, no session limits) — see [ADR-009 and ADR-014](DECISIONS.md).

**What you need:** a Kaggle account (phone-verified, which is what unlocks GPU and Internet).
**Time:** ~5 minutes to set up, then 40–90 minutes per model.

---

## Step 1 — Build the upload archive (on the laptop)

From the repository root:

```bash
git archive --format=zip -o vifeedback.zip HEAD
```

`git archive` exports **committed files only**, which is exactly right: the dataset is gitignored and
will be fetched inside the notebook, so nothing large travels. Expect roughly **1.3 MB / 160 files** (it includes the committed study outputs).

Sanity check before uploading:

```bash
python -c "
import zipfile
z = zipfile.ZipFile('vifeedback.zip')
n = z.namelist()
print(len(n), 'files')
assert 'pyproject.toml' in n and any(f.startswith('src/') for f in n)
print('OK')
"
```

> Commit first if you have uncommitted work — `HEAD` will not see it.

---

## Step 2 — Create the Kaggle Dataset

1. <https://www.kaggle.com/datasets> → **New Dataset**
2. Upload `vifeedback.zip`
3. Title: **`vifeedback-repo`** (the notebook globs for any `.zip` under `/kaggle/input`, so the
   exact name is not critical — but keep it recognisable)
4. Visibility: **Private**
5. **Create**

**Why a Dataset rather than re-uploading each session:** it is versioned, it survives session
restarts, and the notebook keeps working even with Internet off. When the code changes, upload a
**New Version** of the same dataset rather than creating a second one.

---

## Step 3 — Create the notebook

1. <https://www.kaggle.com/code> → **New Notebook**
2. **File → Import Notebook** → upload `notebooks/kaggle_train.ipynb`
3. In the **Settings** panel on the right:

| Setting | Value | Why |
|---|---|---|
| **Accelerator** | **GPU P100** | 16 GB. `T4 x2` also works but the code uses one GPU |
| **Internet** | **On** | Required to reach the HF Hub for the model and the dataset |
| **Environment** | Latest / pinned | Either is fine |

4. **Add Data** (top right) → **Your Datasets** → attach `vifeedback-repo`

Confirm it mounted: `/kaggle/input/vifeedback-repo/vifeedback.zip` should exist. Cell 2 of the
notebook asserts this and fails with a pointer back here if it does not.

---

## Step 4 *(optional)* — Hugging Face token

Unauthenticated Hub downloads are rate-limited and occasionally fail mid-download.

1. <https://huggingface.co/settings/tokens> → create a **read** token
2. In the notebook: **Add-ons → Secrets → Add a secret**
3. Label exactly **`HF_TOKEN`**, paste the value, and tick **Attach to notebook**

The notebook picks it up automatically and carries on without it if absent. Skip this on the first
run; add it if you hit a download error.

---

## Step 5 — Run

Cells 1–3 (environment, repository, data) take about 3–5 minutes and always run first. After that,
run **only the cells the current cycle needs**. Each training cell is a multi-seed sweep, and every
run that updates weights counts against the cycle budget in `configs/experiments/ledger.csv`.

### Current cycle — Cycle 1 (`configs/experiments/cycle1.yaml`)

| Cell | What | Runs | Est. (T4) | Needed? |
|---|---|---:|---|---|
| **4e** | **H3: `xlmr-base` on raw text + on pyvi, same session, 5 seeds each, robustness in-run** | 10 | **~65 min** | **yes, the only Kaggle-only hypothesis** |
| 4d | Release FP32 + dynamic INT8 + static INT8 through the verified release step | 1 | ~10 min | optional: INT8 needs `onnx`, blocked on the laptop |

Cell 4e prints the paired result (raw − pyvi, 95% CI, wins, p) at the end, so the answer is visible
before the session closes.

Cell 4d reports each variant as `released` or `BLOCKED`. **A blocked INT8 release is a result, not a
crash**: it means that quantization lost more than the pre-registered 0.005 macro-F1. Every
manifest, released or blocked, is copied to `results/studies/export/` so it comes home in the archive.

### History — Tier E (already in the registry)

| Cell | Model | Est. |
|---|---|---|
| 4a | `phobert-large` (368M, ~7.9 GB) | ~75 min |
| 4b | `xlm-roberta-base` full, pyvi (277M, ~5.9 GB) | ~40 min |
| 4c | `CafeBERT` (560M). Commented out; not registered in `constants.MODEL_IDS` | ~90 min |

Do not re-run these unless you are deliberately reproducing Tier E. Cell 4e already re-runs the one
Tier E condition that Cycle 1 needs.

**Check the effective batch.** The train command prints
`effective batch = 16 x 2 = 32`. `phobert-large` uses batch 16 to fit, so `--grad-accum 2` restores
the effective batch to the 32 that `phobert-base` used. Comparing two models at different effective
batch sizes is not a controlled comparison, and that mismatch was a real defect in the previous
version of this notebook.

---

## Step 6 — Verify before the session ends

Cell 5 asserts that runs actually reached the registry and prints mean ± std per configuration. **Do
not skip it.** A session that finished without writing rows produced nothing, and noticing that now
costs seconds rather than a repeat of the whole sweep.

---

## Step 7 — Bring the results back

**Save Version → Save & Run All (Commit)** persists `/kaggle/working` with the notebook version.
Cell 6 also writes `kaggle_results.zip`, downloadable from the **Output** panel.

> **Never extract the Kaggle output over the repository, and never delete or replace `results/`.**
> The Kaggle `results/` only knows about Kaggle's runs. On 2026-09-27 the output was extracted into
> the repository and the local `results/`, holding 19 uncommitted laptop runs, was lost. The runs had
> to be regenerated (training is deterministic, so they came back identical, at a cost of ~2 GPU-hours).

```bash
# 1. extract kaggle_results.zip into its own folder (gitignored), e.g. kaggle_results/
# 2. check what would be merged, then merge: append-only, never overwrites a run directory
python -m vifeedback.cli results merge kaggle_results --dry-run
python -m vifeedback.cli results merge kaggle_results
python -m vifeedback.cli study tables
```

Then add the session's weight-updating runs to `configs/experiments/ledger.csv`.

**Latency of the INT8 artifacts** must be measured on the reference machine (see below): download
`models/serve_int8_dynamic/` and `models/serve_int8_static/` from the Output panel, place them
under `models/`, and run `vifeedback serve bench --model-dir models/serve_int8_dynamic/sentiment`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No GPU` assertion | Accelerator not set | Settings → Accelerator → GPU P100 |
| `No internet` assertion | Internet off | Settings → Internet → On |
| `No zip found under /kaggle/input` | Dataset not attached | Add Data → attach `vifeedback-repo` |
| `pyproject.toml missing` | Archive built from the wrong directory | Re-run `git archive` from the repo root |
| `CUDA out of memory` | Batch too large for the model | Halve `--batch-size`, double `--grad-accum` — the effective batch stays fixed |
| HF `429` / download fails | Hub rate limit | Add the `HF_TOKEN` secret (Step 4) |
| `datasets` script error | Old `datasets` | Cell 2's `pip install -U 'datasets>=3.0'` handles it |
| Session dies mid-sweep | 9 h limit or idle timeout | Reduce `--seeds all` to `--seeds 42,1337,2024`; runs already written are kept |
| Tests in cell 3 fail | Upstream data changed | **Stop.** That assertion exists to prevent silently training on different data |

---

## Quota and limits

| | Kaggle free |
|---|---|
| GPU quota | **30 hours/week**, stated and visible in your account |
| Session length | up to 9 hours |
| Idle timeout | ~20 minutes with the browser closed (use *Save & Run All* for long sweeps) |
| Output persisted | yes, with the notebook version |

Cycle 1's Kaggle workload, cells 4e and 4d, is about **65 GPU-minutes**, roughly 4% of one
week's quota.

---

## What must never run here

**Any latency benchmark.** [EVALUATION_PROTOCOL § Latency harness](EVALUATION_PROTOCOL.md#latency-harness)
fixes the reference machine as the laptop (AMD Ryzen 5 6600H, AVX2, **no AVX512-VNNI**), recorded in
every `env.json`. A p95 from a Kaggle VM measures different silicon and must not enter the registry.

Accuracy is hardware-independent and transfers fine — which is the whole reason this split works.
