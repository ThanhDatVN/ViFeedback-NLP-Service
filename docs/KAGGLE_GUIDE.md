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
will be fetched inside the notebook, so nothing large travels. Expect roughly **300 KB / 75 files**.

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

**Run All** works, but cells 4a/4b/4c are each a full 5-seed sweep. Run only the ones you need:

| Cell | Model | Why it is here | Est. |
|---|---|---|---|
| 4a | `phobert-large` | 368M params, ~7.9 GB — does not fit the laptop | **~75 min** |
| 4b | `xlm-roberta-base` full | 277M, ~5.9 GB. *The frozen-embedding version fits the laptop — run that one there* | ~40 min |
| 4c | `CafeBERT` | 560M, ~11 GB. Commented out; needs registering in `constants.MODEL_IDS` first | ~90 min |

Cells 1–3 (environment, repo, data) take about 3 minutes and must run first.

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

On the laptop, merge **by `run_id`** — never append blind, because the archive's `registry.csv` also
holds the rows that were already committed when you built the dataset:

```bash
unzip -o kaggle_results.zip -d /tmp/kag
cp -rn /tmp/kag/runs/* results/runs/
python - <<'EOF'
import pandas as pd
a = pd.read_csv('results/registry.csv')
b = pd.read_csv('/tmp/kag/registry.csv')
out = pd.concat([a, b]).drop_duplicates(subset='run_id', keep='first')
out.to_csv('results/registry.csv', index=False)
print(f'{len(out) - len(a)} new rows merged')
EOF
```

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

The whole remaining external workload — `phobert-large` and `xlm-roberta-base` at 5 seeds each — is
about **2 GPU-hours**, so roughly 7% of one week's quota.

---

## What must never run here

**Any latency benchmark.** [EVALUATION_PROTOCOL § Latency harness](EVALUATION_PROTOCOL.md#latency-harness)
fixes the reference machine as the laptop (AMD Ryzen 5 6600H, AVX2, **no AVX512-VNNI**), recorded in
every `env.json`. A p95 from a Kaggle VM measures different silicon and must not enter the registry.

Accuracy is hardware-independent and transfers fine — which is the whole reason this split works.
