# Notebooks

**Notebooks narrate; the library computes.** Every analysis function lives in `src/vifeedback/` and
is covered by tests, so a notebook cannot disagree with the pipeline and the laptop and the cloud
run identical code (docs/ROADMAP.md § 4).

| Notebook | What it is | Executed? |
|---|---|---|
| [`01_eda.ipynb`](01_eda.ipynb) | Exploratory analysis. Each section ends in a **decision**, and where an analysis changed the plan the ADR is named | ✅ outputs + 4 figures |
| [`02_results.ipynb`](02_results.ipynb) | Results read from the registry, the gate aggregates and `results/studies/`: test headline, ladder, seed-paired comparisons, Cycle 0 (neutral errors, calibration, abstention by class, robustness) and Cycle 1's declared decisions. Nothing typed by hand | ✅ outputs + 5 figures |
| [`kaggle_train.ipynb`](kaggle_train.ipynb) | Logic-free wrapper for what the laptop cannot run: Cycle 1 H3 (§ 4e, XLM-R raw vs pyvi) and the INT8 releases (§ 4d). See [KAGGLE_GUIDE](../docs/KAGGLE_GUIDE.md) | ▢ run on Kaggle |

> **Colab notebook removed.** It was a near-duplicate of the Kaggle wrapper and carried both bugs
> that a real Kaggle run exposed — the namespace shadowing and the post-install `sys.path` refresh —
> because a fix applied to one wrapper does not reach the other. Kaggle is the documented platform
> (ADR-014); one wrapper means one place to fix a bug. It is recoverable from git history at
> `1f6bb0b~1` if a Colab path is ever needed.

## Reproducing

```bash
pip install -e ".[dev]"
vifeedback data fetch
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

`01_eda` needs only the corpus. `02_results` reads the registry and `results/studies/`, so it reflects
whatever runs exist. Regenerate the studies (`vifeedback study …`) first, then re-execute it.

Per-run artifacts from Phases 2–4 were lost in an accidental deletion. `02_results` therefore reads
the surviving gate aggregates (`results/g4_*.json`, `results/phase*_summary.json`) for those phases,
and says so in its first cell.
