# Notebooks

**Notebooks narrate; the library computes.** Every analysis function lives in `src/vifeedback/` and
is covered by tests, so a notebook cannot disagree with the pipeline and the laptop and the cloud
run identical code (docs/ROADMAP.md § 4).

| Notebook | What it is | Executed? |
|---|---|---|
| [`01_eda.ipynb`](01_eda.ipynb) | Exploratory analysis. Each section ends in a **decision**, and where an analysis changed the plan the ADR is named | ✅ outputs + 4 figures |
| [`02_results.ipynb`](02_results.ipynb) | Modelling results, read live from `results/registry.csv` — nothing typed by hand | ✅ outputs + 4 figures |
| [`kaggle_train.ipynb`](kaggle_train.ipynb) | Logic-free wrapper for the models that exceed the 4.29 GB laptop GPU. See [KAGGLE_GUIDE](../docs/KAGGLE_GUIDE.md) | ▢ run on Kaggle |
| [`colab_train.ipynb`](colab_train.ipynb) | Same, for Colab. Kaggle is preferred (ADR-014) | ▢ run on Colab |

## Reproducing

```bash
pip install -e ".[dev]"
vifeedback data fetch
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

`01_eda` needs only the corpus. `02_results` reads the registry, so it reflects whatever runs exist
— re-running it after new experiments updates every table and figure.
