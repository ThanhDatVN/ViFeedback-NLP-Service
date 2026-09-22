# `configs/` — one YAML per experiment

A run is fully described by its config, and the config is copied into `results/runs/<run_id>/` when
the run executes. That is what makes "one axis at a time" checkable by diffing two files rather
than by trusting discipline.

| File | What it pins |
|---|---|
| `baseline/tfidf.yaml` | The B0–B5 ladder |
| `phobert/reference.yaml` | The Gate G2 reference configuration — the anchor every ablation moves one axis away from |
| `phobert/segmented.yaml` | The Gate G3/G4 champion |
| `phobert/tier_a_*.yaml` | Phase 4 imbalance recipes |
| `phobert/large_kaggle.yaml` | Exceeds the 4.29 GB laptop GPU; note the `grad_accum` |
| `serve/default.yaml` | Serving runtime |

Fields map 1:1 onto `TrainConfig`, so a config is a CLI invocation written down.
