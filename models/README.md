# `models/` — local weights, never committed

Everything in this folder except this file is gitignored (`/models/*` in [.gitignore](../.gitignore)),
so a fresh clone holds only this README.

## Why weights are not in git

| | |
|---|---|
| Size | A PhoBERT-base checkpoint is about 540 MB, and a 5-seed sweep is 2.7 GB. Git would store every version forever |
| Reproducibility | A checkpoint without its `config.yaml`, `env.json` and seed is not reproducible, and those *are* committed under `results/runs/` |
| Provenance | `results/registry.csv` records what produced every number. The weights are a cache of that record, not the record itself |
| Distribution | The served model is published on the Hugging Face Hub: <https://huggingface.co/Datk4/vifeedback-sentiment-phobert> |

> This folder's *name* once broke the repository. `.gitignore` held the bare pattern `models/`,
> which git matches at any depth, so it silently excluded the `src/vifeedback/models/` **package**
> from every commit. The pattern is anchored now (`/models/*`), and
> `tests/unit/test_packaging.py` fails the build if any source file is ignored again.

## Getting the artifacts

```bash
# The published release, verified against SHA256SUMS, with its validation macro-F1 reproduced
vifeedback serve reproduce                       # -> models/hub/<owner>__<name>/

# Or train and release locally
vifeedback train run --task sentiment --model phobert-base --preprocessing seg_pyvi \
    --augment diac-teen --augment-p 0.3 --seeds 42 --save-checkpoint
make export        # ONNX through the release gate, then the restorer and the scope detector
```

## Layout on the owner's machine (2026-09-29)

```text
models/
├── serve/sentiment/        # what the API serves: model.opt.onnx (FP32, logits + features), tokenizer
│                           #   files, restorer.json (ADR-031), scope.npz (ADR-034), manifest.json
│                           #   with the SHA-256 of every file and the release-gate acceptance
├── publish/<name>/         # `serve publish` dry-run bundle: the above + model card + SHA256SUMS + PyTorch
├── hub/<owner>__<name>/    # `serve reproduce` download cache
├── diacritics/restorer.json   # the restorer before it is attached to a release
├── scope/b4prime_tfidf_logistic.pkl   # the fitted scope detector; `serve add-scope` converts it
├── int8_candidates/        # fp32_plain + pc-head-last2: the graphs S5′ tested (ADR-035/036).
│                           #   The onnx DLL is blocked on this machine, so they cannot be rebuilt here
├── p9-…-s42-…-ckp/         # the served checkpoint (H2-augmented, seed 42)
├── p10-…-aug-diac-teen-s{7,1337,2024,31337}-…-ckp/   # the served recipe's other seeds: controls
│                           #   for Cycle 5 H10 and H11's ensemble teacher
└── p6-…-base-s42-ckp/, p10-…-base-s*-ckp/            # the CE baseline's 5 seeds (study external, V1)
```

Checkpoints of experiments that ended without a release (p11: S2a; p12: H8) are not needed by any
planned step. Their metrics, configs and validation predictions are committed under
`results/runs/`, and the runs can be regenerated from them.
