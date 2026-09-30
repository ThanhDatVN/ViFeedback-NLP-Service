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

## Layout on the owner's machine (2026-09-30)

```text
models/
├── serve/sentiment/        # what the API serves (ADR-040): the 6-layer student, model.fp16.onnx
│                           #   (185 MB, FP16 weights computed in FP32), tokenizer files, restorer.json
│                           #   (ADR-031), scope.npz (ADR-034), manifest.json with every SHA-256
├── serve/.previous-sentiment/  # the 12-layer release (model.opt.onnx, 540 MB); swap back to serve it
├── candidate/sentiment/    # the 12-layer H10b release candidate (ADR-042), waiting for decision 9
├── publish/<name>/         # `serve publish` dry-run bundles: the card, SHA256SUMS, the PyTorch copy
├── hub/<owner>__<name>/    # `serve reproduce` download cache
├── distill/                # H11: the teachers' soft-label cache (SHA-1 keys, no text) and the
│                           #   seed-42 FP16-storage graph h11-confirm checked
├── diacritics/restorer.json, scope/b4prime_tfidf_logistic.pkl   # sources of the two release parts
├── int8_candidates/        # fp32_plain + pc-head-last2: the graphs S5′ tested (ADR-035/036)
├── p9-…-s42-…-ckp/, p10-…-aug-diac-teen-s*-ckp/   # the 12-layer served recipe, 5 seeds: H11's teacher
├── p6-…-base-s42-ckp/, p10-…-base-s*-ckp/         # the CE baseline's 5 seeds (study external, V1)
├── p14-…-h11-pretrained-first6-…-s*-ckp/          # the H11 students, 5 seeds (seed 42 is served)
├── p14-…-h11-teacher-alternate-…-s42-ckp/         # H11's other selection candidate
├── p15-…-h10b-anchored_orig-s*-ckp/                # H10b, 5 seeds (passed, ADR-042); seed 42 is the candidate
├── p15-…-h10b-anchored_both-s42-ckp/              # H10b's other selection candidate
└── p13-…-h10-*-ckp/        # H10 (not passed, ADR-038): no planned step needs them
```

Experiments that ended without a release keep their metrics, configs and validation predictions in
`results/runs/`; their weights can be regenerated, and the owner deletes them when space is needed.
