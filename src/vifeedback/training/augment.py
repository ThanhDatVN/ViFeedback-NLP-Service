"""Training-time input augmentation — experiment E05 of docs/REVIEW_AND_RESEARCH_PLAN.md.

Augmentation replaces a fraction `p` of training sentences with a perturbed version of themselves.
It *replaces* rather than adds, so every epoch sees the same number of examples and an augmented run
costs the same GPU time as its control. Labels never change.

Perturbations use the exact transformations of `evaluation/robustness.py`. That is deliberate and it
has a consequence the results must state: robustness measured with the *same* family of
transformations is in-distribution for the augmented model. `charnoise` is never used for
augmentation, so it stays an out-of-family check.

Like the evaluation suites, perturbation happens on **raw** text; the caller re-segments the changed
sentences, because the deployed pipeline segments whatever the user typed.
"""

from __future__ import annotations

import numpy as np

from vifeedback.evaluation.robustness import SUITES

# recipe name -> suites drawn from with equal probability for each selected sentence
RECIPES: dict[str, tuple[str, ...]] = {
    "diac-teen": ("nodiacritic-50", "teencode-100"),
}


def augment(raw_texts: list[str], recipe: str, p: float, seed: int) -> tuple[list[str], np.ndarray]:
    """Return the augmented raw texts and a mask of the sentences that actually changed.

    Selection and the transformation are both keyed on `seed`, so a run is reproducible and two
    seeds see different augmented subsets.
    """
    if recipe not in RECIPES:
        raise ValueError(
            f"unknown augmentation recipe {recipe!r}; expected one of {sorted(RECIPES)}"
        )
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")

    suites = RECIPES[recipe]
    rng = np.random.default_rng([seed, 7919])
    chosen = rng.random(len(raw_texts)) < p
    which = rng.integers(len(suites), size=len(raw_texts))

    out = list(raw_texts)
    for i in np.flatnonzero(chosen):
        fn = SUITES[suites[which[i]]]
        out[i] = fn(raw_texts[i], np.random.default_rng([seed, int(i), 104729]))
    changed = np.array([a != b for a, b in zip(raw_texts, out, strict=True)])
    return out, changed
