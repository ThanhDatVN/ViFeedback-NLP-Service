"""S3: out-of-scope scores are higher for in-domain inputs, and the threshold keeps 95% of them."""

from __future__ import annotations

import numpy as np

from vifeedback.evaluation import ood as O


def test_scores_separate_a_far_cluster_and_threshold_keeps_95_percent():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 200)
    centers = np.array([[5.0, 0, 0, 0], [0, 5.0, 0, 0], [0, 0, 5.0, 0]])
    f_train = centers[y] + rng.normal(size=(600, 4))
    maha = O.fit_mahalanobis(f_train, y)
    f_in = centers[y] + rng.normal(size=(600, 4))
    f_out = np.array([0, 0, 0, 12.0]) + rng.normal(size=(100, 4))
    logits_in = np.eye(3)[y] * 6 + rng.normal(
        scale=0.5, size=(600, 3)
    )  # continuous, as real scores are
    logits_out = np.full((100, 3), 1.0)
    s_in, s_out = O.scores(logits_in, f_in, maha), O.scores(logits_out, f_out, maha)
    for method in ("max_probability", "neg_energy", "neg_mahalanobis"):
        r = O.evaluate(s_in[method], s_out[method])
        assert r["auroc"] > 0.95, method
        assert abs(r["in_domain_flagged"] - 0.05) < 0.02
