"""TfidfScope against scikit-learn's TfidfVectorizer + LogisticRegression (ADR-034)."""

from __future__ import annotations

import numpy as np
import pytest

from vifeedback.serving.scope_tfidf import TfidfScope

TRAIN = [
    "giảng_viên nhiệt_tình dễ_hiểu",
    "thầy dạy hay nhưng bài_tập nhiều",
    "phòng học nóng máy_chiếu hư",
    "giảng_viên giảng bài chậm",
    "tuyển sinh_viên làm thêm lương cao",
    "giá vàng hôm_nay tăng mạnh",
    "câu_lạc_bộ tổ_chức sự_kiện cuối tuần",
    "bán điện_thoại giá rẻ inbox",
]
IN_SCOPE = np.array([1, 1, 1, 1, 0, 0, 0, 0])
PROBE = [
    "giảng_viên dạy hay",
    "giá vàng tăng",
    "",
    "a",
    "GIẢNG_VIÊN Nhiệt_Tình",
    "bài_tập bài_tập bài_tập nhiều",
    "từ lạ hoàn_toàn không có",
]


def test_equals_scikit_learn():
    sk = pytest.importorskip("sklearn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    assert sk
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000),
    ).fit(TRAIN, IN_SCOPE)
    ours = TfidfScope.from_sklearn(model, threshold=0.0)
    texts = TRAIN + PROBE
    assert np.abs(ours.decision(texts) - model.decision_function(texts)).max() < 1e-12


def test_round_trips_through_npz(tmp_path):
    s = TfidfScope(
        terms=["a b", "cc"], idf=[1.5, 2.0], coef=[0.3, -1.0], intercept=0.1, threshold=-0.2
    )
    s.save(tmp_path / "scope.npz")
    t = TfidfScope.load(tmp_path / "scope.npz")
    texts = ["cc dd", "zz", "cc cc"]
    assert t.decision(texts).tolist() == s.decision(texts).tolist() and t.threshold == -0.2


def test_empty_text_scores_the_intercept():
    s = TfidfScope(terms=["abc"], idf=[1.0], coef=[2.0], intercept=-0.7, threshold=0.0)
    assert s.decision(["", "x"]).tolist() == [-0.7, -0.7]
