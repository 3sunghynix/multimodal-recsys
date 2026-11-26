"""C5 다양성 재정렬 평가 (recsys.diversity_eval.mmr · list_stats). 서빙 DB 구조(backend/app)가 필요해 api 컨테이너에서 돈다."""
import numpy as np
import pytest

pytest.importorskip("app.db")
pytest.importorskip("torch")

from recsys.diversity_eval import list_stats, mmr  # noqa: E402


def _E():
    # 상품 0 · 1 · 2는 거의 같은 방향, 3 · 4는 다른 방향
    v = np.array([[1, 0, 0], [0.99, 0.1, 0], [0.98, 0.15, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_mmr_lambda_one_is_score_order_and_lower_lambda_diversifies():
    E, pool, rel = _E(), np.arange(5), np.array([5, 4.9, 4.8, 3, 2.0])
    assert mmr(pool, rel, E, 3, 1.0).tolist() == [0, 1, 2]                 # λ = 1: 점수 순
    assert mmr(pool, rel, E, 3, 0.5).tolist() == [0, 3, 4]                 # 비슷한 1 · 2 대신 다른 방향


def test_mmr_respects_variant_and_category_limits():
    E, pool, rel = _E(), np.arange(5), np.array([5, 4.9, 4.8, 3, 2.0])
    variants = np.array(["v", "v", "", "", ""], object)
    assert 1 not in mmr(pool, rel, E, 3, 1.0, variants=variants).tolist()  # 같은 옷은 1개
    cats = np.array(["a", "a", "a", "b", "b"], object)
    assert mmr(pool, rel, E, 3, 1.0, cats=cats, cat_limit=1).tolist() == [0, 3, 1]   # 상한 1 → 모자란 자리는 점수 순


def test_list_stats():
    E = _E()
    s = list_stats(np.array([[0, 3, 4], [0, 1, 2]]), E, np.array(["a", "a", "a", "b", "c"], object))
    assert s["categories@12"] == 2.0 and s["max_category_share@12"] == round((1 / 3 + 1) / 2, 3)
    assert 0 < s["ild@12"] < 1
