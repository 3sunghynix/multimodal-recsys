"""랭커 후보 경로 확장 실험 (recsys.ranker_eval.expand_candidates · variant_groups, 11-26 — 서빙 미적용).
서빙 DB 구조(backend/app)가 필요해 api 컨테이너에서 돈다 (test_retrain.py와 같은 방법)."""
import numpy as np
import pytest

pytest.importorskip("app.db")
pytest.importorskip("torch")

from recsys.ranker_eval import expand_candidates, variant_groups  # noqa: E402

T = np.datetime64("2020-09-01")


def _ts(*days):
    return np.array([T + np.timedelta64(d, "D") for d in days], "datetime64[ns]")


def test_variant_groups_skip_empty_key():
    g = variant_groups(np.array(["a", "", "b", "a", ""], object))
    assert set(g) == {"a", "b"} and g["a"].tolist() == [0, 3] and g["b"].tolist() == [2]
    assert variant_groups(np.array(["", ""], object)) == {}


def test_expand_adds_bought_then_other_colors_most_recent_first():
    rule = -np.arange(10, dtype=float)                       # 규칙 순위 = 순번
    variants = np.array(["", "", "", "", "", "", "x", "y", "x", "y"], object)
    c = expand_candidates(rule, 3, 6, np.array([6, 7]), _ts(1, 2), variants, variant_groups(variants))
    assert c.tolist() == [0, 1, 2, 6, 7, 9]                   # 최근 구매 7 → 같은 옷 9 → 6 (자리 다 참), 규칙 점수 순 정렬


def test_expand_fills_with_next_rule_ranks_and_keeps_size():
    rule = -np.arange(10, dtype=float)
    variants = np.full(10, "", object)
    assert expand_candidates(rule, 3, 6, np.array([], np.int64), _ts(), variants, {}).tolist() == [0, 1, 2, 3, 4, 5]
    assert expand_candidates(rule, 3, 6, np.array([1]), _ts(0), variants, {}).tolist() == [0, 1, 2, 3, 4, 5]   # 이미 후보


def test_expand_never_adds_excluded_items():
    rule = -np.arange(6, dtype=float)
    rule[5] = -np.inf                                        # 산 상품 제외 (Amazon)
    c = expand_candidates(rule, 2, 4, np.array([5]), _ts(0), np.full(6, "", object), {})
    assert 5 not in c and len(c) == 4
    assert len(expand_candidates(rule, 2, 50, np.array([5]), _ts(0), np.full(6, "", object), {})) == 5
