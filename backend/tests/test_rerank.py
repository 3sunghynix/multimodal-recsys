"""추천 재정렬 (services/rerank.py)."""
import numpy as np

from app.services.rerank import cap, diversify


def test_category_cap_and_order():
    # 점수 순 위치 0..9, 앞 6개가 모두 같은 소분류 "T"
    cats = np.array(["T"] * 6 + ["P", "S", "P", "S"], object)
    out = diversify(np.arange(10), 6, cats=cats, params={"max_category_share": 0.3, "min_cap": 3})
    assert out.tolist() == [0, 1, 2, 6, 7, 8]          # "T"는 3개까지, 나머지는 다음 점수 순 후보
    assert (cats[out] == "T").sum() == 3


def test_fills_with_skipped_when_caps_too_tight():
    cats = np.array(["T"] * 8 + ["P"], object)
    out = diversify(np.arange(9), 6, cats=cats, params={"max_category_share": 0.3, "min_cap": 3})
    assert len(out) == 6 and out.tolist() == [0, 1, 2, 8, 3, 4]   # 상한으로 못 채우면 건너뛴 후보로 점수 순 채움


def test_empty_category_and_no_brand_cap_by_default():
    order = np.arange(5)
    assert diversify(order, 5, cats=np.array([""] * 5, object)).tolist() == [0, 1, 2, 3, 4]   # 소분류 없음 → 제한 없음
    brands = np.array([7] * 5)
    assert diversify(order, 5, brands=brands).tolist() == [0, 1, 2, 3, 4]                 # 브랜드는 기본 제한 없음
    out = diversify(order, 4, brands=np.array([7, 7, 7, 8, 9]), params={"max_brand_share": 0.25, "min_cap": 1})
    assert out.tolist() == [0, 3, 4, 1]                                                   # 켜면 브랜드도 제한


def test_cap_values():
    assert cap(20, 0.3, 3) == 6 and cap(5, 0.3, 3) == 3 and cap(20, None, 3) is None
