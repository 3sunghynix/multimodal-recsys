"""추천 재정렬 (Re-ranking) — 다양성.

점수 순 후보에서 위부터 고르되, 한 소분류(또는 한 브랜드)가 상한을 넘으면 건너뛰고 다음 후보를 본다.
상한 때문에 k개를 못 채우면 건너뛴 후보로 점수 순 그대로 채운다 → 결과는 항상 min(k, 후보 수)개.

기본은 **소분류만** 제한한다 (예: 20개 중 같은 품목 최대 6개). 브랜드는 기본 제한 없음 —
프로젝트가 브랜드 선호(Amazon β 브랜드 보정)를 의도적으로 반영하므로 재정렬이 그걸 되돌리지 않게 한다.
"""
import math

import numpy as np

DEFAULT = {"max_category_share": 0.3, "max_brand_share": None, "min_cap": 3, "pool": 5}


def cap(k: int, share: float | None, min_cap: int) -> int | None:
    """k개 중 한 그룹이 차지할 수 있는 최대 개수. share가 None이면 제한 없음."""
    if share is None:
        return None
    return max(int(min_cap), math.ceil(k * float(share)))


def diversify(order: np.ndarray, k: int, cats: np.ndarray | None = None, brands: np.ndarray | None = None,
              params: dict | None = None) -> np.ndarray:
    """order: 점수 내림차순 후보 위치. cats · brands: 같은 위치 체계의 소분류 · 브랜드 배열 (""/−1은 제한 대상 아님).
    반환: 고른 위치 min(k, len(order))개, 보여 줄 순서대로."""
    p = {**DEFAULT, **(params or {})}
    k = min(k, len(order))
    cat_cap = cap(k, p["max_category_share"], p["min_cap"]) if cats is not None else None
    brand_cap = cap(k, p["max_brand_share"], p["min_cap"]) if brands is not None else None
    if cat_cap is None and brand_cap is None:
        return order[:k]
    picked, skipped = [], []
    n_cat, n_brand = {}, {}
    for pos in order:
        c = cats[pos] if cat_cap is not None else ""
        b = int(brands[pos]) if brand_cap is not None else -1
        if (c and n_cat.get(c, 0) >= cat_cap) or (b >= 0 and n_brand.get(b, 0) >= brand_cap):
            skipped.append(pos)
            continue
        picked.append(pos)
        if c:
            n_cat[c] = n_cat.get(c, 0) + 1
        if b >= 0:
            n_brand[b] = n_brand.get(b, 0) + 1
        if len(picked) == k:
            break
    if len(picked) < k:                                       # 상한 때문에 모자라면 건너뛴 후보로 점수 순 채움
        picked += skipped[:k - len(picked)]
    return np.asarray(picked, dtype=order.dtype)
