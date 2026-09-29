"""채점. 모든 지표는 이 파일로만 계산한다.

- 고객별 평균 Recall@K: 고객마다 (맞힌 구매 수 / 그 고객의 구매 수) → 평균. 표 1에 쓴다.
- 구매 건별 비율: 어떤 셀에 속한 구매 중 추천 K개 안에 든 비율. 그림 1(상품 유형별)에 쓴다.
- 부트스트랩: 테스트 고객을 복원 추출한다. 조건 간 차이는 같은 추출 결과(가중치)를 공유한다(짝지은 부트스트랩).
"""
import numpy as np
import pandas as pd

SEGMENTS = ("all", "non_repurchase", "new_item")


def target_rows(users, targets):
    """정답 표의 각 구매가 후보 행렬의 몇 번째 고객인지. 후보에 없는 고객이 있으면 멈춘다."""
    row_of = pd.Series(np.arange(len(users)), index=users)
    missing = ~targets["user"].isin(row_of.index)
    if missing.any():
        raise ValueError(f"후보가 없는 정답 고객 {targets.loc[missing, 'user'].nunique()}명")
    return row_of.loc[targets["user"].values].values


def hits_at_k(cands, rows, items, k):
    """구매(rows[i], items[i])가 그 고객의 상위 k개 후보 안에 있는가."""
    n = int(max(cands.max(), items.max())) + 1
    cand_keys = (np.arange(len(cands), dtype=np.int64)[:, None] * n + cands[:, :k]).ravel()
    return np.isin(rows.astype(np.int64) * n + items, cand_keys)


def hit_ranks(cands, rows, items, k):
    """상위 k개 안에서의 순위(0부터), 없으면 -1."""
    rank = np.full(len(rows), -1)
    for j in range(k - 1, -1, -1):
        rank[cands[rows, j] == items] = j
    return rank


def recall_user_mean(cands, users, targets, k):
    rows = target_rows(users, targets)
    hit = hits_at_k(cands, rows, targets["item_idx"].values, k)
    per_user = pd.Series(hit).groupby(rows).mean()
    return float(per_user.mean())


def segment_mask(targets, segment):
    if segment == "all":
        return np.ones(len(targets), dtype=bool)
    if segment == "non_repurchase":
        return ~targets["is_repurchase"].values
    if segment == "new_item":
        return targets["is_new_item"].values
    raise ValueError(segment)


def user_tallies(cands, users, targets, k, ndcg_k, cell_ids, n_cells):
    """고객별 맞힌 수·구매 수를 구간별로 집계. 분석과 부트스트랩은 이 집계만 쓴다.

    반환: {segment: {"hits": (U,), "n": (U,), "cell_hits": (U, C), "cell_n": (U, C), "ndcg": (U,)}}
    cell_ids: 정답 표 각 행의 셀 번호 (없으면 -1)
    """
    U = len(users)
    rows = target_rows(users, targets)
    items = targets["item_idx"].values
    hit = hits_at_k(cands, rows, items, k).astype(np.int64)
    rank = hit_ranks(cands, rows, items, ndcg_k)
    gain = np.zeros(len(rank))
    gain[rank >= 0] = 1.0 / np.log2(rank[rank >= 0] + 2.0)
    out = {}
    for seg in SEGMENTS:
        m = segment_mask(targets, seg)
        r, h, g, c = rows[m], hit[m], gain[m], cell_ids[m]
        n_u = np.bincount(r, minlength=U)
        dcg = np.bincount(r, weights=g, minlength=U)
        ideal = np.array([np.sum(1.0 / np.log2(np.arange(min(x, ndcg_k)) + 2.0)) for x in n_u])
        has_cell = c >= 0
        flat = r[has_cell] * n_cells + c[has_cell]
        out[seg] = {
            "hits": np.bincount(r, weights=h, minlength=U).astype(np.int64),
            "n": n_u.astype(np.int64),
            "ndcg": np.divide(dcg, ideal, out=np.zeros(U), where=ideal > 0),
            "cell_hits": np.bincount(flat, weights=h[has_cell], minlength=U * n_cells).reshape(U, n_cells).astype(np.int64),
            "cell_n": np.bincount(flat, minlength=U * n_cells).reshape(U, n_cells).astype(np.int64),
        }
    return out


def code_level_tallies(cands, users, targets, item_codes, k):
    """product_code 수준 채점: 색상만 다른 상품을 같은 상품으로 본다 (all 구간)."""
    rows = target_rows(users, targets)
    t = pd.DataFrame({"row": rows, "code": item_codes[targets["item_idx"].values]}).drop_duplicates()
    code_cands = item_codes[cands[:, :k]]
    n = int(item_codes.max()) + 1
    keys = (np.arange(len(cands), dtype=np.int64)[:, None] * n + code_cands).ravel()
    hit = np.isin(t["row"].values.astype(np.int64) * n + t["code"].values, keys)
    U = len(users)
    return {"hits": np.bincount(t["row"].values, weights=hit, minlength=U), "n": np.bincount(t["row"].values, minlength=U)}


def user_mean(hits, n, w=None):
    """고객별 Recall 평균. w는 부트스트랩 가중치(고객별 추출 횟수)."""
    has = n > 0
    r = np.divide(hits, n, out=np.zeros(len(n)), where=has)
    w = np.ones(len(n)) if w is None else w
    denom = (w * has).sum()
    return float((w * r * has).sum() / denom) if denom > 0 else float("nan")


def bootstrap_weights(n_users, n_boot, seed):
    """고객별 추출 횟수를 한 번에 하나씩 만든다. seed가 같으면 항상 같은 순서라서,
    모든 조건·seed의 통계를 같은 반복 안에서 계산하면 짝지은 부트스트랩이 된다."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n_users, n_users), minlength=n_users)


def ci(samples, level=0.95):
    """백분위 구간. 추출 결과 셀이 비어 계산이 안 된 반복(NaN)은 제외한다."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))
