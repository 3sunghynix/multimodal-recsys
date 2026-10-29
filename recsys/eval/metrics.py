"""오프라인 지표. 입력 rec는 (평가 고객 수 × K) 상품 id 행렬(점수 내림차순, 행 안에 중복 없음), 행 순서 = users.

정확도 (고객별 값 → 평균, 정답 n개 · 상위 k개)
  Recall@k  = 상위 k 안의 정답 수 / n
  HitRate@k = 상위 k 안에 정답이 하나라도 있으면 1
  NDCG@k    = Σ_{정답의 순위 r < k} 1 / log2(r + 2)  ÷  Σ_{j < min(n, k)} 1 / log2(j + 2)
  MAP@k     = (1 / min(n, k)) × Σ_{j < k} 정밀도@(j+1) × 적중_j   (H&M 대회 지표)
그 외
  Coverage@k            = 한 번이라도 상위 k에 나온 상품 수 / 카탈로그 상품 수
  BrandDiversity@k      = 고객별 (상위 k의 서로 다른 브랜드 수 / k) 평균
  IntraListDiversity@k  = 고객별 상위 k 상품 쌍의 (1 − 코사인 유사도) 평균의 평균 (이미지 임베딩 기준)
  PopularityPct@k       = 상위 k 상품의 학습 기간 인기 백분위(0 = 가장 인기) 평균. 작을수록 인기 상품 위주
  BrandAlignment@k      = 고객별 (상위 k 중 그 고객의 최다 브랜드 상품 비율) 평균
  UserBrandLoyalty      = 고객별 최다 브랜드 이벤트 수 / 전체 이벤트 수 (데이터 특성, 추천과 무관)
"""
import numpy as np
import pandas as pd


def _ranks(rec, users, truth):
    """정답 (user, item)마다 rec 행 안의 순위(0부터), 없으면 −1. 반환: (정답의 행 번호, 순위)."""
    row_of = pd.Series(np.arange(len(users)), index=users)
    rows = row_of.reindex(truth.user_id.values).values
    if np.isnan(rows.astype(float)).any():
        raise ValueError("정답 고객 중 rec에 없는 고객이 있습니다")
    rows = rows.astype(np.int64)
    n_items = int(max(rec.max(), truth.item_id.max())) + 1
    rec_keys = (np.arange(len(rec), dtype=np.int64)[:, None] * n_items + rec).ravel()
    pos = pd.Index(rec_keys).get_indexer(rows * n_items + truth.item_id.values.astype(np.int64))
    rank = np.where(pos >= 0, pos % rec.shape[1], -1)
    return rows, rank


def accuracy_at_k(rec, users, truth, ks=(12, 300)):
    """{f"recall@{k}": 평균, ...} 와 고객별 값 {이름: (U,) 배열}."""
    U = len(users)
    rows, rank = _ranks(rec, users, truth)
    n = np.bincount(rows, minlength=U).astype(float)
    ideal_tab = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(max(ks)) + 2.0))])
    per_user, means = {}, {}
    for k in ks:
        assert k <= rec.shape[1], f"rec 열 수({rec.shape[1]}) < k({k})"
        hit = (rank >= 0) & (rank < k)
        h = np.bincount(rows, weights=hit, minlength=U)
        gain = np.where(hit, 1.0 / np.log2(np.maximum(rank, 0) + 2.0), 0.0)
        r, kk = rows[hit], rank[hit]
        o = np.lexsort((kk, r))
        r, kk = r[o], kk[o]
        j = pd.Series(np.ones(len(r))).groupby(r).cumsum().values if len(r) else np.zeros(0)
        m = np.minimum(n, k)
        vals = {
            f"recall@{k}": h / n,
            f"hitrate@{k}": (h > 0).astype(float),
            f"ndcg@{k}": np.bincount(rows, weights=gain, minlength=U) / ideal_tab[m.astype(int)],
            f"map@{k}": np.bincount(r, weights=j / (kk + 1.0), minlength=U) / m,
        }
        for name, v in vals.items():
            per_user[name] = v
            means[name] = float(v.mean())
    return means, per_user


def coverage_at_k(rec, k, n_catalog):
    return float(len(np.unique(rec[:, :k])) / n_catalog)


def brand_diversity_at_k(rec, k, brand_of):
    """brand_of: 상품 id → 브랜드 id 배열 (id로 인덱싱)."""
    b = brand_of[rec[:, :k]]
    distinct = np.array([len(np.unique(row)) for row in b])
    return float((distinct / k).mean())


def intra_list_diversity_at_k(rec, k, emb, chunk=4096):
    """emb: 상품 id로 인덱싱되는 L2 정규화 임베딩 (예: embeddings/v2 이미지). 임베딩이 0인 상품은 제외 없이 유사도 0으로 계산."""
    out = []
    iu = np.triu_indices(k, 1)
    for s in range(0, len(rec), chunk):
        E = emb[rec[s:s + chunk, :k]]                       # (c, k, d)
        sim = np.einsum("ckd,cjd->ckj", E, E)
        out.append((1.0 - sim[:, iu[0], iu[1]]).mean(axis=1))
    return float(np.concatenate(out).mean())


def popularity_percentile_at_k(rec, k, history):
    """학습 이력 기준 인기 백분위: 0 = 가장 인기, 1 = 이력 없음에 가까움."""
    cnt = history.item_id.value_counts()
    pct = pd.Series(np.arange(len(cnt)) / max(len(cnt) - 1, 1), index=cnt.index)
    return float(pd.Series(rec[:, :k].ravel()).map(pct).fillna(1.0).mean())


def user_brand_loyalty(history, brand_of):
    """고객별 최다 브랜드 비중 (Series, index = user_id)."""
    b = pd.DataFrame({"user_id": history.user_id.values, "brand": brand_of[history.item_id.values]})
    c = b.groupby(["user_id", "brand"]).size()
    return c.groupby(level=0).max() / c.groupby(level=0).sum()


def brand_alignment_at_k(rec, users, k, history, brand_of):
    """고객별 상위 k 중 그 고객의 최다 브랜드(학습 이력 기준) 상품 비율의 평균."""
    b = pd.DataFrame({"user_id": history.user_id.values, "brand": brand_of[history.item_id.values]})
    top = b.groupby(["user_id", "brand"]).size().reset_index(name="c").sort_values(["user_id", "c"], ascending=[True, False])
    top_brand = top.drop_duplicates("user_id").set_index("user_id").brand.reindex(users).values
    return float((brand_of[rec[:, :k]] == top_brand[:, None]).mean())
