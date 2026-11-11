"""검색 점수 계산 · 결합 (모델 없이 numpy만). 백엔드 /search가 같은 규칙을 쓴다.

질의 벡터 → 상품 임베딩과 코사인 → (데이터셋별 표준화) → 텍스트 · 이미지 가중 합 → 상위 k

- 데이터셋별 표준화(per_source_z): H&M과 Amazon은 임베딩 공간에서 떨어져 있어서(진단 D1, 다른 데이터셋 이웃 0.5% 미만)
  원래 코사인을 그대로 섞으면 한쪽 데이터셋이 결과를 독차지한다. 데이터셋마다 그 질의의 점수를
  평균 0 · 표준편차 1로 바꾼 뒤 합친다 = "그 데이터셋 안에서 얼마나 두드러지게 맞는가".
- 텍스트 · 이미지 우선순위: 각 점수를 표준화한 뒤 w_text · w_image로 가중 합 (화면의 슬라이더).
"""
import numpy as np
import pandas as pd

MERGES = ("global", "per_source_z", "per_source_rank")


def group_rows(groups) -> list[np.ndarray]:
    """그룹(데이터셋)별 행 번호 목록 (그룹 이름 순). 이미 목록이면 그대로.
    np.unique(문자열 배열 10만) 한 번에 약 25ms라, 한 질의에서 여러 번 부르지 않도록 한 번만 계산해 넘긴다."""
    if isinstance(groups, list):
        return groups
    codes, _ = pd.factorize(np.asarray(groups), sort=True)
    return [np.flatnonzero(codes == c) for c in range(codes.max() + 1)] if len(codes) else []


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 0, n, 1)


def project(v: np.ndarray, mean: np.ndarray, components: np.ndarray, center: bool = True) -> np.ndarray:
    """인코더 출력 → 상품 임베딩과 같은 공간: L2 → (v − mean) @ components.T → L2 (embeddings/v2/pca_*.npz).

    center=False: 다른 모달리티 질의(예: Marqo 텍스트 → 이미지 임베딩)용. 이미지 평균은 텍스트 질의와 멀어서
    빼면 모든 질의가 '평균의 반대 방향'으로 쏠린다. 원래 공간의 코사인 순위와 같은 것은 빼지 않는 쪽이다.
    """
    v = l2(v)
    return l2(((v - mean) if center else v) @ components.T)


def standardize(scores: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """scores: (Q, N). groups(N,)가 있으면 그룹(데이터셋)마다, 없으면 전체로 평균 0 · 표준편차 1."""
    s = np.asarray(scores, np.float32)
    if groups is None:
        return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)
    out = np.empty_like(s)
    for m in group_rows(groups):
        x = s[:, m]
        out[:, m] = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    return out


def combine(scores: dict, weights: dict, groups: np.ndarray | None, merge: str = "per_source_z",
            min_rel: float | None = None, k: int | None = None) -> np.ndarray:
    """scores: {"text": (Q, N), "image": (Q, N)} 코사인. weights: {"text": 0.7, "image": 0.3}.

    global: 모달리티가 하나면 코사인 그대로, 여럿이면 전체 기준 표준화 후 가중 합 (데이터셋 구분 없음).
    per_source_z: 데이터셋별 표준화 후 가중 합.
    per_source_rank: per_source_z 점수를 데이터셋 안의 순위 백분위로 바꾼다 (1 = 그 데이터셋 1위).
        상위 k가 카탈로그 비율대로 섞인다 (H&M 19% → 상위 20 중 약 4개). 표준화만으로는 질의마다
        꼬리가 더 긴 데이터셋이 통째로 차지하는 문제(실측: 두 데이터셋이 함께 나온 질의 46%)를 막는다.
    min_rel (per_source_rank 전용): 데이터셋의 최고 상품조차 원래 코사인(가중 평균)이 그 질의의 카탈로그 최고값 × min_rel
        미만이면 그 데이터셋은 이번 질의에서 **후순위**가 된다 — 비율 칸을 받지 못하고, 다른 데이터셋 상품을 모두 쓴 뒤에만
        (유사도 순으로) 채워진다. 제외가 아니므로 k가 다른 데이터셋 상품 수보다 크면 나올 수 있고, 결과는 항상 k개
        (오프라인 평가의 고정 길이와 같음). 참여 데이터셋끼리는 비율대로 섞는다.
        상품 단위로 자르면 코사인이 전반적으로 높은 H&M이 칸을 독차지한다 (두 데이터셋이 임베딩 공간에서 떨어져 있음, 진단 D1).
        맞는 상품이 없는 데이터셋의 "그중 1등"이
        끼는 문제 방지 — 데이터셋 내 z는 상대값이라 이를 구분하지 못한다 (실측: 러닝화 검색의 H&M 양말 z 2.8 ≈ 정답 수준,
        코사인은 0.2 vs 정답 0.4 이상).
    """
    assert merge in MERGES, merge
    used = [m for m, w in weights.items() if w > 0 and m in scores]
    assert used, "가중치가 0보다 큰 모달리티가 없습니다"
    if merge == "global" and len(used) == 1:
        return np.asarray(scores[used[0]], np.float32)
    rows = group_rows(groups) if groups is not None else None     # 한 번만 계산해 아래에서 재사용
    g = None if merge == "global" else rows
    total = sum(weights[m] for m in used)
    z = sum(weights[m] / total * standardize(scores[m], g) for m in used)
    if merge != "per_source_rank":
        return z
    # k를 주면 상위 k 결과만 정확히 (서빙). 없으면 전체 순위 백분위 (오프라인 평가 · 분석)
    out = rank_percentile(z, rows) if k is None else rank_percentile_top(z, rows, k)
    if min_rel:
        raw = sum(weights[m] / total * np.asarray(scores[m], np.float32) for m in used)
        floor = raw.max(axis=1, keepdims=True) * min_rel
        # 데이터셋의 최고 상품도 기준 미달이면 그 데이터셋은 후순위 (예: 러닝화 검색의 H&M 양말, 치마 검색의 H&M 반바지).
        # 참여하는 데이터셋끼리는 그대로 비율 순서. 후순위 데이터셋은 맨 뒤(유사도 순)
        absent = np.zeros(raw.shape, bool)
        for m in rows:
            absent[:, m] = raw[:, m].max(axis=1, keepdims=True) < floor
        out = np.where(absent, raw - 5.0, out)
    return out


def rank_percentile(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """그룹마다 (Q, n_g) 점수 → 1 − 순위/n_g (1위 = 1, 꼴찌 ≈ 0). 그룹 크기가 달라도 같은 척도."""
    out = np.empty_like(scores, dtype=np.float32)
    for m in group_rows(groups):
        r = (-scores[:, m]).argsort(1).argsort(1)
        out[:, m] = 1.0 - r / len(m)
    return out


def rank_percentile_top(scores: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    """rank_percentile과 같은 상위 결과를 훨씬 빠르게 — 그룹마다 상위 k개만 정확한 백분위를 매긴다.
    최종 상위 k는 각 그룹의 상위 k 안에서만 나오므로 top_k 결과는 rank_percentile과 같다.
    나머지 상품은 그 그룹 k위 백분위보다 낮은 값(−1 + 원래 순서)으로 둔다 — 순서만 유지, 값은 의미 없음.
    (전체 argsort 두 번 → argpartition 한 번: 상품 10만 기준 약 140ms → 수 ms)"""
    s = np.asarray(scores, np.float32)
    out = np.empty_like(s)
    for m in group_rows(groups):
        x = s[:, m]
        n, kk = len(m), min(k, len(m))
        lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
        out[:, m] = -1.0 + (x - lo) / (hi - lo + 1e-6) * 0.5          # 상위 k 밖: −1 ~ −0.5 (후순위 데이터셋 raw−5 보다는 위)
        part = np.argpartition(-x, kk - 1, axis=1)[:, :kk]
        order = np.take_along_axis(x, part, 1).argsort(1, kind="stable")[:, ::-1]
        part = np.take_along_axis(part, order, 1)
        np.put_along_axis(out, m[part], (1.0 - np.arange(kk) / n)[None, :].astype(np.float32), 1)
    return out


def top_k(final: np.ndarray, k: int) -> np.ndarray:
    """(Q, N) 점수 → (Q, k) 상위 인덱스 (내림차순)."""
    k = min(k, final.shape[1])
    part = np.argpartition(-final, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(final, part, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(part, order, 1)


def search(queries: dict, items: dict, weights: dict, groups: np.ndarray | None, k: int = 20,
           merge: str = "per_source_z", min_rel: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """queries: {"text": (Q, d) 질의 벡터}, items: {"text": (N, d) 상품 벡터}. 모두 L2 정규화된 같은 공간.

    반환: (Q, k) 상품 행 번호, (Q, k) 최종 점수.
    """
    scores = {m: queries[m] @ items[m].T for m in queries if weights.get(m, 0) > 0}
    final = combine(scores, weights, groups, merge, min_rel, k=k)
    idx = top_k(final, k)
    return idx, np.take_along_axis(final, idx, 1)
