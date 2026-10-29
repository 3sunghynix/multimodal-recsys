"""검색 점수 계산 · 결합 (모델 없이 numpy만). 백엔드 /search가 같은 규칙을 쓴다.

질의 벡터 → 상품 임베딩과 코사인 → (데이터셋별 표준화) → 텍스트 · 이미지 가중 합 → 상위 k

- 데이터셋별 표준화(per_source_z): H&M과 Amazon은 임베딩 공간에서 떨어져 있어서(진단 D1, 다른 데이터셋 이웃 0.5% 미만)
  원래 코사인을 그대로 섞으면 한쪽 데이터셋이 결과를 독차지한다. 데이터셋마다 그 질의의 점수를
  평균 0 · 표준편차 1로 바꾼 뒤 합친다 = "그 데이터셋 안에서 얼마나 두드러지게 맞는가".
- 텍스트 · 이미지 우선순위: 각 점수를 표준화한 뒤 w_text · w_image로 가중 합 (화면의 슬라이더).
"""
import numpy as np

MERGES = ("global", "per_source_z")


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
    for g in np.unique(groups):
        m = groups == g
        x = s[:, m]
        out[:, m] = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    return out


def combine(scores: dict, weights: dict, groups: np.ndarray | None, merge: str = "per_source_z") -> np.ndarray:
    """scores: {"text": (Q, N), "image": (Q, N)} 코사인. weights: {"text": 0.7, "image": 0.3}.

    global: 모달리티가 하나면 코사인 그대로, 여럿이면 전체 기준 표준화 후 가중 합 (데이터셋 구분 없음).
    per_source_z: 데이터셋별 표준화 후 가중 합.
    """
    assert merge in MERGES, merge
    used = [m for m, w in weights.items() if w > 0 and m in scores]
    assert used, "가중치가 0보다 큰 모달리티가 없습니다"
    if merge == "global" and len(used) == 1:
        return np.asarray(scores[used[0]], np.float32)
    g = groups if merge == "per_source_z" else None
    total = sum(weights[m] for m in used)
    return sum(weights[m] / total * standardize(scores[m], g) for m in used)


def top_k(final: np.ndarray, k: int) -> np.ndarray:
    """(Q, N) 점수 → (Q, k) 상위 인덱스 (내림차순)."""
    k = min(k, final.shape[1])
    part = np.argpartition(-final, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(final, part, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(part, order, 1)


def search(queries: dict, items: dict, weights: dict, groups: np.ndarray | None, k: int = 20,
           merge: str = "per_source_z") -> tuple[np.ndarray, np.ndarray]:
    """queries: {"text": (Q, d) 질의 벡터}, items: {"text": (N, d) 상품 벡터}. 모두 L2 정규화된 같은 공간.

    반환: (Q, k) 상품 행 번호, (Q, k) 최종 점수.
    """
    scores = {m: queries[m] @ items[m].T for m in queries if weights.get(m, 0) > 0}
    final = combine(scores, weights, groups, merge)
    idx = top_k(final, k)
    return idx, np.take_along_axis(final, idx, 1)
