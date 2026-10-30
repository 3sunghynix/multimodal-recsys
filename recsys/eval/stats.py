"""고객 단위 짝지은 부트스트랩: 같은 평가 고객에서 두 모델의 지표 차이(B − A)의 95% 구간.

고객을 복원 추출해 고객별 값 차이의 평균을 n_boot번 계산한다 (논문과 같은 방식).
구간이 0을 포함하지 않으면 "차이가 있다", 포함하면 "차이 불확실" (같다는 뜻이 아님).
"""
import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0, chunk: int = 100) -> dict:
    """a, b: 같은 고객 순서의 고객별 지표 (n,). 반환: 차이 평균 · 95% 구간 · 상대 변화."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape and a.ndim == 1
    d = b - a
    rng = np.random.default_rng(seed)
    means = []
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, len(d), size=(min(chunk, n_boot - s), len(d)))
        means.append(d[idx].mean(axis=1))
    means = np.concatenate(means)
    lo, hi = np.percentile(means, [2.5, 97.5])
    base = a.mean()
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "rel": float(d.mean() / base) if base else None, "n_users": int(len(d)), "n_boot": n_boot}
