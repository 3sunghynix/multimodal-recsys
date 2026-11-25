"""2단계 학습형 랭커 (B1): 규칙 점수 상위 후보의 특성 계산 · 모델 읽기. 평가(recsys/ranker_eval.py) · 학습(recsys/ranker_train.py) ·
서빙(recommenders/ume.py)이 **같은 함수**로 특성을 만든다 — 오프라인 결과를 서빙에서 그대로 믿을 수 있게.

특성 (고객 1명 · 후보 상품들; 순서 = FEATURES)
  s_text · s_image     그 고객의 텍스트 · 이미지 타워 점수 (데이터셋 상품 전체로 표준화한 z)
  rule_rank            후보 안 규칙 점수 순위 (0 = 1위)
  pop_z                인기 z (서비스 인기 정의, ume._popularity_z)
  brand_match          고객 구매 중 최다 브랜드와 같은 브랜드
  bought_n             같은 상품을 산 횟수
  days_since_bought    그 상품을 마지막으로 산 뒤 경과일 (기준 = 그 데이터셋 서비스 구매의 마지막 시각, 안 샀으면 결측)
  variant_bought       같은 옷(색상 변형 묶음)의 다른 상품을 산 적 있음
  cat_share            고객 구매 중 같은 소분류 비율
  price_gap            log(1 + 가격) − 고객 구매 평균 log(1 + 가격)
  hist_len             고객 타워 입력 이력 수 (최대 max_hist)
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

FEATURES = ["s_text", "s_image", "rule_rank", "pop_z", "brand_match", "bought_n", "days_since_bought",
            "variant_bought", "cat_share", "price_gap", "hist_len"]
CANDIDATES = 300
MODEL_FILE, META_FILE = "model.joblib", "meta.json"


def candidate_features(cand: np.ndarray, s_text: np.ndarray, s_image: np.ndarray, pop_z: np.ndarray, brands: np.ndarray,
                       cats: np.ndarray, variants: np.ndarray, log_price: np.ndarray, bought: np.ndarray,
                       bought_ts: np.ndarray, t_ref, hist_len: int) -> np.ndarray:
    """(len(cand), len(FEATURES)) float32. 배열은 모두 같은 데이터셋 상품 순번 기준.

    cand: 후보 순번 (규칙 점수 내림차순). s_text · s_image · pop_z · brands · cats · variants · log_price: 상품 전체 배열.
    cats · variants는 문자열 ("" = 없음). bought · bought_ts: 고객 구매 (상품 순번, datetime64) — 이력 전체.
    """
    n_c = len(cand)
    bought = np.asarray(bought, np.int64)
    f = np.zeros((n_c, len(FEATURES)), np.float32)
    f[:, 0], f[:, 1] = s_text[cand], s_image[cand]
    f[:, 2] = np.arange(n_c)
    f[:, 3] = pop_z[cand]
    b = brands[bought]
    b = b[b >= 0]
    fav = np.bincount(b).argmax() if len(b) else -2
    f[:, 4] = brands[cand] == fav
    f[:, 6] = np.nan
    if len(bought):
        items, counts = np.unique(bought, return_counts=True)
        pos = np.searchsorted(items, cand)
        hit = (pos < len(items)) & (items[np.minimum(pos, len(items) - 1)] == cand)
        f[hit, 5] = counts[pos[hit]]
        ts = np.asarray(bought_ts, "datetime64[ns]")
        last = {}
        for i, t in zip(bought.tolist(), ts):
            if i not in last or t > last[i]:
                last[i] = t
        t_ref = np.datetime64(t_ref, "ns")
        for r in np.flatnonzero(hit):
            f[r, 6] = (t_ref - last[int(cand[r])]) / np.timedelta64(1, "D")
        bv = {v for v in variants[bought].tolist() if v}
        f[:, 7] = [bool(v) and v in bv for v in variants[cand].tolist()]
        bc = [c for c in cats[bought].tolist() if c]
        if bc:
            u, cnt = np.unique(np.asarray(bc, object), return_counts=True)
            share = dict(zip(u.tolist(), (cnt / cnt.sum()).tolist()))
            f[:, 8] = [share.get(c, 0.0) if c else 0.0 for c in cats[cand].tolist()]
        lp = log_price[bought]
        mean_lp = np.nanmean(lp) if np.isfinite(lp).any() else np.nan
    else:
        mean_lp = np.nan
    f[:, 9] = log_price[cand] - mean_lp
    f[:, 10] = hist_len
    return f


def log_price(prices) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return np.log1p(np.where(p > 0, p, np.nan))


def ranker_problem(path: str) -> str | None:
    """랭커 폴더 검사. 문제 없으면 None (서빙은 문제가 있으면 그 데이터셋을 규칙 점수로 둔다)."""
    d = Path(path)
    for name in (MODEL_FILE, META_FILE):
        if not (d / name).exists():
            return f"{d / name} 없음"
    try:
        meta = json.loads((d / META_FILE).read_text())
    except Exception as e:
        return f"{d / META_FILE} 읽기 실패: {type(e).__name__}"
    unknown = [k for k in meta.get("features", []) if k not in FEATURES]
    if not meta.get("features") or unknown:
        return f"{d / META_FILE} 특성 목록 이상: {unknown or '비어 있음'}"
    return None


@lru_cache(maxsize=8)
def _load(path: str, mtime: float):
    import joblib
    d = Path(path)
    meta = json.loads((d / META_FILE).read_text())
    return joblib.load(d / MODEL_FILE), [FEATURES.index(k) for k in meta["features"]], meta


def load_ranker(path: str):
    """(모델, 쓰는 특성 열 번호, meta). 파일이 바뀌면 다시 읽는다."""
    return _load(str(path), (Path(path) / MODEL_FILE).stat().st_mtime)


def save_ranker(path: str, model, features: list[str], meta: dict) -> None:
    import joblib
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / MODEL_FILE)
    (d / META_FILE).write_text(json.dumps({**meta, "features": features}, ensure_ascii=False, indent=2, default=str))


def predict(path: str, feats: np.ndarray) -> np.ndarray:
    """후보 300개 정도는 한 스레드가 빠르다. OpenMP 스레드를 쓰면 torch 스레드 풀과 부딪혀 수십 ms씩 늦어진다
    (실측: 요청 안에서 63ms → 한 스레드 수 ms)."""
    from threadpoolctl import threadpool_limits
    model, cols, _ = load_ranker(path)
    with threadpool_limits(limits=1, user_api="openmp"):
        return model.predict_proba(feats[:, cols])[:, 1]
