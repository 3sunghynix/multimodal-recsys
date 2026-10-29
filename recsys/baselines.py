"""기준 모델 (오프라인). 데이터셋(source)마다 따로 학습 · 추천한다.

popularity: 학습 이력의 긍정 이벤트(feedback > 0) 수 상위 k개를 모든 고객에게 같은 순서로.
            recent_days를 주면 학습 이력의 마지막 날짜 기준 최근 N일만 센다.
"""
import numpy as np
import pandas as pd


def popularity(history: pd.DataFrame, users: np.ndarray, k: int, recent_days: int | None = None) -> np.ndarray:
    h = history[history.feedback > 0]
    if recent_days is not None:
        h = h[h.ts >= h.ts.max() - pd.Timedelta(days=recent_days)]
    top = h.item_id.value_counts().index.values[:k]
    if len(top) < k:                                  # 이력이 적으면 나머지는 전체 이력 인기 순으로
        rest = history.item_id.value_counts().index.values
        top = np.concatenate([top, rest[~np.isin(rest, top)]])[:k]
    return np.tile(top.astype(np.int64), (len(users), 1))
