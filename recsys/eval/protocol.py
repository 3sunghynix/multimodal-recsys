"""평가 프로토콜: 고정 분할(events.split)에서 학습 이력과 정답을 만든다.

- 데이터셋(source)마다 따로 평가한다. 두 데이터셋은 고객 · 상품이 연결되지 않는다.
- 정답: 평가 구간(valid 또는 test)의 이벤트 중 feedback ≥ positive_min 인 (고객, 상품). 기본 1.0 (평점 4 ~ 5점 · H&M 구매)
- 평가 고객: 정답이 1개 이상이고 학습 구간 이력이 있는 고객 (Amazon test 고객의 약 36%는 이력이 없어 제외 — 신규 고객 문제로 따로 다룬다)
- test 평가 때의 학습 이력은 train + valid (최종 성적), valid 평가 때는 train만 (튜닝)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvalSet:
    history: pd.DataFrame       # 학습 이력 (user_id, item_id, feedback, ts, ...)
    truth: pd.DataFrame         # 정답 (user_id, item_id), 중복 없음
    users: np.ndarray           # 평가 고객 (정렬)
    split: str
    positive_min: float

    def summary(self):
        return {"split": self.split, "positive_min": self.positive_min, "users": int(len(self.users)),
                "truth_pairs": int(len(self.truth)), "history_events": int(len(self.history))}


def build_eval_set(events: pd.DataFrame, split: str = "test", positive_min: float = 1.0) -> EvalSet:
    assert split in ("valid", "test"), split
    hist_splits = ("train",) if split == "valid" else ("train", "valid")
    history = events[events.split.isin(hist_splits)]
    target = events[(events.split == split) & (events.feedback >= positive_min)]
    target = target[target.user_id.isin(history.user_id.unique())]
    truth = target[["user_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    users = np.sort(truth.user_id.unique())
    history = history[history.user_id.isin(users)].reset_index(drop=True)
    return EvalSet(history=history, truth=truth, users=users, split=split, positive_min=positive_min)
