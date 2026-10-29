"""평가 모듈 검증: 지표를 직접 계산(느린 정의 그대로)과 비교하고, 평가 프로토콜 · 기준 모델 · 실행 스크립트를 합성 데이터로 확인."""
import json

import numpy as np
import pandas as pd
import pytest

from recsys import baselines
from recsys.eval import (accuracy_at_k, brand_alignment_at_k, brand_diversity_at_k, build_eval_set, coverage_at_k,
                         intra_list_diversity_at_k, popularity_percentile_at_k, user_brand_loyalty)
from recsys.run_eval import evaluate


def brute(rec_row, truth_set, k):
    top = list(rec_row[:k])
    n = len(truth_set)
    hits = [i in truth_set for i in top]
    recall = sum(hits) / n
    dcg = sum(1 / np.log2(j + 2) for j, h in enumerate(hits) if h)
    idcg = sum(1 / np.log2(j + 2) for j in range(min(n, k)))
    c, s = 0, 0.0
    for j, h in enumerate(hits):
        if h:
            c += 1
            s += c / (j + 1)
    return recall, float(any(hits)), dcg / idcg, s / min(n, k)


def test_accuracy_matches_definition():
    rng = np.random.default_rng(0)
    U, N, K = 200, 400, 50
    users = np.arange(1000, 1000 + U)
    rec = np.stack([rng.permutation(N)[:K] for _ in range(U)])
    rows = []
    for u in range(U):
        n = rng.integers(1, 15)
        picks = set(rng.choice(rec[u, :20], size=min(n, 3), replace=False)) | set(rng.integers(0, N, n))
        rows += [(users[u], int(i)) for i in picks]
    truth = pd.DataFrame(rows, columns=["user_id", "item_id"]).drop_duplicates()
    means, per = accuracy_at_k(rec, users, truth, ks=(12, 50))
    ts = truth.groupby("user_id").item_id.agg(set)
    for k in (12, 50):
        b = np.array([brute(rec[u], ts[users[u]], k) for u in range(U)])
        for col, name in enumerate(("recall", "hitrate", "ndcg", "map")):
            assert np.allclose(per[f"{name}@{k}"], b[:, col]), (name, k)
            assert means[f"{name}@{k}"] == pytest.approx(b[:, col].mean())


def test_other_metrics():
    rec = np.array([[0, 1, 2], [2, 3, 4]])
    brand_of = np.array([0, 0, 1, 1, 2])
    assert coverage_at_k(rec, 2, 10) == pytest.approx(4 / 10)                     # {0, 1, 2, 3}
    assert brand_diversity_at_k(rec, 2, brand_of) == pytest.approx((1 / 2 + 1 / 2) / 2)
    emb = np.eye(5, 4, dtype=np.float32)                                          # 서로 직교 → 다양성 1
    emb[1] = emb[0]                                                               # 0 · 1은 같음 → 다양성 0
    assert intra_list_diversity_at_k(rec, 2, emb) == pytest.approx((0 + 1) / 2)
    hist = pd.DataFrame({"user_id": [7, 7, 7, 8], "item_id": [2, 2, 0, 4]})
    assert popularity_percentile_at_k(np.array([[2]]), 1, hist) == pytest.approx(0.0)   # 가장 인기 상품
    loyal = user_brand_loyalty(hist, brand_of)
    assert loyal[7] == pytest.approx(2 / 3) and loyal[8] == pytest.approx(1.0)
    # 고객 7의 최다 브랜드 = 1(상품 2 두 번), 고객 8 = 2
    assert brand_alignment_at_k(np.array([[2, 0], [4, 0]]), np.array([7, 8]), 2, hist, brand_of) == pytest.approx((1 / 2 + 1 / 2) / 2)


def make_events():
    ts = pd.to_datetime(["2023-01-01"] * 6 + ["2023-02-01"] * 3 + ["2023-03-01"] * 4)
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 1, 2, 4, 1, 2, 3, 4],
        "item_id": [10, 11, 10, 12, 11, 13, 12, 13, 10, 13, 14, 10, 11],
        "feedback": [1, 1, 1, -1, 1, 0.5, 1, 1, 1, 1, 0.5, 1, 1],
        "ts": ts, "split": ["train"] * 6 + ["valid"] * 3 + ["test"] * 4})


def test_protocol():
    ev = make_events()
    t = build_eval_set(ev, "test")
    # 고객 4는 test 정답이 있지만 train · valid 중 valid 이력만 있음 → test에서는 train+valid가 이력이라 포함
    # 고객 2의 test 이벤트는 feedback 0.5 → 정답 아님
    assert set(t.users) == {1, 3, 4} and set(map(tuple, t.truth.values)) == {(1, 13), (3, 10), (4, 11)}
    assert set(t.history.split) == {"train", "valid"}
    # 학습 이력은 평가 고객이 아닌 고객(2)의 이벤트도 포함 → 인기 모델 등이 전체 로그로 학습
    assert len(t.history) == 9 and 2 in set(t.history.user_id)
    v = build_eval_set(ev, "valid")
    # valid 평가는 train만 이력 → 고객 4(train 이력 없음)는 제외
    assert set(v.users) == {1, 2} and set(v.history.split) == {"train"}
    assert len(build_eval_set(ev, "test", positive_min=0.5).truth) == 4


def test_popularity_baseline():
    ev = make_events()
    hist = ev[ev.split == "train"]
    rec = baselines.popularity(hist, np.array([1, 2]), 3)
    assert rec.shape == (2, 3) and rec[0, 0] == 10 and (rec[0] == rec[1]).all()       # 상품 10이 긍정 2회로 1위
    assert 12 not in rec[0, :2]                                                          # 12는 feedback −1 뿐


def test_run_eval_end_to_end(tmp_path):
    ev = make_events()
    ev.to_parquet(tmp_path / "events.parquet", index=False)
    pd.DataFrame({"item_id": [10, 11, 12, 13, 14], "brand_norm": ["a", "a", "b", "c", "c"]}).to_parquet(
        tmp_path / "items.parquet", index=False)
    emb_dir = tmp_path / "v2"
    emb_dir.mkdir()
    pd.DataFrame({"row": range(5), "source": "hm", "item_id": [10, 11, 12, 13, 14]}).to_parquet(emb_dir / "catalog.parquet")
    np.save(emb_dir / "image_pca256.npy", np.eye(5, 8, dtype=np.float32))
    with pytest.raises(AssertionError):                    # 상품이 5개라 k = 300 불가 → 명확히 실패해야 함
        evaluate(tmp_path, "hm", embeddings=emb_dir)
    import recsys.run_eval as re_
    re_.KS = (2, 3)
    try:
        res = evaluate(tmp_path, "hm", embeddings=emb_dir)
    finally:
        re_.KS = (12, 300)
    json.dumps(res)                                         # 직렬화 가능
    assert res["eval"]["users"] == 3 and 0 <= res["metrics"]["recall@3"] <= 1
    assert {"coverage@2", "brand_diversity@3", "popularity_pct@2", "brand_alignment@3"} <= set(res["metrics"])
