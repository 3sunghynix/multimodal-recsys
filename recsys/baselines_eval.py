"""B2 비교 모델: 인기 · ItemKNN · UME(지금 서비스 규칙)를 같은 조건에서 (서비스 DB 구매 로그, pop_eval과 같은 두 분할 · 고객 5,000명).

실행: docker compose --profile trainer run --rm trainer python -m recsys.baselines_eval   (결과: docs/project/추천모델설계.md "비교 모델")
학습형 랭커(B1)는 recsys.ranker_eval — 같은 분할 · 같은 고객이라 그 표의 수치와 바로 비교된다.

모델
- popularity: 인기 z (서비스 정의, H&M 최근 7일 · Amazon 전체)
- itemknn:    같이 산 상품 코사인 유사도 (아이템 기반 협업 필터링, Sarwar et al., 2001).
              점수(u, i) = Σ_{j ∈ u가 산 상품} co(j, i) / √(n_j · n_i)  (co = j와 i를 모두 산 고객 수, n = 산 고객 수).
              점수가 같으면(이웃이 없는 상품 등) 인기 순 (1e-6 · 인기 z).
- ume_rule:   지금 서비스 규칙 = z(text) + z(image) + γ · 인기 z [+ β · 선호 브랜드], H&M 최근성 λ = 0.9 (C4)
모든 모델에 서비스와 같은 재구매 정책을 적용한다 (Amazon은 산 상품 제외, H&M은 남김 — C3). 재정렬 없음.
지표: MAP@12 · Recall@12 · Coverage@12, ItemKNN − UME 규칙 paired bootstrap 95% 구간.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/app")
from app.db import SessionLocal  # noqa: E402
from app.services.catalog_index import get_index  # noqa: E402
from recsys.eval.metrics import accuracy_at_k  # noqa: E402
from recsys.eval.stats import paired_bootstrap  # noqa: E402
from recsys.pop_eval import BETA, K, POP_DAYS, pop_z, service_purchases, split, zrows  # noqa: E402
from recsys.ranker_eval import EXCLUDE, GAMMA, Catalog, Users  # noqa: E402

CHUNK = 250


def itemknn_scorer(train: pd.DataFrame, n: int):
    """고객 묶음 → (묶음, n) 점수. 행렬: X (고객 × 상품, 이진). X · D^-1/2 → (Xu Xᵀ) X → · D^-1/2 (희소 행렬 곱만)."""
    from scipy import sparse
    pairs = train[["user_id", "item_id"]].drop_duplicates()
    uid, urow = np.unique(pairs.user_id.values, return_inverse=True)
    X = sparse.csr_matrix((np.ones(len(pairs), np.float32), (urow, pairs.item_id.values)), shape=(len(uid), n))
    cnt = np.asarray(X.sum(0)).ravel()
    inv = np.where(cnt > 0, 1.0 / np.sqrt(np.maximum(cnt, 1)), 0.0).astype(np.float32)
    Xt = X.T.tocsr()
    row_of = pd.Series(np.arange(len(uid)), index=uid)

    def score(users: np.ndarray) -> np.ndarray:
        rows = row_of.reindex(users).values
        ok = ~np.isnan(rows)
        out = np.zeros((len(users), n), np.float32)
        if ok.any():
            Xu = X[rows[ok].astype(np.int64)].multiply(inv[None, :]).tocsr()      # 고객이 산 상품 j · 1/√n_j
            S = (Xu @ Xt) @ X                                                     # Σ_j co(j, i) / √n_j
            out[ok] = S.toarray() * inv[None, :]
        return out
    return score


def evaluate(db, ix, source: str, how: str) -> dict:
    cat = Catalog(db, ix, source)
    ev = service_purchases(db, cat.ids)
    train, users, truth = split(ev, how)
    us = Users(cat, train, users)
    knn = itemknn_scorer(train, cat.n)
    pz = us.pz
    empty = (np.array([], np.int64), None)
    recs = {"popularity": [], "itemknn": [], "ume_rule": []}
    for s in range(0, len(users), CHUNK):
        sl = slice(s, s + CHUNK)
        rows = np.arange(sl.start, min(sl.stop, len(users)))
        st = zrows(us.U["text"][sl] @ cat.V["text"].T)
        si = zrows(us.U["image"][sl] @ cat.V["image"].T)
        rule = st + si + GAMMA * pz[None, :]
        if source in BETA:
            rule = rule + BETA[source] * (cat.brands[None, :] == us.fav[sl, None])
        scores = {"popularity": np.broadcast_to(pz, rule.shape).copy(), "itemknn": knn(users[sl]) + 1e-6 * pz[None, :],
                  "ume_rule": rule}
        for name, S in scores.items():
            if source in EXCLUDE:
                for r, u in enumerate(rows):
                    S[r, us.bought.get(u, empty)[0]] = -np.inf
            top = np.argpartition(-S, K, axis=1)[:, :K]
            recs[name].append(np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1, kind="stable"), 1))
    out = {"eval_users": int(len(users)), "catalog": int(cat.n), "test_window": how}
    per = {}
    for name, parts in recs.items():
        rec = np.concatenate(parts)
        acc, per[name] = accuracy_at_k(rec, users, truth, ks=(K,))
        out[name] = {"map@12": round(acc["map@12"], 5), "recall@12": round(acc["recall@12"], 5),
                     "coverage@12": round(len(np.unique(rec)) / cat.n, 4)}
    out["itemknn_minus_ume_rule"] = {}
    for mt in ("map@12", "recall@12"):
        b = paired_bootstrap(per["ume_rule"][mt], per["itemknn"][mt])
        out["itemknn_minus_ume_rule"][mt] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]],
                                             "rel": round(b["rel"], 3) if b["rel"] is not None else None}
    return out


def main():
    db = SessionLocal()
    ix = get_index(db)
    which = sys.argv[1:] or [f"{s}/{h}" for s in ("hm", "amazon") for h in ("log_lastout", "time_week")]
    res = {}
    for key in which:
        src, how = key.split("/")
        res[key] = evaluate(db, ix, src, how)
        print(json.dumps({key: res[key]}, ensure_ascii=False), flush=True)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
