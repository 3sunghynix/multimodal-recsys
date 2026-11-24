"""C4 최근성 반영 고객 벡터: 고객 타워 입력을 최근 행동일수록 크게 가중한 평균으로 (서비스 DB 구매 로그, pop_eval과 같은 두 분할).

실행: docker compose --profile trainer run --rm trainer python -m recsys.recency_eval   (결과: docs/project/추천모델설계.md "최근성 반영")

지금 서비스: 고객 타워 입력 = 최근 max_hist(50)개 이력 임베딩의 **단순 평균** (학습도 같은 방식).
비교하는 가중 (이력 i번째, 가장 최근 = 0)
- pos λ:   w = λ^순위          (λ = 1이면 지금과 같음)
- time h:  w = 0.5^(경과일 / h)  (경과일 = 고객의 마지막 이력 시각 − 그 이력 시각, h = 반감기 일수)
순위 = 운영 점수 z(text) + z(image) + γ · 인기 z [+ β · 선호 브랜드] (γ = 1, Amazon β = 2), Amazon은 산 상품 제외 (C3 운영 정책),
재정렬 없음. 지표: MAP@12 · Recall@12 · Coverage@12, 가중마다 − 단순 평균 paired bootstrap 95% 구간.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/app")
from app.db import SessionLocal  # noqa: E402
from app.services.catalog_index import get_index  # noqa: E402
from recsys.eval.metrics import accuracy_at_k  # noqa: E402
from recsys.eval.stats import paired_bootstrap  # noqa: E402
from recsys.pop_eval import BETA, CHUNK, K, POP_DAYS, pop_z, service_purchases, split, zrows  # noqa: E402
from recsys.retrain import _tower_from_npz  # noqa: E402
from recsys.ume import DEFAULT, build_sequences, tower_vectors  # noqa: E402

GAMMA = 1.0
EXCLUDE = {"amazon"}
WEIGHTINGS = {"mean": ("pos", 1.0), "pos0.9": ("pos", 0.9), "pos0.8": ("pos", 0.8), "pos0.7": ("pos", 0.7),
              "pos0.5": ("pos", 0.5), "time7d": ("time", 7.0), "time30d": ("time", 30.0), "time90d": ("time", 90.0)}


def history_rows(train: pd.DataFrame, users: np.ndarray, max_hist: int):
    """고객별 마지막 max_hist개 이력 (상품, 시각, 고객 순번). build_sequences와 같은 정렬 (고객 · 시각, stable)."""
    pos = train[train.feedback > 0].sort_values(["user_id", "ts"], kind="stable")
    pos = pos[pos.user_id.isin(users)].groupby("user_id").tail(max_hist)
    order = pd.Series(np.arange(len(users)), index=users)
    return pos.item_id.values, pos.ts.values, order[pos.user_id.values].values


def weights(kind: str, param: float, ts: np.ndarray, uidx: np.ndarray, n_users: int) -> np.ndarray:
    df = pd.DataFrame({"u": uidx, "ts": ts})
    if kind == "pos":
        rank = df.groupby("u").cumcount(ascending=False).values           # 가장 최근 = 0
        return np.power(param, rank).astype(np.float32)
    last = df.groupby("u").ts.transform("max").values
    days = (last - ts) / np.timedelta64(1, "D")
    return np.power(0.5, days / param).astype(np.float32)


def user_vectors(model, X: np.ndarray, items, uidx, w, n_users) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    num = np.zeros((n_users, X.shape[1]), np.float32)
    np.add.at(num, uidx, X[items] * w[:, None])
    den = np.bincount(uidx, weights=w, minlength=n_users).astype(np.float32)
    inp = num / np.maximum(den, 1e-12)[:, None]
    with torch.no_grad():
        return F.normalize(model["user"](torch.from_numpy(inp)), dim=1).numpy()


def evaluate(db, ix, source: str, how: str) -> dict:
    rows = np.flatnonzero(ix.sources == source)
    ids, n = ix.ids[rows], len(rows)
    ev = service_purchases(db, ids)
    train, users, truth = split(ev, how)

    base = Path(f"/data/models/ume/{source}_test_pca")
    cfg = json.loads((base / "text/config.json").read_text())
    p = {**DEFAULT, **{k: v for k, v in cfg.items() if k in DEFAULT}}
    items, ts, uidx = history_rows(train, users, p["max_hist"])
    X = {"text": ix.text[rows], "image": ix.image[rows]}
    models = {m: _tower_from_npz(dict(np.load(base / m / "weights.npz")), p, "cpu") for m in ("text", "image")}
    seq = build_sequences(train, p["max_hist"])
    V, U_check = {}, {}
    for m in ("text", "image"):
        U_check[m], V[m] = tower_vectors(models[m], X[m], seq, users, np.arange(n), p["max_hist"], "cpu")
    pz = pop_z(train, n, POP_DAYS[source])

    tu = train[train.user_id.isin(users)]
    fav_u = brands = None
    if source in BETA:
        brands = ix.brands[rows]
        h = tu.assign(brand=lambda d: brands[d.item_id.values])
        h = h[h.brand >= 0]
        fav = h.groupby("user_id").brand.agg(lambda s: s.value_counts().idxmax())
        fav_u = np.array([fav.get(u, -2) for u in users])
    bought = tu.groupby("user_id").item_id.apply(lambda s: np.unique(s.values)) if source in EXCLUDE else None
    empty = np.array([], np.int64)

    out = {"eval_users": int(len(users)), "catalog": int(n), "test_window": how,
           "hist_len_mean": round(float(np.bincount(uidx, minlength=len(users)).mean()), 2)}
    per = {}
    for name, (kind, param) in WEIGHTINGS.items():
        w = weights(kind, param, ts, uidx, len(users))
        U = {m: user_vectors(models[m], X[m], items, uidx, w, len(users)) for m in ("text", "image")}
        if name == "mean":                                    # 단순 평균은 학습 · 서빙 코드(tower_vectors)와 같아야 한다
            out["mean_matches_tower_vectors"] = bool(all(np.allclose(U[m], U_check[m], atol=1e-4) for m in U))
        recs = []
        for s in range(0, len(users), CHUNK):
            sl = slice(s, s + CHUNK)
            S = sum(zrows(U[m][sl] @ V[m].T) for m in ("text", "image")) + GAMMA * pz[None, :]
            if fav_u is not None:
                S = S + BETA[source] * (brands[None, :] == fav_u[sl, None])
            if bought is not None:
                for r, u in enumerate(users[sl]):
                    S[r, bought.get(u, empty)] = -np.inf
            top = np.argpartition(-S, K, axis=1)[:, :K]
            recs.append(np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1), 1))
        rec = np.concatenate(recs)
        acc, per[name] = accuracy_at_k(rec, users, truth, ks=(K,))
        out[name] = {"map@12": round(acc["map@12"], 5), "recall@12": round(acc["recall@12"], 5),
                     "coverage@12": round(len(np.unique(rec)) / n, 4)}
    out["best"] = max((k for k in WEIGHTINGS if k != "mean"), key=lambda k: out[k]["map@12"])
    for name in WEIGHTINGS:                                   # 가중마다 − 단순 평균 (고르기 전 구간을 모두 보고)
        if name == "mean":
            continue
        out[name]["minus_mean"] = {}
        for mt in ("map@12", "recall@12"):
            b = paired_bootstrap(per["mean"][mt], per[name][mt])
            out[name]["minus_mean"][mt] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]],
                                           "rel": round(b["rel"], 3) if b["rel"] is not None else None}
    return out


def main():
    db = SessionLocal()
    ix = get_index(db)
    res = {f"{src}/{how}": evaluate(db, ix, src, how)
           for src in ("hm", "amazon") for how in ("log_lastout", "time_week")}
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
