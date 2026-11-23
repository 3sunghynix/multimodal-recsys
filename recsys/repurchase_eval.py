"""C3 재구매 정책: 이미 산 상품을 추천에서 뺄지, 남길지 (서비스 DB 구매 로그, 운영 가중치, pop_eval과 같은 두 분할).

실행: docker compose --profile trainer run --rm trainer python -m recsys.repurchase_eval   (결과: docs/project/추천모델설계.md "재구매 정책")

순위 = 운영 점수 z(text) + z(image) + γ · 인기 z [+ β · 선호 브랜드] (γ = 1, Amazon β = 2), 재정렬 없음 — pop_eval과 같다.
정책 (고객이 학습 구간에 산 상품 기준)
- keep:        그대로 둔다 (지금 서비스)
- exclude:     산 상품은 전부 뺀다
- exclude_30d: 학습 구간 마지막 구매 시각 기준 최근 30일 안에 산 상품만 뺀다 (방금 산 것만 숨김)
같이 재는 것: 정답 구매 중 학습 구간에 이미 산 상품의 비율(재구매율), 정책별 MAP@12 · Recall@12 · Coverage@12,
              keep 목록 안에 이미 산 상품이 몇 개 들어 있는지, exclude − keep paired bootstrap 95% 구간.
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
from recsys.pop_eval import BETA, CHUNK, K, POP_DAYS, pop_z, split, zrows  # noqa: E402
from recsys.retrain import _tower_from_npz, load_events  # noqa: E402
from recsys.ume import DEFAULT, build_sequences, tower_vectors  # noqa: E402

GAMMA = 1.0
POLICIES = ("keep", "exclude", "exclude_30d")
RECENT_DAYS = 30


def evaluate(db, ix, source: str, how: str) -> dict:
    rows = np.flatnonzero(ix.sources == source)
    ids, n = ix.ids[rows], len(rows)
    ev = load_events(db, 2**31 - 1)
    ev = ev[(ev.type == "purchase") & ~ev.is_setup & ev.item_id.isin(ids)].copy()
    ev["item_id"] = np.searchsorted(ids, ev.item_id.values)
    train, users, truth = split(ev, how)

    # 고객별 학습 구간 구매 (전체 · 최근 30일)
    tu = train[train.user_id.isin(users)]
    recent_cut = train.ts.max() - pd.Timedelta(days=RECENT_DAYS)
    bought = {"exclude": tu.groupby("user_id").item_id.apply(lambda s: np.unique(s.values)),
              "exclude_30d": tu[tu.ts >= recent_cut].groupby("user_id").item_id.apply(lambda s: np.unique(s.values))}
    empty = np.array([], np.int64)
    rep = truth.merge(tu[["user_id", "item_id"]].drop_duplicates(), how="left", indicator=True)
    repurchase_rate = float((rep["_merge"] == "both").mean())

    base = Path(f"/data/models/ume/{source}_test_pca")
    cfg = json.loads((base / "text/config.json").read_text())
    p = {**DEFAULT, **{k: v for k, v in cfg.items() if k in DEFAULT}}
    seq = build_sequences(train, p["max_hist"])
    X = {"text": ix.text[rows], "image": ix.image[rows]}
    vecs = {}
    for m in ("text", "image"):
        model = _tower_from_npz(dict(np.load(base / m / "weights.npz")), p, "cpu")
        vecs[m] = tower_vectors(model, X[m], seq, users, np.arange(n), p["max_hist"], "cpu")
    pz = pop_z(train, n, POP_DAYS[source])

    fav_u = brands = None
    if source in BETA:
        brands = ix.brands[rows]
        h = tu.assign(brand=lambda d: brands[d.item_id.values])
        h = h[h.brand >= 0]
        fav = h.groupby("user_id").brand.agg(lambda s: s.value_counts().idxmax())
        fav_u = np.array([fav.get(u, -2) for u in users])

    recs = {pol: [] for pol in POLICIES}
    bought_in_keep = []
    for s in range(0, len(users), CHUNK):
        sl = slice(s, s + CHUNK)
        S = sum(zrows(vecs[m][0][sl] @ vecs[m][1].T) for m in ("text", "image")) + GAMMA * pz[None, :]
        if fav_u is not None:
            S = S + BETA[source] * (brands[None, :] == fav_u[sl, None])
        for pol in POLICIES:
            sc = S.copy()
            if pol != "keep":
                for r, u in enumerate(users[sl]):
                    sc[r, bought[pol].get(u, empty)] = -np.inf
            top = np.argpartition(-sc, K, axis=1)[:, :K]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(sc, top, 1), 1), 1)
            recs[pol].append(top)
            if pol == "keep":
                bought_in_keep += [np.isin(top[r], bought["exclude"].get(u, empty)).sum() for r, u in enumerate(users[sl])]

    out = {"eval_users": int(len(users)), "catalog": int(n), "test_window": how,
           "repurchase_rate": round(repurchase_rate, 4),
           "keep_list_bought_mean": round(float(np.mean(bought_in_keep)), 3),
           "keep_list_any_bought": round(float(np.mean(np.array(bought_in_keep) > 0)), 4)}
    per = {}
    for pol in POLICIES:
        rec = np.concatenate(recs[pol])
        acc, per[pol] = accuracy_at_k(rec, users, truth, ks=(K,))
        out[pol] = {"map@12": round(acc["map@12"], 5), "recall@12": round(acc["recall@12"], 5),
                    "coverage@12": round(len(np.unique(rec)) / n, 4)}
    for pol in ("exclude", "exclude_30d"):
        out[f"{pol}_minus_keep"] = {}
        for mt in ("map@12", "recall@12"):
            b = paired_bootstrap(per["keep"][mt], per[pol][mt])
            out[f"{pol}_minus_keep"][mt] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]],
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
