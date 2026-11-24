"""C2 인기 보정 γ 재확인: 서비스 DB 구매 로그, 운영 가중치({source}_test_pca), 두 가지 분할.

실행: docker compose --profile trainer run --rm trainer python -m recsys.pop_eval   (결과: docs/project/추천모델설계.md "인기 보정 γ")

배경: variant_eval 첫 실행에서 "UME + γ2 · 인기"가 H&M MAP@12 0.0161로 UME 단독(0.0425)보다 크게 낮았다.
원인은 평가 쪽이었다 — 인기 창(학습 구간 마지막 7일)을 전체 이벤트의 마지막 시각으로 잡았는데, DB에 시연용 클릭 20건이
2026-11 시각으로 들어 있어 창이 시연 클릭 몇 건만 담았다. 서비스(_popularity_z)는 비 setup 구매의 마지막 시각 기준이라 영향 없음.
여기서는 서비스와 같은 정의로 다시 잰다: 구매만, 인기 창 = 학습 구간 마지막 구매 시각 기준 최근 N일(H&M 7일, Amazon 전체).

분할
- log_lastout: trainer 채택 판정과 같은 leave-last-out (최근 활동 고객 10%의 마지막 구매 하나)
- time_week:   마지막 7일 구매가 정답, 그 전이 이력 (Kaggle · 오프라인 실험과 같은 방식)
순위 = z(text) + z(image) + γ · 인기 z [+ β · 선호 브랜드, Amazon β = 2]. 재정렬 · 구매 상품 제외 없음 (variant_eval과 같음).
지표: MAP@12 · Recall@12 (γ=1 − γ=2 paired bootstrap 95% 구간), Coverage@12(추천된 서로 다른 상품 / 카탈로그), 상위 1% 인기 상품 비중(목록 안).
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
from recsys.retrain import _tower_from_npz, load_events, split_holdout  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.models import User  # noqa: E402
from recsys.ume import DEFAULT, build_sequences, tower_vectors  # noqa: E402

K, MAXU, HOLD, CHUNK, SEED = 12, 5000, 0.1, 500, 42
GAMMAS = (0.0, 0.5, 1.0, 2.0, 4.0)
POP_DAYS = {"hm": 7, "amazon": None}
BETA = {"amazon": 2.0}


def zrows(a):
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)


def pop_z(train: pd.DataFrame, n: int, days: int | None) -> np.ndarray:
    buy = train[train.type == "purchase"]
    if days is not None:
        buy = buy[buy.ts >= buy.ts.max() - pd.Timedelta(days=days)]
    c = np.log1p(np.bincount(buy.item_id.values, minlength=n).astype(np.float32))
    return (c - c.mean()) / (c.std() + 1e-6)


def split(ev: pd.DataFrame, how: str):
    if how == "log_lastout":
        train, test = split_holdout(ev, HOLD)
    else:
        cut = ev.ts.max() - pd.Timedelta(days=7)
        train, test = ev[ev.ts < cut], ev[ev.ts >= cut]
    targets = test.groupby("user_id").item_id.apply(set)
    seen = set(train.user_id)
    users = np.array([u for u in targets.index if u in seen], np.int64)
    if len(users) > MAXU:
        users = np.sort(np.random.default_rng(SEED).choice(users, MAXU, replace=False))
    truth = pd.DataFrame([(u, i) for u in users for i in targets[u]], columns=["user_id", "item_id"])
    return train, users, truth


def service_purchases(db, ids: np.ndarray) -> pd.DataFrame:
    """평가용 구매 로그: 비 setup 구매, 이 데이터셋 상품, 시뮬레이터 고객 제외 (서비스 인기 통계와 같은 범위,
    app/services/event_scope.py · #74). item_id는 ids 안의 순번으로 바꾼다. C2 · C3 · C4 평가가 함께 쓴다."""
    ev = load_events(db, 2**31 - 1)
    sim = np.array(db.scalars(select(User.id).where(User.source == "sim")).all(), np.int64)
    ev = ev[(ev.type == "purchase") & ~ev.is_setup & ev.item_id.isin(ids) & ~ev.user_id.isin(sim)].copy()
    ev["item_id"] = np.searchsorted(ids, ev.item_id.values)
    return ev


def evaluate(db, ix, source: str, how: str) -> dict:
    rows = np.flatnonzero(ix.sources == source)
    ids, n = ix.ids[rows], len(rows)
    ev = service_purchases(db, ids)
    train, users, truth = split(ev, how)

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
    top1 = pz >= np.quantile(pz, 0.99)

    brand_bonus = None
    if source in BETA:
        brands = ix.brands[rows]
        h = train[train.user_id.isin(users)].assign(brand=lambda d: brands[d.item_id.values])
        h = h[h.brand >= 0]
        fav = h.groupby("user_id").brand.agg(lambda s: s.value_counts().idxmax())
        fav_u = np.array([fav.get(u, -2) for u in users])
        brand_bonus = (BETA[source], brands, fav_u)

    recs = {g: [] for g in GAMMAS}
    for s in range(0, len(users), CHUNK):
        sl = slice(s, s + CHUNK)
        S = sum(zrows(vecs[m][0][sl] @ vecs[m][1].T) for m in ("text", "image"))
        if brand_bonus is not None:
            beta, brands, fav_u = brand_bonus
            S = S + beta * (brands[None, :] == fav_u[sl, None])
        for g in GAMMAS:
            sc = S + g * pz[None, :]
            top = np.argpartition(-sc, K, axis=1)[:, :K]
            recs[g].append(np.take_along_axis(top, np.argsort(-np.take_along_axis(sc, top, 1), 1), 1))

    out = {"eval_users": int(len(users)), "catalog": int(n), "train_events": int(len(train)),
           "test_window": how, "pop_days": POP_DAYS[source]}
    per = {}
    for g in GAMMAS:
        rec = np.concatenate(recs[g])
        acc, per[g] = accuracy_at_k(rec, users, truth, ks=(K,))
        out[f"gamma={g}"] = {"map@12": round(acc["map@12"], 5), "recall@12": round(acc["recall@12"], 5),
                             "coverage@12": round(len(np.unique(rec)) / n, 4),
                             "top1pct_share": round(float(top1[rec].mean()), 4)}
    # γ=1 − γ=2 (운영값) 고객 단위 paired bootstrap
    out["gamma1_minus_gamma2"] = {}
    for mt in ("map@12", "recall@12"):
        b = paired_bootstrap(per[2.0][mt], per[1.0][mt])
        out["gamma1_minus_gamma2"][mt] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]],
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
