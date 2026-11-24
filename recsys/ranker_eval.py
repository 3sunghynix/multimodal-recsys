"""B1 2단계 학습형 랭커 (오프라인): UME 규칙 점수 상위 C개 후보를 GBDT로 다시 순위 매기면 MAP@12가 오르나.

실행: docker compose --profile trainer run --rm trainer python -m recsys.ranker_eval   (결과: docs/project/추천모델설계.md "학습형 랭커")

지금 서비스(규칙): 점수 = z(text) + z(image) + γ · 인기 z [+ β · 선호 브랜드], Amazon은 산 상품 제외, H&M 고객 벡터는 최근성 λ = 0.9.
랭커: 그 규칙 점수 상위 C(300)개를 후보로 두고, 아래 특성으로 학습한 GBDT(scikit-learn HistGradientBoosting, 새 의존성 없음) 확률 순으로 12개.

분할 (pop_eval과 같은 두 방식, 서비스 범위 구매 로그 = pop_eval.service_purchases)
  평가:   이력 = 평가 구간 전, 정답 = 평가 구간 (마지막 구매 / 마지막 7일)
  학습:   평가 구간 전 이력을 같은 방식으로 한 번 더 나눈다 — 그 안의 이력으로 후보 · 특성, 그 안의 정답 구간으로 라벨.
          평가 구간 정보는 학습에 들어가지 않는다.
특성 (고객 · 후보 상품 쌍)
  s_text · s_image (고객별 z), rule_rank (후보 안 규칙 순위), pop_z, brand_match, bought_n · days_since_bought (같은 상품 구매),
  variant_bought (같은 옷 다른 색 구매, H&M), cat_share (고객 이력 중 같은 소분류 비율), price_gap (log 가격 − 고객 평균 log 가격),
  hist_len (고객 이력 수)
지표: MAP@12 · Recall@12 · Coverage@12, 후보 재현율(정답이 후보 C개 안에 든 비율), 랭커 − 규칙 paired bootstrap 95% 구간,
      특성 중요도 (permutation, 학습 검증 표본 AUC 감소).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.models import Item  # noqa: E402
from app.services.catalog_index import get_index  # noqa: E402
from recsys.eval.metrics import accuracy_at_k  # noqa: E402
from recsys.eval.stats import paired_bootstrap  # noqa: E402
from recsys.pop_eval import BETA, K, POP_DAYS, pop_z, service_purchases, split, zrows  # noqa: E402
from recsys.recency_eval import history_rows, user_vectors, weights  # noqa: E402
from recsys.retrain import _tower_from_npz  # noqa: E402
from recsys.ume import DEFAULT  # noqa: E402

C, CHUNK, NEG_PER_USER, SEED = 300, 250, 50, 42
GAMMA = 1.0
EXCLUDE = {"amazon"}
DECAY = {"hm": 0.9}
FEATURES = ["s_text", "s_image", "rule_rank", "pop_z", "brand_match", "bought_n", "days_since_bought",
            "variant_bought", "cat_share", "price_gap", "hist_len"]


def _codes(values: np.ndarray) -> np.ndarray:
    """문자열 배열 → 정수 코드 ("" · None은 −1)."""
    v = pd.Series(values).fillna("").astype(str)
    codes, _ = pd.factorize(v)
    codes = codes.astype(np.int64)
    codes[v.values == ""] = -1
    return codes


class Catalog:
    """데이터셋 하나의 상품 쪽 고정 값 (순번 = ids 안 위치)."""

    def __init__(self, db, ix, source: str):
        rows = np.flatnonzero(ix.sources == source)
        self.source, self.ids, self.n = source, ix.ids[rows], len(rows)
        self.X = {"text": ix.text[rows], "image": ix.image[rows]}
        self.brands = ix.brands[rows]
        self.cats = _codes(ix.cats[rows]) if ix.cats is not None else np.full(self.n, -1)
        self.variants = _codes(ix.variants[rows]) if ix.variants is not None else np.full(self.n, -1)
        price = dict(db.execute(select(Item.id, Item.price).where(Item.source == source)).all())
        p = np.array([price.get(int(i)) for i in self.ids], dtype=float)
        self.log_price = np.log1p(np.where(p > 0, p, np.nan))
        base = Path(f"/data/models/ume/{source}_test_pca")
        cfg = json.loads((base / "text/config.json").read_text())
        self.p = {**DEFAULT, **{k: v for k, v in cfg.items() if k in DEFAULT}}
        self.models = {m: _tower_from_npz(dict(np.load(base / m / "weights.npz")), self.p, "cpu") for m in ("text", "image")}
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            self.V = {m: F.normalize(self.models[m]["item"](torch.from_numpy(self.X[m])), dim=1).numpy() for m in self.X}


def _pair_lookup(keys_sorted: np.ndarray, values: np.ndarray, query: np.ndarray, missing):
    pos = np.searchsorted(keys_sorted, query)
    pos = np.minimum(pos, len(keys_sorted) - 1) if len(keys_sorted) else pos
    hit = (len(keys_sorted) > 0) & (keys_sorted[pos] == query) if len(keys_sorted) else np.zeros(query.shape, bool)
    out = np.full(query.shape, missing, dtype=float)
    out[hit] = values[pos[hit]]
    return out


class Users:
    """고객 쪽 상태: 이력(train)만으로 만든다."""

    def __init__(self, cat: Catalog, train: pd.DataFrame, users: np.ndarray):
        self.users, n_u, n = users, len(users), cat.n
        items, ts, uidx = history_rows(train, users, cat.p["max_hist"])
        decay = DECAY.get(cat.source)
        w = weights("pos", decay, ts, uidx, n_u) if decay else np.ones(len(items), np.float32)
        self.U = {m: user_vectors(cat.models[m], cat.X[m], items, uidx, w, n_u) for m in ("text", "image")}
        self.pz = pop_z(train, n, POP_DAYS[cat.source])
        self.hist_len = np.bincount(uidx, minlength=n_u).astype(float)
        order = pd.Series(np.arange(n_u), index=users)
        tu = train[train.user_id.isin(users)]
        ur = order[tu.user_id.values].values
        t_ref = train.ts.max()
        # 같은 상품 구매 수 · 마지막 구매 후 경과일
        g = pd.DataFrame({"k": ur * n + tu.item_id.values, "ts": tu.ts.values}).groupby("k").ts.agg(["size", "max"])
        self.bought_keys, self.bought_n = g.index.values, g["size"].values.astype(float)
        self.bought_days = ((t_ref - g["max"]).dt.total_seconds() / 86400).values
        # 같은 옷(색상 변형) 구매
        v = cat.variants[tu.item_id.values]
        vk = np.unique(ur[v >= 0] * (cat.variants.max() + 1) + v[v >= 0])
        self.variant_keys, self.nv = vk, cat.variants.max() + 1
        # 소분류 비율
        nc = max(cat.cats.max() + 1, 1)
        cc = cat.cats[tu.item_id.values]
        M = np.zeros((n_u, nc), np.float32)
        np.add.at(M, (ur[cc >= 0], cc[cc >= 0]), 1.0)
        self.cat_share = M / np.maximum(M.sum(1, keepdims=True), 1.0)
        # 고객 평균 log 가격
        lp = cat.log_price[tu.item_id.values]
        s = pd.Series(lp).groupby(ur).mean()
        self.mean_lp = np.full(n_u, np.nan)
        self.mean_lp[s.index.values] = s.values
        # 선호 브랜드 · 산 상품 (Amazon 제외용)
        b = cat.brands[tu.item_id.values]
        fav = pd.Series(b[b >= 0]).groupby(ur[b >= 0]).agg(lambda x: x.value_counts().idxmax())
        self.fav = np.full(n_u, -2)
        self.fav[fav.index.values] = fav.values
        self.bought_by_user = pd.Series(tu.item_id.values).groupby(ur).apply(lambda x: np.unique(x.values))


def candidates(cat: Catalog, us: Users, sl: slice):
    """규칙 점수 상위 C개 (규칙 순) + 특성 행렬 (rows × C × F)."""
    st = zrows(us.U["text"][sl] @ cat.V["text"].T)
    si = zrows(us.U["image"][sl] @ cat.V["image"].T)
    bm = (cat.brands[None, :] == us.fav[sl, None]).astype(np.float32)
    rule = st + si + GAMMA * us.pz[None, :] + BETA.get(cat.source, 0.0) * bm
    rows = np.arange(sl.start, min(sl.stop, len(us.users)))
    if cat.source in EXCLUDE:
        empty = np.array([], np.int64)
        for r, u in enumerate(rows):
            rule[r, us.bought_by_user.get(u, empty)] = -np.inf
    top = np.argpartition(-rule, C, axis=1)[:, :C]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(rule, top, 1), 1, kind="stable"), 1)
    R = rows[:, None]
    f = {
        "s_text": np.take_along_axis(st, top, 1), "s_image": np.take_along_axis(si, top, 1),
        "rule_rank": np.broadcast_to(np.arange(C, dtype=float), top.shape),
        "pop_z": us.pz[top], "brand_match": np.take_along_axis(bm, top, 1),
        "bought_n": _pair_lookup(us.bought_keys, us.bought_n, R * cat.n + top, 0.0),
        "days_since_bought": _pair_lookup(us.bought_keys, us.bought_days, R * cat.n + top, np.nan),
        "variant_bought": np.where(cat.variants[top] >= 0,
                                   _pair_lookup(us.variant_keys, np.ones(len(us.variant_keys)), R * us.nv + cat.variants[top], 0.0), 0.0),
        "cat_share": np.where(cat.cats[top] >= 0, us.cat_share[R, np.maximum(cat.cats[top], 0)], 0.0),
        "price_gap": cat.log_price[top] - us.mean_lp[R],
        "hist_len": np.broadcast_to(us.hist_len[R], top.shape),
    }
    return top, np.stack([np.asarray(f[k], dtype=np.float32) for k in FEATURES], axis=-1)


def label_matrix(top: np.ndarray, users: np.ndarray, rows: np.ndarray, truth: pd.DataFrame, n: int) -> np.ndarray:
    keys = np.sort(np.unique(pd.Series(np.arange(len(users)), index=users)[truth.user_id.values].values * n + truth.item_id.values))
    return _pair_lookup(keys, np.ones(len(keys)), rows[:, None] * n + top, 0.0) > 0


def build_training(cat: Catalog, train: pd.DataFrame, how: str, rng):
    """학습용: 평가 구간 전 이력을 같은 방식으로 한 번 더 나눠 후보 · 라벨. 양성 전부 + 고객당 음성 NEG_PER_USER개."""
    tr2, users, truth = split(train, how)
    us = Users(cat, tr2, users)
    Xs, ys, cand_recall = [], [], []
    for s in range(0, len(users), CHUNK):
        sl = slice(s, s + CHUNK)
        top, F = candidates(cat, us, sl)
        rows = np.arange(sl.start, min(sl.stop, len(users)))
        y = label_matrix(top, users, rows, truth, cat.n)
        cand_recall.append(y.sum(1))
        for r in range(len(rows)):
            pos = np.flatnonzero(y[r])
            neg = np.flatnonzero(~y[r])
            neg = rng.choice(neg, min(NEG_PER_USER, len(neg)), replace=False)
            idx = np.concatenate([pos, neg])
            Xs.append(F[r, idx])
            ys.append(y[r, idx])
    n_truth = truth.groupby("user_id").size().reindex(users).fillna(0).values
    recall = float(np.concatenate(cand_recall).sum() / max(n_truth.sum(), 1))
    return np.concatenate(Xs), np.concatenate(ys).astype(int), recall, len(users)


def evaluate(db, ix, source: str, how: str) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    cat = Catalog(db, ix, source)
    ev = service_purchases(db, cat.ids)
    train, users, truth = split(ev, how)

    Xtr, ytr, recall_tr, n_tr_users = build_training(cat, train, how, rng)
    # 이 데이터셋에서 값이 하나뿐인 특성은 뺀다 (예: Amazon은 산 상품을 후보에서 빼므로 구매 특성이 전부 0 · 결측)
    keep = [j for j in range(len(FEATURES)) if np.unique(Xtr[~np.isnan(Xtr[:, j]), j]).size > 1]
    used = [FEATURES[j] for j in keep]
    Xtr = Xtr[:, keep]
    perm = rng.permutation(len(ytr))
    n_val = len(ytr) // 5
    va, tr = perm[:n_val], perm[n_val:]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.1, random_state=SEED)
    model.fit(Xtr[tr], ytr[tr])
    auc = float(roc_auc_score(ytr[va], model.predict_proba(Xtr[va])[:, 1]))
    sub = va[:min(len(va), 40000)]
    pi = permutation_importance(model, Xtr[sub], ytr[sub], scoring="roc_auc", n_repeats=3, random_state=SEED)
    importance = {k: round(float(v), 4) for k, v in sorted(zip(used, pi.importances_mean), key=lambda t: -t[1])}

    us = Users(cat, train, users)
    rec_rule, rec_rank, hits = [], [], 0
    for s in range(0, len(users), CHUNK):
        sl = slice(s, s + CHUNK)
        top, F = candidates(cat, us, sl)
        rows = np.arange(sl.start, min(sl.stop, len(users)))
        hits += label_matrix(top, users, rows, truth, cat.n).sum()
        prob = model.predict_proba(F.reshape(-1, len(FEATURES))[:, keep])[:, 1].reshape(top.shape)
        rec_rule.append(top[:, :K])
        rec_rank.append(np.take_along_axis(top, np.argsort(-prob, 1, kind="stable")[:, :K], 1))
    out = {"eval_users": int(len(users)), "catalog": int(cat.n), "test_window": how,
           "train_users": int(n_tr_users), "train_rows": int(len(ytr)), "train_pos": int(ytr.sum()),
           "candidate_recall@300_train": round(recall_tr, 4),
           "candidate_recall@300_test": round(float(hits / max(len(truth), 1)), 4),
           "val_auc": round(auc, 4), "iterations": int(model.n_iter_), "features_dropped": [f for f in FEATURES if f not in used],
           "importance": importance}
    per = {}
    for name, rec in (("rule", np.concatenate(rec_rule)), ("ranker", np.concatenate(rec_rank))):
        acc, per[name] = accuracy_at_k(rec, users, truth, ks=(K,))
        out[name] = {"map@12": round(acc["map@12"], 5), "recall@12": round(acc["recall@12"], 5),
                     "coverage@12": round(len(np.unique(rec)) / cat.n, 4)}
    out["ranker_minus_rule"] = {}
    for mt in ("map@12", "recall@12"):
        b = paired_bootstrap(per["rule"][mt], per["ranker"][mt])
        out["ranker_minus_rule"][mt] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]],
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
