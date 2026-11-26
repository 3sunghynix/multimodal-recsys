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
from app.services.ranker import CANDIDATES as C, FEATURES, candidate_features, log_price  # noqa: E402

CHUNK, NEG_PER_USER, SEED = 250, 50, 42
GAMMA = 1.0
EXCLUDE = {"amazon"}
DECAY = {"hm": 0.9}
EXTRA = 0          # 후보 경로 확장 실험 (11-26, 효과 없음 → 서빙 미적용): 규칙 상위 C개 + 추가 경로 최대 EXTRA개 (0 = 끔)
EXPAND = True      # False면 같은 후보 수(C + EXTRA)를 규칙 순위로만 채운다 (대조군)


def variant_groups(variants: np.ndarray) -> dict:
    """같은 옷(색상 변형 묶음 키) → 상품 순번 배열. 키 ""(없음)는 뺀다."""
    keys = np.asarray(variants, object)
    idx = np.flatnonzero(keys != "")
    if len(idx) == 0:
        return {}
    order = idx[np.argsort(keys[idx], kind="stable")]
    k = keys[order]
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    ends = np.r_[starts[1:], len(k)]
    return {k[a]: order[a:b] for a, b in zip(starts, ends)}


def expand_candidates(rule: np.ndarray, n_rule: int, n_total: int, bought: np.ndarray, bought_ts: np.ndarray,
                      variants: np.ndarray, groups: dict) -> np.ndarray:
    """후보 경로 확장 (규칙 점수 내림차순, 길이 n_total — 상품이 적으면 그만큼).

    = 규칙 상위 n_rule개 + 추가 경로 (최근 산 상품부터: 그 상품 → 같은 옷 다른 색) 최대 n_total − n_rule개,
      남는 자리는 규칙 다음 순위로 채운다 (후보 수가 같아 "후보를 늘린 효과"와 "경로 효과"를 나눠 본다).
    rule: 데이터셋 상품 전체 규칙 점수 (뺀 상품은 −inf, 후보가 되지 않는다). bought · bought_ts: 고객 구매 (순번, 시각).
    """
    n_total = int(min(n_total, np.isfinite(rule).sum()))
    if n_total == 0:
        return np.zeros(0, np.int64)
    top = np.argpartition(-rule, n_total - 1)[:n_total]
    top = top[np.argsort(-rule[top], kind="stable")]
    n_rule = min(n_rule, n_total)
    chosen = [int(j) for j in top[:n_rule]]
    seen = set(chosen)
    room = n_total - n_rule
    if room and len(bought):
        for i in np.asarray(bought, np.int64)[np.argsort(np.asarray(bought_ts, "datetime64[ns]"), kind="stable")[::-1]]:
            v = variants[i]
            for j in [int(i)] + (groups.get(v, np.zeros(0, np.int64)).tolist() if v else []):
                if room and j not in seen and np.isfinite(rule[j]):
                    seen.add(j)
                    chosen.append(j)
                    room -= 1
            if not room:
                break
    for j in top[n_rule:]:
        if not room:
            break
        if int(j) not in seen:
            seen.add(int(j))
            chosen.append(int(j))
            room -= 1
    cand = np.asarray(chosen, np.int64)
    return cand[np.argsort(-rule[cand], kind="stable")]


class Catalog:
    """데이터셋 하나의 상품 쪽 고정 값 (순번 = ids 안 위치). 서빙과 같은 배열 (색인 · DB 가격)."""

    def __init__(self, db, ix, source: str, tower_dir: str | Path | None = None):
        """tower_dir: UME 타워 폴더 ({text,image}/weights.npz). 기본 = 처음 배포한 {source}_test_pca (평가 기준선).
        재학습(recsys.retrain → ranker_train.retrain_ranker)은 후보 버전의 타워를 넘긴다 — 랭커 특성이 그 타워 점수라서."""
        import torch
        import torch.nn.functional as F
        rows = np.flatnonzero(ix.sources == source)
        self.source, self.ids, self.n = source, ix.ids[rows], len(rows)
        self.X = {"text": ix.text[rows], "image": ix.image[rows]}
        self.brands = ix.brands[rows]
        self.cats = ix.cats[rows] if ix.cats is not None else np.full(self.n, "", object)
        self.variants = ix.variants[rows] if ix.variants is not None else np.full(self.n, "", object)
        price = dict(db.execute(select(Item.id, Item.price).where(Item.source == source)).all())
        self.log_price = log_price([price.get(int(i)) for i in self.ids])
        base = Path(tower_dir) if tower_dir else Path(f"/data/models/ume/{source}_test_pca")
        cfg = json.loads((base / "text/config.json").read_text())
        self.p = {**DEFAULT, **{k: v for k, v in cfg.items() if k in DEFAULT}}
        self.groups = variant_groups(self.variants)
        self.models = {m: _tower_from_npz(dict(np.load(base / m / "weights.npz")), self.p, "cpu") for m in ("text", "image")}
        with torch.no_grad():
            self.V = {m: F.normalize(self.models[m]["item"](torch.from_numpy(self.X[m])), dim=1).numpy() for m in self.X}


def _pair_lookup(keys_sorted: np.ndarray, values: np.ndarray, query: np.ndarray, missing):
    out = np.full(query.shape, missing, dtype=float)
    if len(keys_sorted) == 0:
        return out
    pos = np.minimum(np.searchsorted(keys_sorted, query), len(keys_sorted) - 1)
    hit = keys_sorted[pos] == query
    out[hit] = values[pos[hit]]
    return out


class Users:
    """고객 쪽 상태: 이력(train)만으로 만든다."""

    def __init__(self, cat: Catalog, train: pd.DataFrame, users: np.ndarray):
        self.users, n_u = users, len(users)
        items, ts, uidx = history_rows(train, users, cat.p["max_hist"])
        decay = DECAY.get(cat.source)
        w = weights("pos", decay, ts, uidx, n_u) if decay else np.ones(len(items), np.float32)
        self.U = {m: user_vectors(cat.models[m], cat.X[m], items, uidx, w, n_u) for m in ("text", "image")}
        self.pz = pop_z(train, cat.n, POP_DAYS[cat.source])
        self.hist_len = np.bincount(uidx, minlength=n_u)
        self.t_ref = train.ts.max()                                # 서빙: 그 데이터셋 서비스 구매의 마지막 시각
        order = pd.Series(np.arange(n_u), index=users)
        tu = train[train.user_id.isin(users)]
        g = pd.DataFrame({"r": order[tu.user_id.values].values, "i": tu.item_id.values, "t": tu.ts.values}).groupby("r")
        self.bought = {r: (d.i.values, d.t.values) for r, d in g}
        b = cat.brands[tu.item_id.values]                          # 선호 브랜드 = 구매 중 최다 (서빙 bincount와 같은 동점 처리)
        rr = order[tu.user_id.values].values
        self.fav = np.full(n_u, -2)
        for r in np.unique(rr[b >= 0]):
            self.fav[r] = np.bincount(b[(rr == r) & (b >= 0)]).argmax()


def candidates(cat: Catalog, us: Users, sl: slice):
    """규칙 점수 상위 C개 (규칙 순) + 특성 (rows × C × F) — 특성은 서빙과 같은 app.services.ranker.candidate_features."""
    st = zrows(us.U["text"][sl] @ cat.V["text"].T)
    si = zrows(us.U["image"][sl] @ cat.V["image"].T)
    rule = st + si + GAMMA * us.pz[None, :]
    if cat.source in BETA:
        rule = rule + BETA[cat.source] * (cat.brands[None, :] == us.fav[sl, None])
    rows = np.arange(sl.start, min(sl.stop, len(us.users)))
    empty = (np.array([], np.int64), np.array([], "datetime64[ns]"))
    if cat.source in EXCLUDE:
        for r, u in enumerate(rows):
            rule[r, us.bought.get(u, empty)[0]] = -np.inf
    n_total = min(C + EXTRA, cat.n)
    if EXTRA and EXPAND:
        top = np.stack([expand_candidates(rule[r], C, n_total, *us.bought.get(u, empty), cat.variants, cat.groups)
                        for r, u in enumerate(rows)])
    else:
        top = np.argpartition(-rule, n_total - 1, axis=1)[:, :n_total]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(rule, top, 1), 1, kind="stable"), 1)
    F = np.stack([candidate_features(top[r], st[r], si[r], us.pz, cat.brands, cat.cats, cat.variants, cat.log_price,
                                     *us.bought.get(u, empty), us.t_ref, int(us.hist_len[u])) for r, u in enumerate(rows)])
    return top, F


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


def fit(Xtr: np.ndarray, ytr: np.ndarray, rng):
    """GBDT 학습 (검증 20%는 AUC용으로 떼어 둠). 이 데이터셋에서 값이 하나뿐인 특성은 뺀다
    (예: Amazon은 산 상품을 후보에서 빼므로 구매 특성이 전부 0 · 결측). 반환: (모델, 열 번호, 특성 이름, 검증 AUC, 검증 표본)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    keep = [j for j in range(len(FEATURES)) if np.unique(Xtr[~np.isnan(Xtr[:, j]), j]).size > 1]
    X = Xtr[:, keep]
    perm = rng.permutation(len(ytr))
    va, tr = perm[:len(ytr) // 5], perm[len(ytr) // 5:]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.1, random_state=SEED)
    model.fit(X[tr], ytr[tr])
    auc = float(roc_auc_score(ytr[va], model.predict_proba(X[va])[:, 1]))
    return model, keep, [FEATURES[j] for j in keep], auc, (X[va], ytr[va])


def evaluate(db, ix, source: str, how: str) -> dict:
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(SEED)
    cat = Catalog(db, ix, source)
    ev = service_purchases(db, cat.ids)
    train, users, truth = split(ev, how)

    Xtr, ytr, recall_tr, n_tr_users = build_training(cat, train, how, rng)
    model, keep, used, auc, (Xva, yva) = fit(Xtr, ytr, rng)
    sub = slice(0, min(len(yva), 40000))
    pi = permutation_importance(model, Xva[sub], yva[sub], scoring="roc_auc", n_repeats=3, random_state=SEED)
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
