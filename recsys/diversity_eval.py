"""C5 다양성 재정렬: 지금 서비스(소분류 30% 상한)와 MMR을 같은 조건에서 — 정확도를 얼마나 잃고 다양성을 얼마나 얻나.

실행: docker compose --profile trainer run --rm trainer python -m recsys.diversity_eval   (결과: docs/project/추천모델설계.md "다양성 재정렬")

조건: 서비스 DB 구매 로그(pop_eval.service_purchases), 두 분할(마지막 구매 · 마지막 7일), 데이터셋마다 고객 5,000명 (C2 ~ B2와 같음).
점수 = 서비스 규칙 (z(text) + z(image) + γ · 인기 z [+ β · 선호 브랜드], H&M 최근성 λ = 0.9, Amazon 산 상품 제외).
모든 방법에 같은 색상 변형 묶기(같은 옷 1개)를 적용하고, 상위 K × POOL개 후보 안에서 K개를 고른다 (서빙 pool = 5).

방법
- none:    점수 순 (색상 묶기만)
- cap:     지금 서비스 (services/rerank.diversify: 소분류 30% 상한, 빈 자리는 개인화 점수 순)
- mmr_λ:   Maximal Marginal Relevance (Carbonell & Goldstein, 1998). 매 자리 argmax λ · rel − (1 − λ) · max_sim(이미 고른 것).
           rel = 후보 안 점수 min-max 정규화, sim = 상품 콘텐츠 임베딩(텍스트 · 이미지 평균) 코사인.
- mmr_λ+cap: MMR에 소분류 상한도 함께 (상한에 걸린 후보는 건너뜀)
- mmrp_λ+cap: 위와 같지만 rel = 개인화 점수(인기 · 브랜드 보정 전) — 지금 서비스의 빈 자리 채우기(#52)와 같은 이유로,
           인기 보정이 큰 상품이 고객마다 똑같이 들어와 Coverage가 줄지 않게. 후보(pool)는 그대로 최종 점수 상위
지표: MAP@12 · Recall@12 (정확도), ILD@12 = 목록 안 상품 쌍 평균 (1 − 코사인), 소분류 수 · 최다 소분류 비율 (목록 다양성),
      Coverage@12 = 추천된 서로 다른 상품 / 카탈로그 (고객 전체 다양성). 각 방법 − cap paired bootstrap (MAP).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/app")
from app.db import SessionLocal  # noqa: E402
from app.services.catalog_index import get_index  # noqa: E402
from app.services.rerank import DEFAULT as RR, cap as cat_cap, diversify  # noqa: E402
from recsys.eval.metrics import accuracy_at_k  # noqa: E402
from recsys.eval.stats import paired_bootstrap  # noqa: E402
from recsys.pop_eval import BETA, K, service_purchases, split, zrows  # noqa: E402
from recsys.ranker_eval import EXCLUDE, GAMMA, Catalog, Users  # noqa: E402

CHUNK = 250
LAMBDAS = (0.9, 0.8, 0.7)


def content_vectors(ix, rows) -> np.ndarray:
    v = ix.text[rows].astype(np.float32) + ix.image[rows].astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def mmr(pool: np.ndarray, rel: np.ndarray, E: np.ndarray, k: int, lam: float, cats=None, variants=None,
        cat_limit: int | None = None) -> np.ndarray:
    """pool: 점수 내림차순 후보, rel: pool과 같은 순서의 점수. 같은 옷은 1개, cat_limit이 있으면 소분류 상한."""
    r = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
    S = E[pool] @ E[pool].T
    picked, maxsim = [], np.full(len(pool), -1.0)
    alive = np.ones(len(pool), bool)
    n_cat, seen_var = {}, set()
    while len(picked) < k and alive.any():
        gain = np.where(alive, lam * r - (1 - lam) * np.maximum(maxsim, 0), -np.inf) if picked else np.where(alive, r, -np.inf)
        j = int(np.argmax(gain))
        alive[j] = False
        p = pool[j]
        v = variants[p] if variants is not None else ""
        c = cats[p] if cats is not None else ""
        if v and v in seen_var:
            continue
        if cat_limit is not None and c and n_cat.get(c, 0) >= cat_limit:
            continue
        picked.append(j)
        if v:
            seen_var.add(v)
        if c:
            n_cat[c] = n_cat.get(c, 0) + 1
        maxsim = np.maximum(maxsim, S[j])
    if len(picked) < k:                                     # 상한 · 묶기로 모자라면 남은 후보를 점수 순으로
        rest = [j for j in range(len(pool)) if j not in set(picked)]
        picked += rest[:k - len(picked)]
    return pool[np.asarray(picked, int)]


def list_stats(rec: np.ndarray, E: np.ndarray, cats: np.ndarray) -> dict:
    ild, ncat, top = [], [], []
    iu = np.triu_indices(rec.shape[1], 1)
    for r in rec:
        S = E[r] @ E[r].T
        ild.append(float((1 - S[iu]).mean()))
        c = [x for x in cats[r] if x]
        _, cnt = np.unique(np.asarray(c, object), return_counts=True) if c else (None, np.array([0]))
        ncat.append(len(cnt) if c else 0)
        top.append(cnt.max() / len(r) if c else 0.0)
    return {"ild@12": round(float(np.mean(ild)), 4), "categories@12": round(float(np.mean(ncat)), 2),
            "max_category_share@12": round(float(np.mean(top)), 3)}


def evaluate(db, ix, source: str, how: str) -> dict:
    cat = Catalog(db, ix, source)
    rows = np.flatnonzero(ix.sources == source)
    E = content_vectors(ix, rows)
    ev = service_purchases(db, cat.ids)
    train, users, truth = split(ev, how)
    us = Users(cat, train, users)
    pool_n = K * int(RR["pool"])
    limit = cat_cap(K, RR["max_category_share"], RR["min_cap"])
    methods = (["none", "cap"] + [f"mmr_{l}" for l in LAMBDAS] + [f"mmr_{l}+cap" for l in LAMBDAS]
               + [f"mmrp_{l}+cap" for l in LAMBDAS])
    recs = {m: [] for m in methods}
    empty = (np.array([], np.int64), None)
    var = cat.variants if source == "hm" else None
    for s in range(0, len(users), CHUNK):
        sl = slice(s, s + CHUNK)
        rr = np.arange(sl.start, min(sl.stop, len(users)))
        st = zrows(us.U["text"][sl] @ cat.V["text"].T)
        si = zrows(us.U["image"][sl] @ cat.V["image"].T)
        pers = st + si
        score = pers + GAMMA * us.pz[None, :]
        if source in BETA:
            score = score + BETA[source] * (cat.brands[None, :] == us.fav[sl, None])
        if source in EXCLUDE:
            for r, u in enumerate(rr):
                score[r, us.bought.get(u, empty)[0]] = -np.inf
        top = np.argpartition(-score, pool_n, axis=1)[:, :pool_n]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(score, top, 1), 1, kind="stable"), 1)
        for r in range(len(rr)):
            pool = top[r]
            rel = score[r, pool]
            recs["none"].append(diversify(pool, K, None, None, {"max_category_share": None}, variants=var))
            recs["cap"].append(diversify(pool, K, cat.cats, cat.brands, None, fill_scores=pers[r], variants=var))
            for lam in LAMBDAS:
                recs[f"mmr_{lam}"].append(mmr(pool, rel, E, K, lam, variants=var))
                recs[f"mmr_{lam}+cap"].append(mmr(pool, rel, E, K, lam, cats=cat.cats, variants=var, cat_limit=limit))
                recs[f"mmrp_{lam}+cap"].append(mmr(pool, pers[r, pool], E, K, lam, cats=cat.cats, variants=var, cat_limit=limit))
    out = {"eval_users": int(len(users)), "catalog": int(cat.n), "test_window": how, "pool": pool_n, "category_cap": limit}
    per = {}
    for m in methods:
        rec = np.stack(recs[m])
        acc, per[m] = accuracy_at_k(rec, users, truth, ks=(K,))
        out[m] = {"map@12": round(acc["map@12"], 5), "recall@12": round(acc["recall@12"], 5),
                  "coverage@12": round(len(np.unique(rec)) / cat.n, 4), **list_stats(rec, E, cat.cats)}
    out["minus_cap_map@12"] = {}
    for m in methods:
        if m == "cap":
            continue
        b = paired_bootstrap(per["cap"]["map@12"], per[m]["map@12"])
        out["minus_cap_map@12"][m] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]],
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
