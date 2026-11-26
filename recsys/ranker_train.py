"""B1 랭커 학습 잡: 서비스 구매 로그 전체로 H&M 랭커를 학습해 저장한다 (서빙은 모델 버전 params "ranker"로 켬).

처음 한 번 (수동):
  docker compose --profile trainer run --rm trainer python -m recsys.ranker_train [--source hm] [--out /data/models/ranker]
  → /data/models/ranker/{source}/model.joblib · meta.json
  docker compose exec api python -m scripts.register_ume --version v1-g1-rk --ranker-root /data/models/ranker
재학습 (자동): 기준 버전에 "ranker"가 있으면 recsys.retrain이 타워를 이어 학습한 뒤 retrain_ranker로 랭커도 다시 학습 · 비교해
  후보 버전에 넣는다 (retrain_{시각}/ranker/{source}). 운영 중인 랭커 폴더는 덮어쓰지 않는다.

학습: 서비스 범위 구매(pop_eval.service_purchases — 비 setup, 시뮬레이터 고객 제외)의 마지막 7일을 라벨, 그 전을 이력으로
후보 300개 · 특성 (ranker_eval과 같은 함수). 타워 = tower_dir (기본: 처음 배포한 {source}_test_pca).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/app")
from app.db import SessionLocal  # noqa: E402
from app.services.catalog_index import get_index  # noqa: E402
from app.services.ranker import CANDIDATES, FEATURES, load_ranker, predict, ranker_problem, save_ranker  # noqa: E402
from recsys.eval.metrics import accuracy_at_k  # noqa: E402
from recsys.pop_eval import K, service_purchases, split  # noqa: E402
from recsys.ranker_eval import (CHUNK, DECAY, GAMMA, NEG_PER_USER, SEED, Catalog, Users, build_training,  # noqa: E402
                                candidates, fit)

MIN_LABEL_USERS = 50        # 라벨 구간(마지막 7일)에 구매한 고객이 이보다 적으면 랭커를 다시 학습하지 않는다


def train(db, source: str, out: Path, tower_dir=None, ix=None, cat=None, ev=None) -> dict:
    """전체 구매로 학습 → out/{source}에 저장. cat · ev를 넘기면 다시 읽지 않는다."""
    rng = np.random.default_rng(SEED)
    cat = cat or Catalog(db, ix if ix is not None else get_index(db), source, tower_dir)
    ev = service_purchases(db, cat.ids) if ev is None else ev
    X, y, recall, n_users = build_training(cat, ev, "time_week", rng)
    model, keep, used, auc, _ = fit(X, y, rng)
    meta = {"source": source, "trained_at": datetime.now(timezone.utc).isoformat(), "label_window": "last 7 days",
            "label_end": str(ev.ts.max()), "train_users": int(n_users), "train_rows": int(len(y)), "train_pos": int(y.sum()),
            "candidate_recall": round(recall, 4), "val_auc": round(auc, 4), "candidates": CANDIDATES, "gamma": GAMMA,
            "recency_decay": DECAY.get(source), "neg_per_user": NEG_PER_USER, "model": type(model).__name__,
            "iterations": int(model.n_iter_), "tower_dir": str(tower_dir) if tower_dir else None}
    save_ranker(str(out / source), model, used, meta)
    return {**meta, "features": used, "path": str(out / source)}


def retrain_ranker(db, ix, source: str, tower_dir, base_ranker: str | None, out: Path, tolerance: float = 0.02) -> dict:
    """후보 타워(tower_dir) 기준으로 랭커를 다시 학습하고, 기준 랭커와 같은 평가 구간에서 비교한다.

    평가: 서비스 구매의 마지막 7일을 정답으로 떼고(고객 최대 5,000명), 그 전 이력으로 새 랭커를 학습(ranker_eval과 같은 중첩 분할) →
          같은 후보 300개(후보 타워 · 규칙 점수)를 새 랭커 / 기준 랭커 / 규칙 순으로 매겨 MAP@12 비교.
          두 랭커가 같은 타워 · 같은 후보에서 비교되므로, 타워가 바뀌어도 "이 타워에는 어느 랭커가 나은가"를 잰다.
    채택: 새 MAP@12 ≥ 비교 대상 × (1 − tolerance) (UME 재학습과 같은 허용 폭). 비교 대상 = 기준 랭커, 단
          기준 랭커가 평가 구간을 이미 학습에 썼으면(meta label_end ≥ 평가 구간 시작 — 새 구매 없이 다시 돌린 경우) 정답을 본 모델이라
          비교가 공정하지 않으므로 규칙 점수와 비교한다 (11-26 실측: 같은 로그로 기준 0.0560 vs 새 0.0495 — 기준은 평가 구간을 라벨로 학습).
          기준 랭커가 없거나 읽을 수 없어도 규칙과 비교.
    채택되면 평가 구간까지 포함한 전체 구매로 다시 학습해 out/{source}에 저장한다 (수동 ranker_train과 같은 방식).
    """
    rng = np.random.default_rng(SEED)
    cat = Catalog(db, ix, source, tower_dir)
    ev = service_purchases(db, cat.ids)
    report = {"status": "skipped", "tower_dir": str(tower_dir), "base_ranker": base_ranker}
    if len(ev) == 0:
        return {**report, "reason": "서비스 범위 구매 없음"}
    train_ev, users, truth = split(ev, "time_week")
    if len(users) < MIN_LABEL_USERS:
        return {**report, "reason": f"평가 구간 고객 {len(users)} < {MIN_LABEL_USERS}"}
    X, y, _, n_train_users = build_training(cat, train_ev, "time_week", rng)
    if n_train_users < MIN_LABEL_USERS or y.sum() == 0 or y.sum() == len(y):
        return {**report, "reason": f"학습 라벨 부족 (고객 {n_train_users}, 양성 {int(y.sum())})"}
    model, keep, _, auc, _ = fit(X, y, rng)
    base_ok = bool(base_ranker) and ranker_problem(base_ranker) is None
    cut = ev.ts.max() - pd.Timedelta(days=7)                       # pop_eval.split time_week와 같은 평가 구간 시작
    base_end = load_ranker(base_ranker)[2].get("label_end") if base_ok else None
    base_seen_eval = base_end is not None and pd.Timestamp(base_end) >= cut

    us = Users(cat, train_ev, users)
    recs = {"rule": [], "new": [], "base": []}
    for s in range(0, len(users), CHUNK):
        sl = slice(s, s + CHUNK)
        top, F = candidates(cat, us, sl)
        flat = F.reshape(-1, len(FEATURES))
        recs["rule"].append(top[:, :K])
        for name, prob in (("new", model.predict_proba(flat[:, keep])[:, 1]),
                           ("base", predict(base_ranker, flat) if base_ok else None)):
            if prob is not None:
                recs[name].append(np.take_along_axis(top, np.argsort(-prob.reshape(top.shape), 1, kind="stable")[:, :K], 1))
    maps = {name: round(float(accuracy_at_k(np.concatenate(r), users, truth, ks=(K,))[0][f"map@{K}"]), 5)
            for name, r in recs.items() if r}
    ref = "base" if base_ok and not base_seen_eval else "rule"
    accepted = bool(maps["new"] >= maps[ref] * (1 - tolerance))
    report.update(status="trained", accepted=accepted, compared_with=ref, eval_users=int(len(users)),
                  evaluation_key=f"ranker:time_week:until={ev.ts.max()}:users={len(users)}", val_auc=round(auc, 4),
                  **{f"{name}_map@{K}": v for name, v in maps.items()})
    if not base_ok and base_ranker:
        report["base_problem"] = ranker_problem(base_ranker)
    if base_seen_eval:
        report["base_seen_eval_window"] = str(base_end)
    if accepted:
        final = train(db, source, out, tower_dir, cat=cat, ev=ev)
        report.update(path=final["path"], final_train_users=final["train_users"])
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="hm", help="랭커를 학습할 데이터셋 (Amazon은 오프라인에서 효과가 없어 기본 제외)")
    ap.add_argument("--out", type=Path, default=Path("/data/models/ranker"))
    ap.add_argument("--tower-dir", default=None, help="UME 타워 폴더 (기본: /data/models/ume/{source}_test_pca)")
    a = ap.parse_args()
    with SessionLocal() as db:
        print(json.dumps(train(db, a.source, a.out, a.tower_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
