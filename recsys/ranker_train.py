"""B1 랭커 학습 잡: 서비스 구매 로그 전체로 H&M 랭커를 학습해 저장한다 (서빙은 모델 버전 params "ranker"로 켬).

실행: docker compose --profile trainer run --rm -v ./backend/app:/app/app:ro trainer python -m recsys.ranker_train [--source hm] [--out /data/models/ranker]
      → /data/models/ranker/{source}/model.joblib · meta.json
등록: docker compose run --rm api python -m scripts.register_ume --version v1-g1-rk --ranker-root /data/models/ranker --activate

학습: 서비스 범위 구매(pop_eval.service_purchases)의 마지막 7일을 라벨, 그 전을 이력으로 후보 300개 · 특성 (ranker_eval과 같은 함수).
평가는 recsys.ranker_eval (평가 구간을 뗀 뒤 같은 방식으로 학습)로 한다 — 여기 val_auc는 학습 표본 안의 확인용.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
from app.db import SessionLocal  # noqa: E402
from app.services.catalog_index import get_index  # noqa: E402
from app.services.ranker import CANDIDATES, save_ranker  # noqa: E402
from recsys.pop_eval import service_purchases  # noqa: E402
from recsys.ranker_eval import DECAY, GAMMA, NEG_PER_USER, SEED, Catalog, build_training, fit  # noqa: E402


def train(db, source: str, out: Path) -> dict:
    rng = np.random.default_rng(SEED)
    cat = Catalog(db, get_index(db), source)
    ev = service_purchases(db, cat.ids)
    X, y, recall, n_users = build_training(cat, ev, "time_week", rng)
    model, keep, used, auc, _ = fit(X, y, rng)
    meta = {"source": source, "trained_at": datetime.now(timezone.utc).isoformat(), "label_window": "last 7 days",
            "label_end": str(ev.ts.max()), "train_users": int(n_users), "train_rows": int(len(y)), "train_pos": int(y.sum()),
            "candidate_recall": round(recall, 4), "val_auc": round(auc, 4), "candidates": CANDIDATES, "gamma": GAMMA,
            "recency_decay": DECAY.get(source), "neg_per_user": NEG_PER_USER, "model": type(model).__name__,
            "iterations": int(model.n_iter_)}
    save_ranker(str(out / source), model, used, meta)
    return {**meta, "features": used, "path": str(out / source)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="hm", help="랭커를 학습할 데이터셋 (Amazon은 오프라인에서 효과가 없어 기본 제외)")
    ap.add_argument("--out", type=Path, default=Path("/data/models/ranker"))
    a = ap.parse_args()
    with SessionLocal() as db:
        print(json.dumps(train(db, a.source, a.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
