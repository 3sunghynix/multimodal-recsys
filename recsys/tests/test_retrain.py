"""UME 재학습 (recsys/retrain.py). 서빙 DB 구조(backend/app)와 torch가 필요해 trainer · api 컨테이너에서 돈다.

docker compose run --rm --no-deps -v "$PWD/recsys:/app/recsys" api python -m pytest -q recsys/tests/test_retrain.py
"""
import json
from datetime import datetime, timedelta

import numpy as np
import pytest

pytest.importorskip("torch")
app_db = pytest.importorskip("app.db")

from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import recommenders  # noqa: E402
from app.models import Event, Item, MetricSnapshot, ModelVersion, User  # noqa: E402
from app.recommenders.ume import tower_problem  # noqa: E402
from app.services import catalog_index  # noqa: E402
from recsys.retrain import run  # noqa: E402

D, P = 8, {"hidden": 16, "emb_dim": 4, "dropout": 0.0, "batch_size": 64, "max_hist": 10, "seed": 0}
T0 = datetime(2026, 1, 1)


def _base_tower(path, seed):
    rng = np.random.default_rng(seed)
    w = {}
    for side in ("user", "item"):
        w.update({f"{side}.0.weight": rng.normal(size=(P["hidden"], D)), f"{side}.0.bias": rng.normal(size=P["hidden"]),
                  f"{side}.3.weight": rng.normal(size=(P["emb_dim"], P["hidden"])), f"{side}.3.bias": rng.normal(size=P["emb_dim"])})
    path.mkdir(parents=True)
    np.savez(path / "weights.npz", **{k: v.astype(np.float32) for k, v in w.items()})
    (path / "config.json").write_text(json.dumps(P), encoding="utf-8")


@pytest.fixture()
def env(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'rt.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    rng = np.random.default_rng(1)
    with Session() as s:
        items = []
        for i in range(40):
            v = rng.normal(size=D)
            items.append(Item(source="hm", source_item_id=str(i), title=f"i{i}",
                              text_embedding=(v / np.linalg.norm(v)).tolist(), image_embedding=(v[::-1] / np.linalg.norm(v)).tolist()))
        users = [User(source="sim") for _ in range(30)]
        s.add_all(items + users)
        s.flush()
        # 고객마다 비슷한 상품(같은 4개 묶음)을 시간순으로 구매 · 클릭 → 학습할 신호가 있음
        for u, user in enumerate(users):
            group = [items[(u % 10) * 4 + j] for j in range(4)]
            for t, it in enumerate(group * 2):
                s.add(Event(user_id=user.id, item_id=it.id, type="purchase" if t % 2 else "click",
                            created_at=T0 + timedelta(hours=u * 10 + t)))
        base_dir = tmp_path / "models" / "base" / "hm"
        for j, m in enumerate(("text", "image")):
            _base_tower(base_dir / m, seed=j)
        s.add(ModelVersion(name="ume", version="base", status="active",
                           params={"artifact_dirs": {"hm": str(base_dir)}, "gamma": 0.0, "beta": {}}))
        s.commit()
    catalog_index._cache["key"] = None
    return Session, tmp_path


def test_retrain_registers_candidate_usable_by_serving(env):
    Session, tmp = env
    with Session() as db:
        r = run(db, tmp / "models", min_new_logs=10, holdout=0.2, k=5, min_samples=10, tolerance=1.0, log=lambda *_: None)
        assert r["status"] == "trained" and r["sources"]["hm"]["status"] == "trained"
        src = r["sources"]["hm"]
        assert src["eval_users"] > 0 and src["base_recall@5"] is not None and src["new_recall@5"] is not None
        new = db.get(ModelVersion, r["registered_version_id"])
        assert new.status == "candidate"                                      # 운영 교체는 하지 않는다
        assert db.query(ModelVersion).filter_by(status="active").one().version == "base"
        new_dir = new.params["artifact_dirs"]["hm"]
        assert new_dir != str(tmp / "models" / "base" / "hm") and new.params["trained_until_event_id"] == r["max_event_id"]
        assert tower_problem(new_dir, D) is None                              # 서빙이 읽을 수 있는 가중치
        recs = recommenders.get("ume", new.params).recommend(db, 1, 5)        # 새 버전으로 실제 추천
        assert len(recs) == 5
        metrics = {m.metric for m in db.query(MetricSnapshot).all()}
        assert metrics == {"hm/holdout/recall@5"}
        assert db.query(MetricSnapshot).count() == 2                          # 기준 · 새 모델 하나씩
        assert (tmp / "models").glob("retrain_*/report.json")


def test_skips_when_not_enough_new_logs(env):
    Session, tmp = env
    with Session() as db:
        r = run(db, tmp / "models", min_new_logs=10_000, log=lambda *_: None)
        assert r["status"] == "skipped" and db.query(ModelVersion).count() == 1


def test_not_registered_when_worse(env, monkeypatch):
    Session, tmp = env
    import recsys.retrain as rt
    scores = iter([0.50, 0.30])                                                # 기준 0.50 → 새 모델 0.30 (나빠짐)
    monkeypatch.setattr(rt, "recall_at_k", lambda rec, targets: next(scores))
    with Session() as db:
        r = run(db, tmp / "models", min_new_logs=10, holdout=0.2, k=5, min_samples=10, tolerance=0.02, log=lambda *_: None)
        assert r["sources"]["hm"]["accepted"] is False and "registered_version_id" not in r
        assert db.query(ModelVersion).count() == 1                             # 기존 운영 모델 그대로


def test_holdout_leaves_history_for_every_eval_user():
    import pandas as pd

    from recsys.retrain import split_holdout
    ev = pd.DataFrame({"event_id": range(1, 9), "user_id": [1, 1, 1, 2, 2, 3, 4, 4],
                       "is_setup": [False] * 8, "item_id": range(8)})
    train, test = split_holdout(ev, 1.0)
    assert sorted(test.event_id) == [3, 5, 8]                                  # 이벤트 2개 이상인 고객 1 · 2 · 4의 마지막
    assert set(test.user_id) <= set(train.user_id)                             # 평가 고객은 모두 학습 이력이 있음
