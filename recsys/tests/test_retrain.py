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
        r = run(db, tmp / "models", min_new_logs=10, holdout=0.2, k=5, min_samples=10, tolerance=1.0,
                recall_tolerance=1.0, log=lambda *_: None)
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
        assert 0 <= src["base_map@12"] <= 1 and 0 <= src["new_map@12"] <= 1    # 모니터링 runner 비교 지표
        assert src["evaluation_key"].startswith("leave-last-out:")
        metrics = {m.metric for m in db.query(MetricSnapshot).all()}
        assert metrics == {"hm/holdout/recall@5", "hm/holdout/map@12"}
        assert db.query(MetricSnapshot).count() == 4                          # 지표 2개 × 기준 · 새 모델
        assert (tmp / "models").glob("retrain_*/report.json")


def test_skips_when_not_enough_new_logs(env):
    Session, tmp = env
    with Session() as db:
        r = run(db, tmp / "models", min_new_logs=10_000, log=lambda *_: None)
        assert r["status"] == "skipped" and db.query(ModelVersion).count() == 1


def test_not_registered_when_worse(env, monkeypatch):
    Session, tmp = env
    import recsys.retrain as rt
    maps = iter([0.50, 0.30])                                                  # MAP@12 기준 0.50 → 새 모델 0.30 (나빠짐)
    monkeypatch.setattr(rt, "accuracy_at_k", lambda rec, users, truth, ks: ({f"map@{ks[0]}": next(maps)}, {}))
    with Session() as db:
        r = run(db, tmp / "models", min_new_logs=10, holdout=0.2, k=5, min_samples=10, tolerance=0.02, log=lambda *_: None)
        assert r["sources"]["hm"]["accepted"] is False and "registered_version_id" not in r
        assert db.query(ModelVersion).count() == 1                             # 기존 운영 모델 그대로


@pytest.mark.parametrize("base_map,new_map,base_rec,new_rec,ok", [
    (0.0425, 0.0428, 0.1530, 0.1516, True),     # 11-16 실측 H&M
    (0.0039, 0.0041, 0.0198, 0.0192, True),     # 11-16 실측 Amazon: Recall −3%는 보조 허용 10% 안
    (0.0400, 0.0393, 0.15, 0.15, True),         # MAP −1.75%: 주 허용 2% 안
    (0.0400, 0.0390, 0.15, 0.15, False),        # MAP −2.5%: 거절
    (0.0400, 0.0500, 0.15, 0.13, False),        # MAP는 좋아져도 Recall −13%면 거절 (안전장치)
    (None, None, None, None, True),             # 평가 고객 없음 → 판정 안 함
])
def test_accept_rule_map12_primary_recall_guard(base_map, new_map, base_rec, new_rec, ok):
    from recsys.retrain import accept
    assert accept(base_map, new_map, base_rec, new_rec) is ok


def test_holdout_map12_matches_reference():
    """trainer의 MAP@12는 공통 평가 코드와 같다: 정답 1개가 3위면 AP = 1/3, 없으면 0."""
    import pandas as pd

    from recsys.eval.metrics import accuracy_at_k
    rec = np.array([[5, 6, 7] + list(range(10, 19)), [1, 2, 3] + list(range(20, 29))])
    truth = pd.DataFrame({"user_id": [100, 200], "item_id": [7, 99]})
    m = accuracy_at_k(rec, np.array([100, 200]), truth, ks=(12,))[0]["map@12"]
    assert m == pytest.approx((1 / 3 + 0) / 2)


def test_holdout_leaves_history_for_every_eval_user():
    import pandas as pd

    from recsys.retrain import split_holdout
    ev = pd.DataFrame({"event_id": range(1, 9), "user_id": [1, 1, 1, 2, 2, 3, 4, 4],
                       "is_setup": [False] * 8, "item_id": range(8)})
    train, test = split_holdout(ev, 1.0)
    assert sorted(test.event_id) == [3, 5, 8]                                  # 이벤트 2개 이상인 고객 1 · 2 · 4의 마지막
    assert set(test.user_id) <= set(train.user_id)                             # 평가 고객은 모두 학습 이력이 있음


# ---------------------------------------------------------------- 학습형 랭커 재학습 (B1 자동화)

def _fake_ranker(path, **meta):
    """규칙 순위를 그대로 따르는 GBDT + meta (app.services.ranker 형식)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    from app.services.ranker import FEATURES, save_ranker
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, len(FEATURES))).astype(np.float32)
    X[:, 2] = rng.integers(0, 20, 500)
    model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, (X[:, 2] < 3).astype(int))
    save_ranker(str(path), model, FEATURES, {"source": "hm", **meta})


@pytest.fixture()
def rk_env(tmp_path, monkeypatch):
    """H&M 고객 60명(시뮬레이터 아님 — 랭커 학습 범위)이 32일 동안 4개 묶음을 반복 구매. 기준 버전에 랭커가 켜져 있음."""
    import recsys.ranker_eval as re_
    import recsys.ranker_train as rt_
    monkeypatch.setattr(re_, "C", 20)                    # 상품 40개 카탈로그: 후보 20개 (≥ MAP@12)
    monkeypatch.setattr(rt_, "MIN_LABEL_USERS", 5)
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'rk.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    rng = np.random.default_rng(2)
    with Session() as s:
        items = []
        for i in range(40):
            v = rng.normal(size=D)
            items.append(Item(source="hm", source_item_id=str(i), title=f"i{i}", price=float(10 + i),
                              text_embedding=(v / np.linalg.norm(v)).tolist(), image_embedding=(v[::-1] / np.linalg.norm(v)).tolist()))
        users = [User(source="hm") for _ in range(60)]
        s.add_all(items + users)
        s.flush()
        for u, user in enumerate(users):
            group = [items[(u % 10) * 4 + j] for j in range(4)]
            for t in range(8):                            # 4일 간격 → 마지막 7일 · 그 전 7일 모두 구매가 있음
                s.add(Event(user_id=user.id, item_id=group[t % 4].id, type="purchase",
                            created_at=T0 + timedelta(days=t * 4 + u % 4)))
        base_dir = tmp_path / "models" / "base" / "hm"
        for j, m in enumerate(("text", "image")):
            _base_tower(base_dir / m, seed=j)
        _fake_ranker(tmp_path / "ranker" / "hm")
        s.add(ModelVersion(name="ume", version="base-rk", status="active",
                           params={"artifact_dirs": {"hm": str(base_dir)}, "gamma": 1.0, "beta": {},
                                   "ranker": {"hm": str(tmp_path / "ranker" / "hm")}}))
        s.commit()
    catalog_index._cache["key"] = None
    return Session, tmp_path


def _run(db, tmp, **kw):
    return run(db, tmp / "models", min_new_logs=10, holdout=0.2, k=5, min_samples=10, log=lambda *_: None, **kw)


def test_ranker_retrained_on_candidate_towers_and_registered(rk_env):
    """기준 버전에 랭커가 있으면 후보 타워로 랭커도 다시 학습 → 새 폴더로 후보에 넣고, 운영 랭커 폴더는 그대로."""
    from app.services.ranker import ranker_problem
    Session, tmp = rk_env
    base_model = (tmp / "ranker" / "hm" / "model.joblib").stat().st_mtime
    with Session() as db:
        r = _run(db, tmp, tolerance=1.0, recall_tolerance=1.0)
        rr = r["rankers"]["hm"]
        assert rr["status"] == "trained", rr
        assert rr["accepted"] and rr["compared_with"] == "base" and rr["base_tower_dir"] != rr["tower_dir"]   # 기준 조합은 기준 타워로
        assert {"rule_map@12", "new_map@12", "base_map@12"} <= set(rr) and rr["evaluation_key"].startswith("ranker:time_week:")
        new = db.get(ModelVersion, r["registered_version_id"])
        path = new.params["ranker"]["hm"]
        assert path == rr["path"] and path.startswith(r["out"]) and ranker_problem(path) is None
        meta = json.loads(open(f"{path}/meta.json", encoding="utf-8").read())
        assert meta["tower_dir"] == new.params["artifact_dirs"]["hm"]                  # 후보 타워로 학습
        assert (tmp / "ranker" / "hm" / "model.joblib").stat().st_mtime == base_model  # 운영 랭커 덮어쓰지 않음
        m = recommenders.get("ume", new.params)
        assert len(m.recommend(db, 1, 5)) == 5 and m.ranker_sources == ["hm"]          # 서빙이 새 랭커로 추천
        assert {"hm/ranker/map@12"} <= {x.metric for x in db.query(MetricSnapshot).all()}


def test_ranker_kept_when_new_one_is_worse(rk_env, monkeypatch):
    import recsys.ranker_train as rt_
    Session, tmp = rk_env
    maps = iter([0.10, 0.10, 0.50])                                                   # 규칙 · 새 랭커 · 기준 랭커
    monkeypatch.setattr(rt_, "accuracy_at_k", lambda rec, users, truth, ks: ({f"map@{ks[0]}": next(maps)}, {}))
    import recsys.retrain as retrain_mod
    monkeypatch.setattr(retrain_mod, "accept", lambda *a, **k: True)                  # 타워는 통과
    with Session() as db:
        r = _run(db, tmp, tolerance=0.02)
        assert r["rankers"]["hm"]["accepted"] is False and "path" not in r["rankers"]["hm"]
        new = db.get(ModelVersion, r["registered_version_id"])
        assert new.params["ranker"]["hm"] == str(tmp / "ranker" / "hm")              # 기준 랭커 유지


def test_ranker_failure_does_not_block_tower_candidate(rk_env, monkeypatch):
    import recsys.ranker_train as rt_
    Session, tmp = rk_env

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(rt_, "retrain_ranker", boom)
    with Session() as db:
        r = _run(db, tmp, tolerance=1.0, recall_tolerance=1.0)
        assert r["rankers"]["hm"]["status"] == "failed" and r["registered_version_id"]
        assert db.get(ModelVersion, r["registered_version_id"]).params["ranker"]["hm"] == str(tmp / "ranker" / "hm")


def test_ranker_skipped_without_service_purchases(env, tmp_path):
    """시뮬레이터 고객 구매만 있으면 랭커는 다시 학습하지 않는다 (서비스 범위 = 인기 통계와 같음)."""
    Session, tmp = env
    _fake_ranker(tmp / "ranker" / "hm")
    with Session() as db:
        base = db.query(ModelVersion).one()
        base.params = {**base.params, "ranker": {"hm": str(tmp / "ranker" / "hm")}}
        db.commit()
        r = _run(db, tmp, tolerance=1.0, recall_tolerance=1.0)
        assert r["rankers"]["hm"]["status"] == "skipped"
        assert db.get(ModelVersion, r["registered_version_id"]).params["ranker"]["hm"] == str(tmp / "ranker" / "hm")


def test_no_ranker_flag_skips(rk_env):
    Session, tmp = rk_env
    with Session() as db:
        r = _run(db, tmp, tolerance=1.0, recall_tolerance=1.0, ranker=False)
        assert "rankers" not in r


def test_ranker_compared_with_refit_base_when_base_saw_eval_window(rk_env):
    """기준 랭커가 평가 구간을 라벨로 학습했으면(label_end ≥ 평가 구간 시작) 정답을 본 모델이라, 기준 타워로 같은 중첩 분할에서
    다시 학습한 랭커(base_refit)와 비교한다. 정답을 본 값(base_map@12)은 참고로 남는다."""
    Session, tmp = rk_env
    _fake_ranker(tmp / "ranker" / "hm", label_end="2026-02-01 00:00:00")             # 마지막 구매(1월 말) 이후
    with Session() as db:
        r = _run(db, tmp, tolerance=1.0, recall_tolerance=1.0)
        rr = r["rankers"]["hm"]
        assert rr["compared_with"] == "base_refit" and rr["base_seen_eval_window"].startswith("2026-02-01")
        assert {"base_refit_map@12", "base_map@12", "rule_map@12"} <= set(rr)
        metrics = {x.metric for x in db.query(MetricSnapshot).all()}
        assert "hm/ranker_refit/map@12" in metrics and "hm/ranker/map@12" not in metrics      # 기준 랭커 값으로 읽히지 않게


def test_ranker_compared_with_base_tower_rule_when_base_ranker_missing(rk_env):
    """기준 랭커 파일이 없으면 서빙도 규칙 순이므로 기준 조합 = 기준 타워 규칙 순 (base_rule)."""
    import shutil
    Session, tmp = rk_env
    shutil.rmtree(tmp / "ranker" / "hm")
    with Session() as db:
        rr = _run(db, tmp, tolerance=1.0, recall_tolerance=1.0)["rankers"]["hm"]
        assert rr["compared_with"] == "base_rule" and "base_rule_map@12" in rr and rr["base_problem"]
