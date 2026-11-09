"""UME 서빙 (recommenders/ume.py). 학습 산출물과 같은 형식의 가짜 가중치로 SQLite에서 확인한다."""
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app import recommenders
from app.config import settings
from app.main import app
from app.models import Brand, Event, Item, ModelVersion, User
from app.recommenders.popularity import _popular
from app.recommenders.ume import _z, mlp

D, H, E = 8, 16, 4
T0 = datetime(2023, 1, 1)


def write_tower(path, seed):
    """recsys.ume.save_tower와 같은 키 (nn.Sequential의 0번 · 3번 Linear)."""
    rng = np.random.default_rng(seed)
    w = {}
    for side in ("user", "item"):
        w.update({f"{side}.0.weight": rng.normal(size=(H, D)), f"{side}.0.bias": rng.normal(size=H),
                  f"{side}.3.weight": rng.normal(size=(E, H)), f"{side}.3.bias": rng.normal(size=E)})
    w = {k: v.astype(np.float32) for k, v in w.items()}
    path.mkdir(parents=True)
    np.savez(path / "weights.npz", **w)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "load_image_encoder", False)
    monkeypatch.setattr(settings, "load_text_encoder", False)
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'ume.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    rng = np.random.default_rng(0)
    with Session() as s:
        brands = [Brand(name=f"b{i}") for i in range(3)]
        s.add_all(brands)
        s.flush()
        items = []
        for src, n in (("amazon", 30), ("hm", 10)):
            for i in range(n):
                v = rng.normal(size=D)
                items.append(Item(source=src, source_item_id=str(i), brand_id=brands[i % 3].id if src == "amazon" else brands[0].id,
                                  title=f"{src}-{i}", text_embedding=(v / np.linalg.norm(v)).tolist(),
                                  image_embedding=(v[::-1] / np.linalg.norm(v)).tolist()))
        users = [User(source="amazon"), User(source="sim"), User(source="sim")]
        s.add_all(items + users)
        s.flush()
        amz = [it for it in items if it.source == "amazon"]
        hm = [it for it in items if it.source == "hm"]
        ev = [Event(user_id=users[0].id, item_id=amz[j].id, type="purchase", created_at=T0 + timedelta(days=j)) for j in (0, 3, 6, 4)]
        ev += [Event(user_id=users[1].id, item_id=it.id, type="click", created_at=T0) for it in (amz[1], hm[2])]
        ev += [Event(user_id=users[2].id, item_id=amz[5].id, type="purchase", created_at=T0 + timedelta(days=j)) for j in range(5)]
        s.add_all(ev)
        s.commit()
        ids = {"amazon": [it.id for it in amz], "hm": [it.id for it in hm], "users": [u.id for u in users]}
    for src in ("amazon", "hm"):
        for j, m in enumerate(("text", "image")):
            write_tower(tmp_path / "art" / src / m, seed=10 * j + len(src))
    params = {"artifact_dirs": {src: str(tmp_path / "art" / src) for src in ("amazon", "hm")}, "gamma": 0.0, "beta": {}}
    return Session, ids, params, tmp_path


def test_cold_user_falls_back_to_popularity(env):
    Session, ids, params, _ = env
    with Session() as s:
        new = User(source="sim")
        s.add(new)
        s.commit()
        assert recommenders.get("ume", params).recommend(s, new.id, 5) == _popular(s, 5)


def test_scores_match_numpy_definition(env):
    Session, ids, params, tmp = env
    with Session() as s:
        rec = recommenders.get("ume", params).recommend(s, ids["users"][0], 30)
        assert sorted(rec) == sorted(ids["amazon"])                     # 이력이 Amazon뿐 → Amazon 상품만
        # 직접 계산: 이력 평균 → 고객 타워, 상품 타워, z 합
        items = {it.id: it for it in s.query(Item).all()}
        hist = [ids["amazon"][j] for j in (0, 3, 6, 4)]
        score = 0
        for m, col in (("text", "text_embedding"), ("image", "image_embedding")):
            w = dict(np.load(tmp / "art" / "amazon" / m / "weights.npz"))
            u = mlp(w, "user", np.mean([getattr(items[i], col) for i in hist], axis=0))
            V = mlp(w, "item", np.array([getattr(items[i], col) for i in ids["amazon"]], np.float32))
            score = score + _z(V @ u)
        assert rec[:5] == [ids["amazon"][i] for i in np.argsort(-score, kind="stable")[:5]]


def test_multi_source_user_gets_both_sources(env):
    Session, ids, params, _ = env
    with Session() as s:
        rec = recommenders.get("ume", params).recommend(s, ids["users"][1], 12)
        assert set(rec) & set(ids["amazon"]) and set(rec) & set(ids["hm"])
        assert sum(i in ids["hm"] for i in rec) == 3                     # 카탈로그 비율 10/40 × 12 = 3


def test_brand_boost_and_popularity(env):
    Session, ids, params, _ = env
    with Session() as s:
        items = {it.id: it for it in s.query(Item).all()}
        top_brand = items[ids["amazon"][0]].brand_id                    # 고객 0 이력: 0 · 3 · 6 → 브랜드 b0, 4 → b1
        rec = recommenders.get("ume", {**params, "beta": {"amazon": 100.0}}).recommend(s, ids["users"][0], 10)
        assert all(items[i].brand_id == top_brand for i in rec)
        # 인기: 고객 2가 amazon[5]를 5번 구매 → γ가 크면 모두에게 1위
        rec = recommenders.get("ume", {**params, "gamma": 100.0}).recommend(s, ids["users"][0], 3)
        assert rec[0] == ids["amazon"][5]


def test_api_serves_ume(env):
    Session, ids, params, _ = env

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    with Session() as s:
        s.add(ModelVersion(name="ume", version="test", status="active", params=params))
        s.commit()
    app.dependency_overrides[app_db.get_db] = override
    try:
        r = TestClient(app).get(f"/recommend/{ids['users'][0]}", params={"k": 5})
        assert r.status_code == 200 and r.json()["model"] == "ume" and len(r.json()["items"]) == 5
    finally:
        app.dependency_overrides.clear()


# ---- PR #24 리뷰 회귀 테스트 (광석 재현 시나리오)

def test_new_item_after_popularity_cache(env):
    Session, ids, params, _ = env
    p = {**params, "gamma": 2.0}
    with Session() as s:
        recommenders.get("ume", p).recommend(s, ids["users"][0], 5)          # 인기 캐시 생성 (amazon 30개)
        v = np.ones(D) / np.sqrt(D)
        new = Item(source="amazon", source_item_id="new", title="new", text_embedding=v.tolist(), image_embedding=v.tolist())
        s.add(new)
        s.commit()
        rec = recommenders.get("ume", p).recommend(s, ids["users"][0], 31)  # 이벤트 변화 없이 상품만 추가
        assert len(rec) == 31 and new.id in rec


def test_partial_weights_fall_back(env):
    Session, ids, params, tmp = env
    from app.recommenders.ume import tower_problem
    (tmp / "art" / "amazon" / "image" / "weights.npz").unlink()
    assert "없음" in tower_problem(params["artifact_dirs"]["amazon"])
    with Session() as s:
        assert recommenders.get("ume", params).recommend(s, ids["users"][0], 5) == _popular(s, 5)
    from scripts.register_ume import params_for
    assert set(params_for(tmp / "art_root_missing")["artifact_dirs"]) == set()


def test_history_per_source_not_crowded_out(env):
    Session, ids, params, _ = env
    with Session() as s:
        u = ids["users"][1]                                                   # amazon · hm 둘 다 이력
        s.add_all([Event(user_id=u, item_id=ids["amazon"][j % 30], type="click", created_at=T0 + timedelta(hours=j))
                   for j in range(1, 202)])
        s.commit()
        rec = recommenders.get("ume", params).recommend(s, u, 12)
        assert any(i in ids["hm"] for i in rec)


# ---- PR #24 2차 리뷰: 가중치 모양 · 손상 파일 · 활성화 거절

def test_shape_mismatch_and_corrupt_weights_fall_back(env):
    Session, ids, params, tmp = env
    from app.recommenders.ume import load_tower, tower_problem
    art = params["artifact_dirs"]["amazon"]
    f = tmp / "art" / "amazon" / "image" / "weights.npz"
    w = dict(np.load(f))
    w["item.0.weight"] = w["item.0.weight"][:, :D - 1]                     # 상품 타워 입력만 7차원 (고객은 8)
    np.savez(f, **w)
    load_tower.cache_clear()
    assert "불일치" in tower_problem(art, D)
    with Session() as s:
        assert recommenders.get("ume", params).recommend(s, ids["users"][0], 5) == _popular(s, 5)
    f.write_bytes(b"not a zip")                                             # 손상된 파일
    load_tower.cache_clear()
    assert "읽기 실패" in tower_problem(art, D)
    with Session() as s:
        assert recommenders.get("ume", params).recommend(s, ids["users"][0], 5) == _popular(s, 5)
    load_tower.cache_clear()


def test_register_refuses_activation_without_ready_weights(env):
    Session, ids, params, tmp = env
    from scripts.register_ume import register
    with Session() as s:
        s.add(ModelVersion(name="popularity", status="active"))
        s.commit()
        with pytest.raises(ValueError):
            register(s, tmp / "empty_root", activate=True)
        s.rollback()
        assert [m.name for m in s.query(ModelVersion).filter_by(status="active")] == ["popularity"]
        assert s.query(ModelVersion).filter_by(name="ume").count() == 0


# ---- 실제 점수 반환 (#32 recommend_scored)

def test_scored_values_match_numpy_and_api_labels_model(env):
    Session, ids, params, tmp = env
    with Session() as s:
        scored = recommenders.get("ume", params).recommend_scored(s, ids["users"][0], 5)
        items = {it.id: it for it in s.query(Item).all()}
        hist = [ids["amazon"][j] for j in (0, 3, 6, 4)]
        total = 0
        for m, col in (("text", "text_embedding"), ("image", "image_embedding")):
            w = dict(np.load(tmp / "art" / "amazon" / m / "weights.npz"))
            u = mlp(w, "user", np.mean([getattr(items[i], col) for i in hist], axis=0))
            V = mlp(w, "item", np.array([getattr(items[i], col) for i in ids["amazon"]], np.float32))
            total = total + _z(V @ u)
        expect = {ids["amazon"][i]: round(float(total[i]), 4) for i in range(30)}
        assert [sc for _, sc in scored] == sorted((sc for _, sc in scored), reverse=True)      # 내림차순
        assert all(abs(sc - expect[i]) < 1e-3 for i, sc in scored)                             # 정의대로 계산한 점수

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    with Session() as s:
        s.add(ModelVersion(name="ume", version="scored", status="active", params=params))
        s.commit()
    app.dependency_overrides[app_db.get_db] = override
    try:
        body = TestClient(app).get(f"/recommend/{ids['users'][0]}", params={"k": 5}).json()
        assert body["model"] == "ume" and {r["score_kind"] for r in body["recommendations"]} == {"model"}
        assert [r["product_id"] for r in body["recommendations"]] == [str(i) for i in body["items"]]
    finally:
        app.dependency_overrides.clear()


def test_reason_says_popular_when_ume_falls_back(env):
    Session, ids, params, _ = env

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    with Session() as s:
        s.add(ModelVersion(name="ume", version="reason", status="active", params=params))
        cold = User(source="sim")
        s.add(cold)
        s.commit()
        cold_id = cold.id
    app.dependency_overrides[app_db.get_db] = override
    try:
        client = TestClient(app)
        warm = client.get(f"/recommend/{ids['users'][0]}", params={"k": 3}).json()["recommendations"]
        new = client.get(f"/recommend/{cold_id}", params={"k": 3}).json()["recommendations"]
        assert {r["reason"] for r in warm} == {"personalized"}
        assert {r["reason"] for r in new} == {"popular"}          # 신규 고객은 인기로 대체됐다고 표시
    finally:
        app.dependency_overrides.clear()


def test_score_basis_per_path(env):
    """세 경로의 점수 단위가 달라서 응답 score_basis 로 구분한다 (#38 리뷰).
    데이터셋 하나 → ume_zscore, 데이터셋 여럿 → rank_percentile (0 ~ 1), 신규 고객 → event_weight_sum (인기 대체)."""
    Session, ids, params, _ = env

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    with Session() as s:
        s.add(ModelVersion(name="ume", version="basis", status="active", params=params))
        cold = User(source="sim")
        s.add(cold)
        s.commit()
        cold_id = cold.id
    app.dependency_overrides[app_db.get_db] = override
    try:
        client = TestClient(app)

        def recs(uid):
            return client.get(f"/recommend/{uid}", params={"k": 8}).json()["recommendations"]

        single, multi, new = recs(ids["users"][0]), recs(ids["users"][1]), recs(cold_id)
        assert {r["score_basis"] for r in single} == {"ume_zscore"}
        assert {r["score_basis"] for r in multi} == {"rank_percentile"}
        assert all(0.0 <= r["score"] <= 1.0 for r in multi)
        assert {r["score_basis"] for r in new} == {"event_weight_sum"}
        assert {r["score_kind"] for r in single + multi + new} == {"model"}
        assert {r["reason"] for r in new} == {"popular"}
    finally:
        app.dependency_overrides.clear()
