"""시뮬레이터 ↔ 백엔드 1단계 계약 (A1): POST /users 멱등, event_id 멱등, is_setup 집계 제외.

광석 PR #29의 "다음 PR 전에 확정할 계약" 1·3·5를 반영한다.
동시 재시도 복구 · 인기도 집계 격리는 유환 PR #30 리뷰 반영.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.models import Event, User
from app.schemas import EventIn, UserIn
from scripts.seed_sample import seed


@pytest.fixture(autouse=True)
def _no_encoders(monkeypatch):
    monkeypatch.setattr(settings, "load_image_encoder", False)
    monkeypatch.setattr(settings, "load_text_encoder", False)


@pytest.fixture()
def client(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    with Session() as s:
        seed(s)

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


# ─── POST /users 멱등 ───────────────────────────────────────────────────────

def test_create_user_retry_returns_same_id(client):
    r1 = client.post("/users", json={"source": "sim", "source_user_id": "sim-a", "persona": "p1"})
    assert r1.status_code == 201
    uid = r1.json()["id"]
    r2 = client.post("/users", json={"source": "sim", "source_user_id": "sim-a", "persona": "p1"})
    assert r2.status_code == 200 and r2.json()["id"] == uid


def test_create_user_different_persona_conflicts(client):
    client.post("/users", json={"source": "sim", "source_user_id": "sim-b", "persona": "p1"})
    r = client.post("/users", json={"source": "sim", "source_user_id": "sim-b", "persona": "p2"})
    assert r.status_code == 409


def test_create_user_source_separates_identity(client):
    """같은 source_user_id라도 source가 다르면 서로 다른 고객."""
    r1 = client.post("/users", json={"source": "sim", "source_user_id": "shared"}).json()
    r2 = client.post("/users", json={"source": "sample", "source_user_id": "shared"}).json()
    assert r1["id"] != r2["id"]


def test_create_user_anonymous_always_new(client):
    """source_user_id=None(익명)이면 재전송이라도 새 user를 만든다 (멱등 키가 없음)."""
    ids = {client.post("/users", json={"source": "sim"}).json()["id"] for _ in range(3)}
    assert len(ids) == 3


# ─── POST /events 멱등 ──────────────────────────────────────────────────────

def test_event_server_generates_event_id_when_missing(client):
    body = client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3}).json()
    assert body["event_id"] and len(body["event_id"]) == 36      # UUID 문자열 형식


def test_event_same_event_id_same_content_is_idempotent(client):
    eid = "11111111-1111-1111-1111-111111111111"
    r1 = client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3, "event_id": eid})
    r2 = client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3, "event_id": eid})
    assert r1.status_code == 201 and r2.status_code == 200
    assert r1.json()["id"] == r2.json()["id"]


def test_event_same_event_id_different_content_conflicts(client):
    eid = "22222222-2222-2222-2222-222222222222"
    assert client.post("/events", json={
        "user_id": 1, "type": "click", "item_id": 3, "event_id": eid}).status_code == 201
    r = client.post("/events", json={"user_id": 2, "type": "click", "item_id": 3, "event_id": eid})
    assert r.status_code == 409


# ─── is_setup 집계 제외 ─────────────────────────────────────────────────────

def test_is_setup_events_excluded_from_experiment_metrics(client):
    ids = [m["id"] for m in client.get("/models").json()]
    exp = client.post("/experiments", json={"name": "t", "variants": [
        {"name": "A", "model_version_id": ids[0], "weight": 1},
        {"name": "B", "model_version_id": ids[1], "weight": 1}]}).json()

    # 초기 이력 1건(구매) + 실험 노출 1건 + 실험 클릭 1건 — 모두 user 1 · variant A
    setup = client.post("/events", json={"user_id": 1, "type": "purchase", "item_id": 3,
                                         "experiment_id": exp["id"], "variant": "A", "is_setup": True})
    live_view = client.post("/events", json={"user_id": 1, "type": "view", "item_id": 3,
                                             "experiment_id": exp["id"], "variant": "A"})
    live_click = client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3,
                                              "experiment_id": exp["id"], "variant": "A"})
    assert setup.status_code == 201 and live_view.status_code == 201 and live_click.status_code == 201

    m = {row["variant"]: row for row in client.get(f"/experiments/{exp['id']}/metrics").json()}
    # is_setup=True 구매는 세지 않는다. 실험 노출/클릭만 보인다.
    assert m["A"]["impressions"] == 1 and m["A"]["clicks"] == 1 and m["A"]["purchases"] == 0


def test_popularity_recommendation_unaffected_by_setup_events(client):
    """특정 상품에 is_setup=True 구매를 쏟아부어도 공통 인기도(popularity 추천)가 변하지 않아야 한다.
    is_setup=True 가 _popular 집계에 섞이면 그 상품이 Top-K 로 올라와 before != after 가 된다."""
    before = client.get("/recommend/1?k=10").json()["items"]
    # user 1 의 initial-history 로 item 7 을 50번 구매했다고 가정 (실제로는 극단적인 가중치)
    for _ in range(50):
        r = client.post("/events", json={"user_id": 1, "type": "purchase", "item_id": 7, "is_setup": True})
        assert r.status_code == 201
    after = client.get("/recommend/1?k=10").json()["items"]
    assert after == before, f"setup 이벤트가 공통 인기도에 반영됐다: before={before} after={after}"


# ─── 동시 재시도 복구 (레이스 UNIQUE 충돌 → 멱등 복구) ───────────────────────

def _one_shot_blind(session, missing_values=(None,)):
    """pre-check SELECT 가 '아직 저장된 행을 못 봤다'고 돌려주게 한 번만 가린다.
    두 번째 호출부터는 실제 결과. 이로써 INSERT 가 UNIQUE 로 깨지고 복구 경로가 실행된다."""
    original = session.scalar
    calls = {"n": 0}

    def blind(stmt, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return missing_values[0]
        return original(stmt, *args, **kwargs)

    session.scalar = blind
    return session


def test_concurrent_user_create_recovers_from_unique_violation(tmp_path):
    """같은 (source, source_user_id) 로 두 요청이 레이스: 승자는 저장, 패자는 UNIQUE 충돌.
    패자가 롤백 후 저장된 행을 재조회해 200 멱등으로 응답하는지."""
    from app.routers.users import create_user
    from fastapi import Response

    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'race.db'}")
    app_db.init_db(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    with Session() as writer:                            # 레이스 승자: 미리 저장
        writer.add(User(source="sim", source_user_id="race-u", persona="p1"))
        writer.commit()

    with Session() as loser:                             # 레이스 패자: pre-check SELECT 를 가려 INSERT 를 유도
        _one_shot_blind(loser)
        r = Response()
        result = create_user(UserIn(source="sim", source_user_id="race-u", persona="p1"), r, loser)
        assert r.status_code == 200
        assert result.source_user_id == "race-u" and result.persona == "p1"

    # 중복 저장 없음
    with Session() as s:
        assert s.query(User).filter_by(source="sim", source_user_id="race-u").count() == 1


def test_concurrent_user_create_persona_mismatch_still_409(tmp_path):
    """레이스 패자의 persona 가 저장분과 다르면 409."""
    from app.routers.users import create_user
    from fastapi import HTTPException, Response

    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'race2.db'}")
    app_db.init_db(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    with Session() as writer:
        writer.add(User(source="sim", source_user_id="race-x", persona="p1"))
        writer.commit()

    with Session() as loser:
        _one_shot_blind(loser)
        with pytest.raises(HTTPException) as exc:
            create_user(UserIn(source="sim", source_user_id="race-x", persona="p2"), Response(), loser)
        assert exc.value.status_code == 409


def test_concurrent_event_create_recovers_from_unique_violation(tmp_path):
    """같은 event_id 로 두 요청이 레이스. 패자가 롤백 후 저장된 이벤트를 200 으로 돌려주는지."""
    from app.routers.events import log_event
    from fastapi import Response
    from uuid import UUID

    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'race3.db'}")
    app_db.init_db(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Session() as s:
        seed(s)                                           # user 1 · item 3 등을 심음

    eid = UUID("33333333-3333-3333-3333-333333333333")
    with Session() as writer:                             # 레이스 승자
        writer.add(Event(event_id=eid, user_id=1, type="click", item_id=3, is_setup=False))
        writer.commit()

    with Session() as loser:
        _one_shot_blind(loser)
        r = Response()
        result = log_event(EventIn(user_id=1, type="click", item_id=3, event_id=eid), r, loser)
        assert r.status_code == 200 and result.event_id == eid

    with Session() as s:
        assert s.query(Event).filter_by(event_id=eid).count() == 1
