"""시뮬레이터 ↔ 백엔드 1단계 계약 (A1): POST /users 멱등, event_id 멱등, is_setup 집계 제외.

광석 PR #29의 "다음 PR 전에 확정할 계약" 1·3·5를 반영한다.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
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
