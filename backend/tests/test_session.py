"""현재 세션 구분 — 클라이언트 session_id (이벤트연결규약 2.4). Redis 없이 DB 경로."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.models import Event
from app.services import feature_store
from app.services.session import session_click_ids
from scripts.seed_sample import seed


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "redis_url", None)
    feature_store.reset_for_tests()
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'sess.db'}")
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
    yield Session, TestClient(app)
    app.dependency_overrides.clear()


def _post(c, **kw):
    r = c.post("/events", json={"user_id": 1, "type": "click", **kw})
    assert r.status_code == 201, r.text
    return r.json()


def test_session_id_separates_back_to_back_sessions(env):
    """몇 초 간격으로 이어 보낸 두 세션 — 30분 규칙이면 합쳐지지만 session_id로 나뉜다."""
    Session, c = env
    _post(c, item_id=3, session_id="run:c1:s1")
    _post(c, item_id=4, session_id="run:c1:s1")
    _post(c, item_id=5, session_id="run:c1:s2")
    with Session() as s:
        assert session_click_ids(s, 1) == [5]                              # 마지막 이벤트의 세션 = s2
        assert session_click_ids(s, 1, session_id="run:c1:s1") == [4, 3]   # 요청이 세션을 지정하면 그 세션
        assert session_click_ids(s, 1, session_id="run:c1:s3") == []       # 아직 클릭 없는 새 세션


def test_without_session_id_keeps_30_minute_rule(env):
    """session_id 없는 이벤트만 있으면 예전처럼 최근 30분 클릭 (샘플 데이터의 지금 시각 클릭 포함)."""
    Session, c = env
    _post(c, item_id=3)
    _post(c, item_id=4)
    with Session() as s:
        assert session_click_ids(s, 1)[:2] == [4, 3]


def test_setup_events_do_not_define_current_session(env):
    """초기 이력(is_setup)은 세션 판단에서 빠진다."""
    Session, c = env
    _post(c, item_id=3, session_id="live")
    _post(c, item_id=4, session_id="setup", is_setup=True)
    with Session() as s:
        assert session_click_ids(s, 1) == [3]


def test_session_id_is_stored_and_part_of_idempotency(env):
    Session, c = env
    eid = "6b1c7d9e-0000-4000-8000-000000000001"
    body = _post(c, item_id=3, session_id="s1", event_id=eid)
    assert body["session_id"] == "s1"
    again = c.post("/events", json={"user_id": 1, "type": "click", "item_id": 3, "session_id": "s1", "event_id": eid})
    assert again.status_code == 200                                        # 같은 내용 재전송 = 멱등
    other = c.post("/events", json={"user_id": 1, "type": "click", "item_id": 3, "session_id": "s2", "event_id": eid})
    assert other.status_code == 409                                        # 세션이 다르면 다른 이벤트
    with Session() as s:
        assert s.query(Event).filter_by(session_id="s1").count() == 1


def test_recommend_accepts_session_id(env):
    _, c = env
    _post(c, item_id=3, session_id="a")
    r = c.get("/recommend/1", params={"k": 3, "session_id": "a"})
    assert r.status_code == 200
    assert c.get("/recommend/1", params={"k": 3, "session_id": "x" * 65}).status_code == 422   # 64자 제한
