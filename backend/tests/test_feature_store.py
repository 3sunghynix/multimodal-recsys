"""Feature Store (명세 4.4) — Redis 기반 사용자별 실시간 피처.

실제 Redis 데몬 없이 테스트 — redis 패키지 미설치 폴백 경로와, MagicMock 으로 Redis 호출 패턴
검증. 통합 테스트는 (1) redis_url 비어있을 때 전체 no-op (2) POST /events 가 record_event 호출
(3) GET /recommend 가 Redis 비면 DB 폴백 사용.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.services import feature_store
from scripts.seed_sample import seed


@pytest.fixture(autouse=True)
def _no_encoders(monkeypatch):
    monkeypatch.setattr(settings, "load_image_encoder", False)
    monkeypatch.setattr(settings, "load_text_encoder", False)


@pytest.fixture(autouse=True)
def _reset_feature_store():
    """각 테스트 사이에 feature_store 전역 상태 초기화."""
    feature_store.reset_for_tests()
    yield
    feature_store.reset_for_tests()


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


# ─── no-op 경로 (redis_url 비어있음) ──────────────────────────────────────

def test_record_event_noop_when_no_redis_url(monkeypatch):
    """redis_url=None 이면 전체 no-op — 예외 없이 끝나야 함."""
    monkeypatch.setattr(settings, "redis_url", None)
    feature_store.reset_for_tests()
    feature_store.record_event(user_id=1, event_type="click", item_id=5)  # 예외 X
    assert feature_store.get_recent_clicks(1) == []
    assert feature_store.get_recent_views(1) == []
    assert feature_store.get_session_click_count(1) == 0
    assert feature_store.ping_latency_ms() is None


def test_events_endpoint_works_without_redis(client):
    """POST /events 는 Redis 가 없어도 저장하고 201 돌려줘야 한다. feature_store 쪽은 silent."""
    r = client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3})
    assert r.status_code == 201


def test_features_latency_endpoint_reports_redis_off():
    """Redis 없으면 /features/latency 는 {redis: false} 로 돌려준다 (예외 안 남)."""
    with TestClient(app) as c:
        body = c.get("/features/latency").json()
    assert body["redis"] is False


def test_features_latency_includes_feature_read_times_when_redis_on(monkeypatch):
    """PR #46 리뷰 반영: ping 뿐 아니라 실제 피처 조회 세 가지 시간도 함께 재고 worst 로 종합."""
    mock_client = MagicMock()
    monkeypatch.setattr(feature_store, "_get_client", lambda: mock_client)

    with TestClient(app) as c:
        body = c.get("/features/latency?user_id=7").json()

    assert body["redis"] is True
    assert body["user_id"] == 7
    for key in ("ping_ms", "recent_clicks_ms", "recent_views_ms", "session_clicks_ms", "worst_ms"):
        assert isinstance(body[key], (int, float)) and body[key] >= 0
    assert body["worst_ms"] == max(body["ping_ms"], body["recent_clicks_ms"],
                                    body["recent_views_ms"], body["session_clicks_ms"])
    assert "within_spec" in body and "limit_ms" in body


# ─── 실제 Redis 호출 패턴 (MagicMock) ─────────────────────────────────────

def test_record_click_pushes_to_list_and_increments_session(monkeypatch):
    mock_client = MagicMock()
    mock_pipe = MagicMock()
    mock_client.pipeline.return_value = mock_pipe
    monkeypatch.setattr(feature_store, "_get_client", lambda: mock_client)

    feature_store.record_event(user_id=42, event_type="click", item_id=7)

    mock_pipe.lpush.assert_any_call("u:42:recent_clicks", 7)
    mock_pipe.ltrim.assert_any_call("u:42:recent_clicks", 0, feature_store.RECENT_LIMIT - 1)
    mock_pipe.incr.assert_called_once_with("u:42:session_clicks")
    # session_clicks 에 세션 TTL 이 걸려 있어야 함 (30분)
    expire_calls = [call.args for call in mock_pipe.expire.call_args_list]
    assert ("u:42:session_clicks", feature_store.SESSION_TTL_SEC) in expire_calls
    mock_pipe.execute.assert_called_once()


def test_record_view_updates_recent_views_not_session(monkeypatch):
    mock_client = MagicMock()
    mock_pipe = MagicMock()
    mock_client.pipeline.return_value = mock_pipe
    monkeypatch.setattr(feature_store, "_get_client", lambda: mock_client)

    feature_store.record_event(user_id=7, event_type="view", item_id=11)
    # view 는 session_clicks 를 건드리지 않아야 함
    mock_pipe.incr.assert_not_called()
    mock_pipe.lpush.assert_any_call("u:7:recent_views", 11)


def test_record_event_swallows_redis_errors(monkeypatch):
    """Redis 통신 실패 — 서버는 계속. 예외 전파 금지."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()
    mock_pipe.execute.side_effect = RuntimeError("redis down")
    mock_client.pipeline.return_value = mock_pipe
    monkeypatch.setattr(feature_store, "_get_client", lambda: mock_client)

    feature_store.record_event(user_id=1, event_type="click", item_id=5)   # 예외 X


def test_get_recent_clicks_reads_from_redis(monkeypatch):
    mock_client = MagicMock()
    mock_client.lrange.return_value = ["7", "5", "3"]
    monkeypatch.setattr(feature_store, "_get_client", lambda: mock_client)

    assert feature_store.get_recent_clicks(42, k=3) == ["7", "5", "3"]
    mock_client.lrange.assert_called_once_with("u:42:recent_clicks", 0, 2)


def test_get_session_click_count_handles_missing_and_present(monkeypatch):
    mock_client = MagicMock()
    mock_client.get.return_value = None
    monkeypatch.setattr(feature_store, "_get_client", lambda: mock_client)
    assert feature_store.get_session_click_count(1) == 0

    mock_client.get.return_value = "42"
    assert feature_store.get_session_click_count(1) == 42


def test_ping_latency_returns_float_when_redis_available(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(feature_store, "_get_client", lambda: mock_client)
    ms = feature_store.ping_latency_ms()
    assert isinstance(ms, float) and ms >= 0
    mock_client.ping.assert_called_once()


# ─── 통합 — 이벤트 → Redis write 호출 확인 ────────────────────────────────

def test_events_endpoint_calls_feature_store_record(client, monkeypatch):
    """POST /events 성공 시 feature_store.record_event 가 적절한 인자로 호출되는지."""
    calls = []
    monkeypatch.setattr(
        "app.routers.events.feature_store.record_event",
        lambda **kw: calls.append(kw),
    )
    r = client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3})
    assert r.status_code == 201
    assert calls == [{"user_id": 1, "event_type": "click", "item_id": 3, "session_id": None}]


def test_events_endpoint_skips_feature_store_for_is_setup(client, monkeypatch):
    """is_setup=True 는 초기 이력 적재 — 집계 제외 규칙에 따라 Redis 에도 쓰지 않는다."""
    calls = []
    monkeypatch.setattr(
        "app.routers.events.feature_store.record_event",
        lambda **kw: calls.append(kw),
    )
    r = client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3, "is_setup": True})
    assert r.status_code == 201 and calls == []


# ─── 통합 — Recommend 폴백 (Redis 비면 DB) ──────────────────────────────

def test_recommend_falls_back_to_db_when_redis_cold(client):
    """Redis 가 없거나 cold 이면 DB 에서 최근 클릭을 가져와야 한다 (기존 동작 보존)."""
    client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3})
    body = client.get("/recommend/1?k=3").json()
    assert body["session_context"]["recent_clicks"][:1] == ["3"]


def test_recommend_prefers_redis_when_populated(client, monkeypatch):
    """Redis 가 최근 클릭을 돌려주면 그걸 우선 사용 (DB 쿼리 생략)."""
    monkeypatch.setattr(
        "app.routers.recommend.feature_store.get_recent_clicks",
        lambda user_id, k: ["99", "88"],
    )
    body = client.get("/recommend/1?k=3").json()
    assert body["session_context"]["recent_clicks"] == ["99", "88"]


def test_record_event_new_session_id_resets_session_clicks(monkeypatch):
    """session_id가 바뀌면 새 세션 — session_clicks를 지우고 0부터 (이벤트연결규약 2.4)."""
    mock_client = MagicMock()
    mock_client.pipeline.return_value = MagicMock()
    monkeypatch.setattr(feature_store, "_get_client", lambda: mock_client)

    mock_client.get.return_value = b"s1"
    feature_store.record_event(user_id=5, event_type="click", item_id=1, session_id="s1")   # 같은 세션
    mock_client.delete.assert_not_called()
    feature_store.record_event(user_id=5, event_type="click", item_id=2, session_id="s2")   # 새 세션
    mock_client.delete.assert_called_once_with("u:5:session_clicks")
    mock_client.set.assert_any_call("u:5:session_id", "s2", ex=feature_store.SESSION_TTL_SEC)
