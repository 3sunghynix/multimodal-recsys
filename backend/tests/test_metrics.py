"""Continuous Training 지원 엔드포인트 — 명세 4.6.

MetricSnapshot CRUD: 평가 결과 저장/조회
GET /metrics/window: 광석 monitoring observe() 가 요구하는 window JSON 포맷 (timezone 필수, setup_excluded=True 보장)
GET /metrics/log-delta: since_event_id 이후 쌓인 non-setup 이벤트 수
"""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.models import Event, Item, User
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
    yield TestClient(app), Session
    app.dependency_overrides.clear()


@pytest.fixture()
def empty_client(tmp_path):
    """시드 데이터 없음 — log-delta 등 "순수 count" 테스트에서 seed_sample 의 150 이벤트 간섭을 피한다."""
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    # 테스트에서 이벤트가 user_id · item_id FK 를 걸려면 최소한 user/item 하나씩 필요
    from app.models import Item, User
    with Session() as s:
        s.add(Item(source="sample", source_item_id="s0", title="t"))
        s.add(User(source="sample", source_user_id="u0"))
        s.commit()

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    yield TestClient(app), Session
    app.dependency_overrides.clear()


def _insert_event(Session, user_id, item_id, type_, when, is_setup=False):
    """테스트용 — created_at 을 직접 지정해 window 테스트 (ORM default=now() 는 테스트에서 통제 불가)."""
    with Session() as s:
        ev = Event(user_id=user_id, item_id=item_id, type=type_, is_setup=is_setup, created_at=when)
        s.add(ev)
        s.commit()


# ─── MetricSnapshot CRUD ──────────────────────────────────────────────────

def test_create_and_list_snapshot(client):
    c, _ = client
    r = c.post("/metrics/snapshots", json={"model_version_id": 1, "metric": "recall@50", "value": 0.23, "n": 500})
    assert r.status_code == 201
    body = r.json()
    assert body["metric"] == "recall@50" and body["value"] == 0.23 and body["n"] == 500
    assert "id" in body and "created_at" in body

    rows = c.get("/metrics/snapshots").json()
    assert len(rows) == 1 and rows[0]["id"] == body["id"]


def test_list_snapshots_filter(client):
    c, _ = client
    c.post("/metrics/snapshots", json={"model_version_id": 1, "metric": "recall@50", "value": 0.2})
    c.post("/metrics/snapshots", json={"model_version_id": 2, "metric": "recall@50", "value": 0.3})
    c.post("/metrics/snapshots", json={"model_version_id": 1, "metric": "ndcg@10", "value": 0.4})

    m1 = c.get("/metrics/snapshots?model_version_id=1").json()
    assert {r["metric"] for r in m1} == {"recall@50", "ndcg@10"}

    r50 = c.get("/metrics/snapshots?metric=recall@50").json()
    assert {r["value"] for r in r50} == {0.2, 0.3}


# ─── GET /metrics/window — observe() 포맷 ───────────────────────────────

def test_window_requires_timezone_in_iso(client):
    c, _ = client
    r = c.get("/metrics/window", params={"start": "2026-11-01T00:00:00", "end": "2026-11-02T00:00:00+00:00",
                                        "model_version": "v1", "source": "all"})
    assert r.status_code == 422 and "타임존" in r.json()["detail"]


def test_window_rejects_non_monotonic(client):
    c, _ = client
    r = c.get("/metrics/window", params={"start": "2026-11-02T00:00:00+00:00", "end": "2026-11-01T00:00:00+00:00",
                                        "model_version": "v1", "source": "all"})
    assert r.status_code == 422


def test_window_aggregates_views_clicks_users(client):
    """window 안의 view 3 (distinct user 2) + click 1 (distinct user 1) — 집계 확인."""
    c, Session = client
    t_start = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
    inside = t_start + timedelta(hours=12)

    _insert_event(Session, 1, 3, "view", inside)
    _insert_event(Session, 1, 3, "view", inside)
    _insert_event(Session, 2, 3, "view", inside)
    _insert_event(Session, 1, 3, "click", inside)

    body = c.get("/metrics/window", params={"start": t_start.isoformat(), "end": t_end.isoformat(),
                                             "model_version": "popularity-v1", "source": "all"}).json()
    assert body["views"] == 3
    assert body["clicks"] == 1
    assert body["evaluated_users"] == 2                 # view 한 distinct user
    assert body["hit_users"] == 1                       # click 한 distinct user
    assert body["setup_excluded"] is True
    assert body["model_version"] == "popularity-v1"
    assert body["window_start"] == t_start.isoformat()
    assert body["window_end"] == t_end.isoformat()


def test_window_excludes_is_setup_events(client):
    """is_setup=True 이벤트는 window 집계에서 빠져야 한다 (PR #30 규칙)."""
    c, Session = client
    t_start = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
    inside = t_start + timedelta(hours=1)

    _insert_event(Session, 1, 3, "view", inside)
    _insert_event(Session, 1, 3, "view", inside, is_setup=True)     # 초기 이력 — 제외돼야

    body = c.get("/metrics/window", params={"start": t_start.isoformat(), "end": t_end.isoformat(),
                                             "model_version": "v1", "source": "all"}).json()
    assert body["views"] == 1                            # setup 은 세지 않음


def test_window_excludes_events_outside_window(client):
    """window 경계 밖 이벤트는 집계되지 않아야 한다."""
    c, Session = client
    t_start = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)

    _insert_event(Session, 1, 3, "view", t_start - timedelta(hours=1))   # before
    _insert_event(Session, 1, 3, "view", t_end + timedelta(hours=1))     # after
    _insert_event(Session, 1, 3, "view", t_start + timedelta(hours=1))   # inside

    body = c.get("/metrics/window", params={"start": t_start.isoformat(), "end": t_end.isoformat(),
                                             "model_version": "v1", "source": "all"}).json()
    assert body["views"] == 1


def test_window_cumulative_counts_up_to_end(client):
    """cumulative_new_logs 는 window_end 시점까지 전체 비 setup 이벤트 수 (단조 증가 보장)."""
    c, Session = client
    t_start = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)

    # 1 before start + 2 inside + 1 after end (cumulative 는 after 는 안 셈)
    _insert_event(Session, 1, 3, "view", t_start - timedelta(days=5))
    _insert_event(Session, 1, 3, "view", t_start + timedelta(hours=1))
    _insert_event(Session, 1, 3, "click", t_start + timedelta(hours=2))
    _insert_event(Session, 1, 3, "view", t_end + timedelta(hours=1))

    body = c.get("/metrics/window", params={"start": t_start.isoformat(), "end": t_end.isoformat(),
                                             "model_version": "v1", "source": "all"}).json()
    assert body["cumulative_new_logs"] == 3             # before + inside, after 제외


# ─── GET /metrics/log-delta — 재학습 트리거 판단 ────────────────────────

def test_log_delta_counts_since_id(empty_client):
    """since_event_id 이후로 쌓인 비 setup 이벤트 수 — runner 가 임계값(10000 등) 비교에 쓴다."""
    c, Session = empty_client
    _insert_event(Session, 1, 1, "click", datetime(2026, 11, 1, tzinfo=timezone.utc))
    _insert_event(Session, 1, 1, "click", datetime(2026, 11, 1, tzinfo=timezone.utc))
    _insert_event(Session, 1, 1, "click", datetime(2026, 11, 1, tzinfo=timezone.utc), is_setup=True)
    _insert_event(Session, 1, 1, "click", datetime(2026, 11, 1, tzinfo=timezone.utc))

    body = c.get("/metrics/log-delta?since_event_id=0").json()
    assert body["delta"] == 3                           # setup 1개 제외
    assert body["max_event_id"] == 4
    assert body["since_event_id"] == 0

    body = c.get("/metrics/log-delta?since_event_id=2").json()
    assert body["delta"] == 1                           # id 2 초과 중 비 setup 1개


def test_log_delta_empty_db(empty_client):
    c, _ = empty_client
    body = c.get("/metrics/log-delta?since_event_id=0").json()
    assert body == {"since_event_id": 0, "max_event_id": 0, "delta": 0}
