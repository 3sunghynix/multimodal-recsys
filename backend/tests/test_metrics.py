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
    assert body == {"since_event_id": 0, "max_event_id": 0, "delta": 0, "reset_detected": False}


def test_log_delta_detects_reset(empty_client):
    """--reset 재적재 후 max_event_id < since_event_id → reset_detected=True.
    runner 가 기준점(마지막 training 의 since_event_id)을 다시 잡는 신호 (PR #48 리뷰)."""
    c, Session = empty_client
    _insert_event(Session, 1, 1, "click", datetime(2026, 11, 1, tzinfo=timezone.utc))
    _insert_event(Session, 1, 1, "click", datetime(2026, 11, 1, tzinfo=timezone.utc))

    body = c.get("/metrics/log-delta?since_event_id=100").json()
    assert body["max_event_id"] == 2
    assert body["since_event_id"] == 100
    assert body["reset_detected"] is True

    body = c.get("/metrics/log-delta?since_event_id=1").json()
    assert body["reset_detected"] is False               # max >= since 이면 정상


# ─── PR #48 리뷰 반영: model_version_id 필터 · hit intersection · tz 정규화 ───

def test_window_filters_by_model_version_id(client):
    """A/B 중 모델별 집계 — model_version_id 로 Event 필터링 (PR #48 1차 리뷰).
    seed_sample 이 ModelVersion id=1(popularity), id=2(brand_popularity) 를 등록하므로 그걸 쓴다."""
    c, Session = client
    t_start = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
    inside = t_start + timedelta(hours=1)

    # 두 모델 각각 2개/1개 view
    with Session() as s:
        s.add(Event(user_id=1, item_id=3, type="view", is_setup=False, created_at=inside, model_version_id=1))
        s.add(Event(user_id=2, item_id=3, type="view", is_setup=False, created_at=inside, model_version_id=1))
        s.add(Event(user_id=3, item_id=3, type="view", is_setup=False, created_at=inside, model_version_id=2))
        s.commit()

    # 필터 없으면 전부 (3건) — seed_sample 의 150 이벤트는 현재 시각(2026-11-12)이라 window 밖
    body = c.get("/metrics/window", params={"start": t_start.isoformat(), "end": t_end.isoformat(),
                                             "model_version": "v1", "source": "all"}).json()
    assert body["views"] == 3

    # 1번 모델만 (2건)
    body = c.get("/metrics/window", params={"start": t_start.isoformat(), "end": t_end.isoformat(),
                                             "model_version": "popularity-v1", "source": "all",
                                             "model_version_id": 1}).json()
    assert body["views"] == 2

    # 2번 모델만 (1건)
    body = c.get("/metrics/window", params={"start": t_start.isoformat(), "end": t_end.isoformat(),
                                             "model_version": "brand_popularity-v1", "source": "all",
                                             "model_version_id": 2}).json()
    assert body["views"] == 1


def test_window_rejects_unknown_model_version_id(client):
    """ModelVersion 테이블에 없는 id → 422 (라벨/id 불일치 사전 차단; PR #48 2차 리뷰)."""
    c, _ = client
    r = c.get("/metrics/window", params={"start": "2026-11-01T00:00:00Z", "end": "2026-11-02T00:00:00Z",
                                          "model_version": "v1", "source": "all", "model_version_id": 9999})
    assert r.status_code == 422
    assert "ModelVersion" in r.json()["detail"]


def test_window_model_version_label_overridden_from_id(client):
    """model_version_id 가 오면 응답의 model_version 은 ModelVersion.name-version 으로 override
    (호출자 라벨 오타 보호; PR #48 2차 리뷰)."""
    c, _ = client
    body = c.get("/metrics/window", params={
        "start": "2026-11-01T00:00:00Z", "end": "2026-11-02T00:00:00Z",
        "model_version": "wrong-label", "source": "all", "model_version_id": 1,
    }).json()
    assert body["model_version"] == "popularity-v1"      # seed_sample 의 id=1
    assert "popularity-v1" in body["window_id"]
    assert "wrong-label" not in body["window_id"]


def test_window_rejects_space_decoded_from_plus(client):
    """'+' 가 curl 에서 공백으로 디코딩돼 서버에 도착하면 422 (문서는 Z 또는 %2B 써야; PR #48 2차 리뷰).
    %20 을 URL 에 직접 박아 '+' 가 깨진 상태를 재현."""
    c, _ = client
    r = c.get("/metrics/window?start=2026-11-01T00:00:00%2000:00"
              "&end=2026-11-02T00:00:00Z&model_version=v1&source=all")
    assert r.status_code == 422
    assert "ISO 8601" in r.json()["detail"]


def test_window_accepts_z_suffix(client):
    """Z 접미사 = +00:00 축약 (Python 3.12 지원, curl 에서 '+' 디코딩 문제 회피)."""
    c, Session = client
    _insert_event(Session, 1, 3, "view",
                   datetime(2026, 11, 1, 12, 0))                     # window 안 (DB 는 naive UTC)
    body = c.get("/metrics/window", params={"start": "2026-11-01T00:00:00Z",
                                             "end": "2026-11-02T00:00:00Z",
                                             "model_version": "v1", "source": "all"}).json()
    assert body["views"] == 1


def test_window_id_normalized_to_utc(client):
    """같은 UTC 순간이면 Z / +00:00 / +09:00 (한국 시각) 로 호출해도 window_id 가 같음
    (PR #48 2차 리뷰 nit — Z/+00:00 로 id 분기 방지)."""
    c, _ = client
    # 2026-11-01 00:00 UTC == 2026-11-01 09:00 KST
    body_z = c.get("/metrics/window", params={
        "start": "2026-11-01T00:00:00Z", "end": "2026-11-02T00:00:00Z",
        "model_version": "v1", "source": "all"}).json()
    body_offset = c.get("/metrics/window", params={
        "start": "2026-11-01T00:00:00+00:00", "end": "2026-11-02T00:00:00+00:00",
        "model_version": "v1", "source": "all"}).json()
    body_kst = c.get("/metrics/window", params={
        "start": "2026-11-01T09:00:00+09:00", "end": "2026-11-02T09:00:00+09:00",
        "model_version": "v1", "source": "all"}).json()
    assert body_z["window_id"] == body_offset["window_id"] == body_kst["window_id"]


def test_window_default_metric_definition_version_is_user_click_through_v1(client):
    """기본 metric_definition_version = user_click_through_v1 (runner HitRate@K 와 혼용 금지 신호;
    PR #48 2차 리뷰 by 광석)."""
    c, _ = client
    body = c.get("/metrics/window", params={
        "start": "2026-11-01T00:00:00Z", "end": "2026-11-02T00:00:00Z",
        "model_version": "v1", "source": "all"}).json()
    assert body["metric_definition_version"] == "user_click_through_v1"


def test_window_hit_users_is_intersection(client):
    """hit_users = viewers ∩ clickers — 노출 없이 클릭만 한 유저는 제외 (PR #48 리뷰 by 유환)."""
    c, Session = client
    t_start = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
    inside = t_start + timedelta(hours=1)

    # user 1: view 만 / user 2: click 만 (검색결과 직행) / user 3: view + click (진짜 hit)
    _insert_event(Session, 1, 3, "view", inside)
    _insert_event(Session, 2, 3, "click", inside)
    _insert_event(Session, 3, 3, "view", inside)
    _insert_event(Session, 3, 3, "click", inside)

    body = c.get("/metrics/window", params={"start": t_start.isoformat(), "end": t_end.isoformat(),
                                             "model_version": "v1", "source": "all"}).json()
    assert body["evaluated_users"] == 2                   # view 한 유저: 1, 3
    assert body["hit_users"] == 1                         # view + click 교집합: 3 만
    assert body["clicks"] == 2                            # click 이벤트 자체는 2건


def test_window_normalizes_timezone_for_consistent_db_comparison(client):
    """같은 UTC 순간을 다른 timezone 표기로 줘도 같은 결과 — DB 비교는 UTC-naive 로 정규화 (PR #48 리뷰)."""
    c, Session = client
    # 2026-11-01 09:00 KST == 2026-11-01 00:00 UTC
    moment_utc = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    _insert_event(Session, 1, 3, "view", moment_utc.replace(tzinfo=None))      # DB 는 naive UTC 저장

    # UTC 로 요청
    body_utc = c.get("/metrics/window", params={
        "start": "2026-11-01T00:00:00+00:00", "end": "2026-11-02T00:00:00+00:00",
        "model_version": "v1", "source": "all"}).json()
    # 같은 범위를 KST 로 요청 (2026-11-01 09:00+09:00 ~ 2026-11-02 09:00+09:00 == UTC 동일 범위)
    body_kst = c.get("/metrics/window", params={
        "start": "2026-11-01T09:00:00+09:00", "end": "2026-11-02T09:00:00+09:00",
        "model_version": "v1", "source": "all"}).json()
    assert body_utc["views"] == body_kst["views"] == 1
