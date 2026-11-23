"""서버 연결 2단계 (A1) adapter · driver — mocked HTTP 로 로직만 검증.
실제 서버 end-to-end 는 별도 환경에서 수동 확인 (README 예시).
"""
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from simulator.server_adapter import RecommendResponse, ServerClient
from simulator.server_run import _build_impression_events, run


class _FakeResponse:
    """urlopen context manager 흉내 — body 바이트를 돌려준다."""
    def __init__(self, body):
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _urlopen_sequence(*bodies):
    """순서대로 돌려 줄 body 목록. 각 호출마다 다음 응답을 꺼낸다."""
    it = iter(bodies)
    return lambda req, timeout: _FakeResponse(next(it))


# ─── adapter: 결정론 · 기본 플로우 ────────────────────────────────────────

def test_event_id_is_deterministic():
    """같은 (run_id, 고객, 세션, impression_key) → 같은 UUID. 재전송해도 서버가 멱등 처리."""
    c = ServerClient("http://x")
    a = c.event_id("run-1", 0, 0, "p1:view")
    b = c.event_id("run-1", 0, 0, "p1:view")
    assert a == b
    # 하나라도 바뀌면 다른 UUID
    assert c.event_id("run-1", 0, 0, "p1:click") != a
    assert c.event_id("run-2", 0, 0, "p1:view") != a
    assert c.event_id("run-1", 1, 0, "p1:view") != a
    # 포맷 검증
    uuid.UUID(a)


def test_ensure_customer_uses_deterministic_source_user_id():
    """source_user_id 는 uuid5(run_id, "customer", idx) — 같은 run_id · idx 재호출시 같은 값."""
    captured_bodies = []

    def fake_urlopen(req, timeout):
        captured_bodies.append(json.loads(req.data.decode()))
        return _FakeResponse({"id": 42, "source": "sim", "source_user_id": captured_bodies[-1]["source_user_id"],
                              "persona": None})

    c = ServerClient("http://x")
    with patch("simulator.server_adapter.urlopen", side_effect=fake_urlopen):
        uid1 = c.ensure_customer("run-A", 7)
        uid2 = c.ensure_customer("run-A", 7)                        # 재호출 — 멱등 보장
        uid3 = c.ensure_customer("run-A", 8)                        # 다른 idx
        uid4 = c.ensure_customer("run-B", 7)                        # 다른 run_id

    assert uid1 == uid2 == 42
    assert captured_bodies[0]["source_user_id"] == captured_bodies[1]["source_user_id"]   # 같은 키 멱등
    assert captured_bodies[0]["source_user_id"] != captured_bodies[2]["source_user_id"]   # idx 바뀌면 다른 키
    assert captured_bodies[0]["source_user_id"] != captured_bodies[3]["source_user_id"]   # run_id 바뀌면 다른 키


def test_recommend_parses_impressions_and_validates_positions():
    """recommendations[].position 이 1..k 연속이 아니면 거절. 서버 계약 사전 검증."""
    good = {
        "request_id": "req-123", "user_id": 5, "model": "ume", "model_version_id": 2,
        "experiment_id": None, "variant": None,
        "recommendations": [
            {"product_id": "10", "position": 1},
            {"product_id": "20", "position": 2},
            {"product_id": "30", "position": 3},
        ],
    }
    c = ServerClient("http://x")
    with patch("simulator.server_adapter.urlopen", _urlopen_sequence(good)):
        rec = c.recommend(user_id=5, session_id="s1", k=3)
    assert isinstance(rec, RecommendResponse)
    assert rec.request_id == "req-123"
    assert rec.impressions == [(1, 10), (2, 20), (3, 30)]

    # position 이 빠진 경우 거절
    bad = {**good, "recommendations": [{"product_id": "10", "position": 1}, {"product_id": "30", "position": 3}]}
    with patch("simulator.server_adapter.urlopen", _urlopen_sequence(bad)):
        with pytest.raises(RuntimeError, match="positions"):
            c.recommend(user_id=5, session_id="s1", k=2)


def test_recommend_rejects_user_id_mismatch():
    """서버가 다른 user_id 응답을 돌려주면 즉시 중단 (예: 라우팅 버그)."""
    bad = {"request_id": "req-1", "user_id": 999, "recommendations": []}
    c = ServerClient("http://x")
    with patch("simulator.server_adapter.urlopen", _urlopen_sequence(bad)):
        with pytest.raises(RuntimeError, match="user_id"):
            c.recommend(user_id=5, session_id="s1", k=3)


def test_send_events_batch_rejects_over_1000():
    c = ServerClient("http://x")
    with pytest.raises(ValueError, match="1,000건"):
        c.send_events_batch([{"user_id": 1}] * 1001)


# ─── driver: 이벤트 구성 · 집계 ──────────────────────────────────────────

def test_build_impression_events_covers_all_positions_with_unique_event_ids():
    c = ServerClient("http://x")
    rec = RecommendResponse(request_id="req-xyz", model="ume", model_version_id=1,
                             experiment_id=None, variant=None,
                             impressions=[(1, 100), (2, 200), (3, 300)])
    events = _build_impression_events(c, rec, run_id="r1", customer_idx=0, session_idx=0,
                                       user_id=5, session_id="s0")
    assert len(events) == 3
    assert all(e["type"] == "view" for e in events)
    assert [e["position"] for e in events] == [1, 2, 3]
    assert [e["item_id"] for e in events] == [100, 200, 300]
    assert all(e["request_id"] == "req-xyz" for e in events)
    # event_id 는 서로 달라야 서버에서 중복이 아니라 다른 이벤트로 저장됨
    assert len({e["event_id"] for e in events}) == 3


def test_run_end_to_end_mocked_aggregates_counts():
    """driver 가 recommend + batch 응답을 집계해서 돌려주는지 — 실제 서버 없이 flow 만 검증."""
    recommend_response = {
        "request_id": "req-1", "user_id": 1, "model": "ume", "model_version_id": 1,
        "experiment_id": None, "variant": None,
        "recommendations": [{"product_id": str(i), "position": i} for i in (1, 2)],
    }
    created_response = {"results": [], "created": 2, "duplicate": 0, "error": 0}

    # 호출 순서 — batch 는 모든 고객 세션 돌고 끝에 1회 flush (default batch_size=1000):
    #   POST /users (c0), GET /recommend (c0 s0), POST /users (c1), GET /recommend (c1 s0),
    #   POST /events/batch (flush 4 events)
    created_response = {"results": [], "created": 4, "duplicate": 0, "error": 0}
    responses = [{"id": 1}, recommend_response, {"id": 1}, recommend_response, created_response]

    with patch("simulator.server_adapter.urlopen", _urlopen_sequence(*responses)):
        summary = run("http://fake", customers=2, sessions=1, k=2, run_id="test-run",
                       click_rate=0.0, purchase_rate=0.0, seed=0)

    assert summary["customers"] == 2 and summary["sessions"] == 1
    assert summary["recommend_calls"] == 2
    assert summary["events_created"] == 4         # 2 customers × 2 impressions
    assert summary["events_error"] == 0


def test_run_rerun_with_same_run_id_hits_duplicates():
    """같은 run_id 재실행 → 서버는 이미 저장된 event_id 들을 duplicate 로 돌려줘야 (멱등 보장).
    driver 는 duplicate 수를 그대로 집계한다."""
    recommend_response = {
        "request_id": "req-1", "user_id": 1, "model": "ume", "model_version_id": 1,
        "experiment_id": None, "variant": None,
        "recommendations": [{"product_id": "10", "position": 1}],
    }
    # 두 번째 run 은 batch 가 duplicate 2건으로 응답
    duplicate_response = {"results": [], "created": 0, "duplicate": 1, "error": 0}
    responses = [{"id": 1}, recommend_response, duplicate_response]

    with patch("simulator.server_adapter.urlopen", _urlopen_sequence(*responses)):
        summary = run("http://fake", customers=1, sessions=1, k=1, run_id="replay",
                       click_rate=0.0, purchase_rate=0.0, seed=0)

    assert summary["events_created"] == 0 and summary["events_duplicate"] == 1
