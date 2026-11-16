"""이벤트연결규약 (PR #54) — 추천 응답에 request_id · position · impression_id · source · source_item_id
가 붙고, 요청 1건당 recommend_requests 테이블에 1 row 가 저장되는지 확인.

후속 작업 (11-15):
    · POST /events 에 request_id · position · session_id 추가 + 자동 채움/검증
    · POST /events/batch
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.models import RecommendRequest
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


def test_recommend_response_has_request_id_and_per_item_fields(client):
    """request_id 는 UUID, 상품마다 position (1부터) · impression_id (request_id:position) · source · source_item_id."""
    c, _ = client
    body = c.get("/recommend/1", params={"k": 3}).json()

    req_id = body["request_id"]
    uuid.UUID(req_id)                                    # UUID 로 파싱되는지 (잘못되면 예외)

    assert len(body["recommendations"]) == 3
    for pos, rec in enumerate(body["recommendations"], start=1):
        assert rec["position"] == pos
        assert rec["impression_id"] == f"{req_id}:{pos}"
        assert rec["source"] == "sample"                 # seed_sample 이 쓰는 값
        assert rec["source_item_id"]                     # 비어 있지 않음 (예: "s0")


def test_recommend_writes_row_to_recommend_requests(client):
    """응답의 request_id 로 recommend_requests 테이블에 row 가 저장돼 있어야 한다
    (후속 작업의 /events 역조회 기준)."""
    c, Session = client
    body = c.get("/recommend/1", params={"k": 5}).json()
    req_id = uuid.UUID(body["request_id"])

    with Session() as s:
        row = s.get(RecommendRequest, req_id)
        assert row is not None
        assert row.user_id == 1
        assert row.k == 5
        assert row.item_ids == [int(r["product_id"]) for r in body["recommendations"]]
        # 실험 없을 때 — experiment_id · variant 둘 다 None
        assert row.experiment_id is None and row.variant is None
        # active 모델이 세팅됐으므로 model_version_id 는 채워짐
        assert row.model_version_id is not None


def test_recommend_two_requests_have_distinct_request_ids(client):
    """같은 유저 두 번 호출하면 서로 다른 request_id → DB 에도 두 row."""
    c, Session = client
    r1 = c.get("/recommend/1", params={"k": 2}).json()["request_id"]
    r2 = c.get("/recommend/1", params={"k": 2}).json()["request_id"]
    assert r1 != r2

    with Session() as s:
        rows = s.scalars(select(RecommendRequest).where(RecommendRequest.user_id == 1)).all()
        req_ids = {str(r.request_id) for r in rows}
        assert {r1, r2} <= req_ids


def test_recommend_saves_session_id_to_recommend_requests(client):
    """?session_id= 로 받은 값이 recommend_requests 에 저장돼 세션 단위 분석 가능 (PR #57 2차 리뷰)."""
    c, Session = client
    body = c.get("/recommend/1", params={"k": 2, "session_id": "run:c1:s1"}).json()
    with Session() as s:
        row = s.get(RecommendRequest, uuid.UUID(body["request_id"]))
        assert row.session_id == "run:c1:s1"

    # session_id 없이 호출하면 null 저장 (기존 호출자 호환)
    body2 = c.get("/recommend/1", params={"k": 2}).json()
    with Session() as s:
        row2 = s.get(RecommendRequest, uuid.UUID(body2["request_id"]))
        assert row2.session_id is None


def test_recommend_total_ms_includes_db_commit(client):
    """total_ms 는 recommend_requests commit 까지 포함 (PR #57 2차 리뷰 — PR #49 와 같은 종류 수정).
    candidate/ranking/reranking 합보다 커야 (commit 이 그 뒤니까)."""
    c, _ = client
    body = c.get("/recommend/1", params={"k": 3}).json()
    lat = body["pipeline_latency"]
    stage_sum = lat["candidate_ms"] + lat["ranking_ms"] + lat["reranking_ms"]
    assert lat["total_ms"] >= stage_sum         # 세션 컨텍스트 조회 + DB insert/commit 포함


def test_recommend_under_ab_records_variant_and_experiment(client):
    """A/B 실험이 돌 때 variant · experiment_id 가 recommend_requests 에 기록된다."""
    c, Session = client
    ids = [m["id"] for m in c.get("/models").json()]
    exp = c.post("/experiments", json={"name": "pop vs brand", "variants": [
        {"name": "A", "model_version_id": ids[0], "weight": 1},
        {"name": "B", "model_version_id": ids[1], "weight": 1}]}).json()

    body = c.get("/recommend/3", params={"k": 2}).json()
    assert body["experiment_id"] == exp["id"] and body["variant"] in ("A", "B")

    with Session() as s:
        row = s.get(RecommendRequest, uuid.UUID(body["request_id"]))
        assert row.experiment_id == exp["id"]
        assert row.variant == body["variant"]
        assert row.model_version_id == body["model_version_id"]


# ─── PR #57 ③ POST /events request_id 검증 · 자동 채움 ───────────────────

def _recommend(c, user_id=1, k=3):
    """추천 요청 → (request_id, item_ids, model_version_id, experiment_id, variant) 돌려줌."""
    body = c.get(f"/recommend/{user_id}", params={"k": k}).json()
    return (uuid.UUID(body["request_id"]),
            [int(r["product_id"]) for r in body["recommendations"]],
            body["model_version_id"], body.get("experiment_id"), body.get("variant"))


def test_event_with_request_id_autofills_model_info(client):
    """request_id 만 주고 model_version_id 등을 생략하면 서버가 recommend_requests 역조회로 채움 (톡방 결정 5)."""
    c, _ = client
    req_id, items, mv_id, _, _ = _recommend(c, user_id=1, k=3)

    r = c.post("/events", json={"user_id": 1, "type": "click", "item_id": items[1],
                                 "request_id": str(req_id), "position": 2})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["model_version_id"] == mv_id          # 서버가 자동 채움
    assert body["request_id"] == str(req_id)
    assert body["position"] == 2


def test_event_with_matching_model_info_passes(client):
    """request_id + 일치하는 model_version_id 명시 → 그대로 저장 (덮어쓰지 않음)."""
    c, _ = client
    req_id, items, mv_id, _, _ = _recommend(c, user_id=1, k=2)
    r = c.post("/events", json={"user_id": 1, "type": "view", "item_id": items[0],
                                 "request_id": str(req_id), "position": 1,
                                 "model_version_id": mv_id})
    assert r.status_code == 201 and r.json()["model_version_id"] == mv_id


def test_event_with_mismatched_model_version_id_rejected(client):
    """request_id 서버 기록과 model_version_id 가 다르면 422 (톡방 결정 5)."""
    c, _ = client
    req_id, items, mv_id, _, _ = _recommend(c, user_id=1, k=2)
    r = c.post("/events", json={"user_id": 1, "type": "click", "item_id": items[0],
                                 "request_id": str(req_id), "position": 1,
                                 "model_version_id": mv_id + 999})
    assert r.status_code == 422 and "model_version_id" in r.json()["detail"]


def test_event_with_mismatched_user_id_rejected(client):
    """다른 유저의 request_id 가로채면 422 (보안 이슈)."""
    c, _ = client
    req_id, items, _, _, _ = _recommend(c, user_id=1, k=2)
    r = c.post("/events", json={"user_id": 2, "type": "click", "item_id": items[0],
                                 "request_id": str(req_id), "position": 1})
    assert r.status_code == 422 and "user_id" in r.json()["detail"]


def test_event_with_unknown_request_id_rejected(client):
    """recommend_requests 에 없는 request_id → 422."""
    c, _ = client
    r = c.post("/events", json={"user_id": 1, "type": "click", "item_id": 3,
                                 "request_id": str(uuid.uuid4())})
    assert r.status_code == 422 and "recommend_requests" in r.json()["detail"]


def test_event_position_validated_against_item_ids(client):
    """position 명시 시 그 자리 상품이 item_id 와 같아야 함; 다르면 422."""
    c, _ = client
    req_id, items, _, _, _ = _recommend(c, user_id=1, k=3)
    # items[0] 을 position=2 로 보내면 미스매치
    r = c.post("/events", json={"user_id": 1, "type": "click", "item_id": items[0],
                                 "request_id": str(req_id), "position": 2})
    assert r.status_code == 422 and ("position" in r.json()["detail"] or "불일치" in r.json()["detail"])


def test_event_position_autofilled_from_item_id(client):
    """position 생략 + item_id 가 item_ids 안에 있으면 서버가 position 자동 채움."""
    c, _ = client
    req_id, items, _, _, _ = _recommend(c, user_id=1, k=3)
    r = c.post("/events", json={"user_id": 1, "type": "click", "item_id": items[2],
                                 "request_id": str(req_id)})
    assert r.status_code == 201 and r.json()["position"] == 3


def test_event_without_request_id_still_works(client):
    """기존 호출 (request_id 없음) 그대로 작동 — 호환성."""
    c, _ = client
    r = c.post("/events", json={"user_id": 1, "type": "click", "item_id": 3})
    assert r.status_code == 201 and r.json()["request_id"] is None


# ─── PR #57 ④ POST /events/batch ─────────────────────────────────────────

def test_events_batch_creates_all_valid(client):
    """유효한 3건 → 모두 created."""
    c, _ = client
    body = c.post("/events/batch", json={"events": [
        {"user_id": 1, "type": "click", "item_id": 3},
        {"user_id": 1, "type": "click", "item_id": 4},
        {"user_id": 2, "type": "view", "item_id": 3},
    ]}).json()
    assert body["created"] == 3 and body["duplicate"] == 0 and body["error"] == 0
    assert all(r["status"] == "created" for r in body["results"])


def test_events_batch_partial_success_keeps_valid_ones(client):
    """3건 중 1건이 검증 실패여도 나머지 2건은 저장 (톡방 결정 2)."""
    c, _ = client
    body = c.post("/events/batch", json={"events": [
        {"user_id": 1, "type": "click", "item_id": 3},
        {"user_id": 999, "type": "click", "item_id": 3},            # 없는 고객 → 404
        {"user_id": 1, "type": "click", "item_id": 4},
    ]}).json()
    assert body["created"] == 2 and body["error"] == 1
    assert body["results"][1]["status"] == "error" and body["results"][1]["code"] == 404
    assert body["results"][0]["status"] == "created"
    assert body["results"][2]["status"] == "created"


def test_events_batch_duplicate_event_id(client):
    """같은 event_id 로 저장된 이벤트가 있으면 duplicate (id · event_id 반환)."""
    c, _ = client
    eid = "6b1c7d9e-0000-4000-8000-000000000001"
    first = c.post("/events", json={"user_id": 1, "type": "click", "item_id": 3, "event_id": eid}).json()
    body = c.post("/events/batch", json={"events": [
        {"user_id": 1, "type": "click", "item_id": 3, "event_id": eid},        # 멱등 같은 내용
        {"user_id": 1, "type": "view",  "item_id": 3, "event_id": eid},        # 다른 내용 → 409
    ]}).json()
    assert body["duplicate"] == 1 and body["error"] == 1
    assert body["results"][0]["status"] == "duplicate" and body["results"][0]["id"] == first["id"]
    assert body["results"][1]["status"] == "error" and body["results"][1]["code"] == 409


def test_events_batch_rejects_over_1000(client):
    """1,001 건이면 Pydantic 이 422 로 거절 (max_length)."""
    c, _ = client
    r = c.post("/events/batch", json={
        "events": [{"user_id": 1, "type": "click", "item_id": 3}] * 1001})
    assert r.status_code == 422


def test_events_batch_autofills_from_request_id(client):
    """배치 안에서 request_id 로 model_version_id 자동 채움 — 배치 내 distinct request_id 는 한 번에 fetch (N+1 방지)."""
    c, _ = client
    req_id, items, mv_id, _, _ = _recommend(c, user_id=1, k=3)
    body = c.post("/events/batch", json={"events": [
        {"user_id": 1, "type": "view",  "item_id": items[0], "request_id": str(req_id)},
        {"user_id": 1, "type": "click", "item_id": items[1], "request_id": str(req_id), "position": 2},
    ]}).json()
    assert body["created"] == 2
    # 저장된 각 이벤트의 model_version_id 가 서버 기록으로 채워졌는지는 /events/count 로는 못 보지만,
    # 결과에 id 가 반환되면 저장 자체는 성공. 상세 검증은 아래 통합 테스트에서.


def test_events_batch_rejects_empty(client):
    """빈 배치는 422 (min_length=1)."""
    c, _ = client
    r = c.post("/events/batch", json={"events": []})
    assert r.status_code == 422


# ─── PR #59 리뷰 반영 (유환) ────────────────────────────────────────────

def test_event_autofill_then_resend_is_idempotent(client):
    """request_id 로 model_version_id 자동 채움 후, 같은 event_id · 같은 생략 상태로 재전송하면
    멱등 통과 (409 안 남; PR #59 리뷰 by 유환). 자동 채움을 멱등 비교 전에 적용해야 저장된
    (채워진) 값과 incoming(채워진) 값이 같아진다."""
    c, _ = client
    req_id, items, _mv_id, _, _ = _recommend(c, user_id=1, k=3)
    eid = "7b2c8e9f-0000-4000-8000-000000000011"

    # 1차: model_version_id 생략 + request_id 만 → 서버가 자동 채움해 저장
    r1 = c.post("/events", json={"user_id": 1, "type": "click", "item_id": items[1],
                                  "request_id": str(req_id), "position": 2, "event_id": eid})
    assert r1.status_code == 201

    # 2차: 같은 생략 상태로 재전송 → 서버가 또 자동 채움 → 비교 통과 → 200 멱등
    r2 = c.post("/events", json={"user_id": 1, "type": "click", "item_id": items[1],
                                  "request_id": str(req_id), "position": 2, "event_id": eid})
    assert r2.status_code == 200 and r2.json()["id"] == r1.json()["id"]


def test_events_batch_in_batch_duplicate_event_id_doesnt_rollback_others(client):
    """배치 안에 같은 event_id 가 두 번 있으면 첫 건은 created, 두 번째는 duplicate(같은 내용)
    또는 error(409, 다른 내용). 그리고 **전체 롤백 없이** 다른 valid 건은 그대로 저장됨
    (톡방 결정 2, PR #59 리뷰)."""
    c, _ = client
    eid_dup = "7b2c8e9f-0000-4000-8000-000000000021"
    eid_other = "7b2c8e9f-0000-4000-8000-000000000022"
    body = c.post("/events/batch", json={"events": [
        {"user_id": 1, "type": "click", "item_id": 3, "event_id": eid_dup},    # 1번째 — created
        {"user_id": 1, "type": "click", "item_id": 3, "event_id": eid_dup},    # 2번째 — 같은 내용 → duplicate
        {"user_id": 1, "type": "view",  "item_id": 3, "event_id": eid_dup},    # 3번째 — 다른 내용 → error 409
        {"user_id": 1, "type": "click", "item_id": 4, "event_id": eid_other},  # 4번째 — 정상 → created
    ]}).json()
    assert body["created"] == 2 and body["duplicate"] == 1 and body["error"] == 1
    assert body["results"][0]["status"] == "created"
    assert body["results"][1]["status"] == "duplicate" and body["results"][1]["id"] == body["results"][0]["id"]
    assert body["results"][2]["status"] == "error" and body["results"][2]["code"] == 409
    assert body["results"][3]["status"] == "created"       # 중복이 전체 롤백 유발 안 함


def test_events_batch_autofill_then_resend_mix(client):
    """배치: autofill 저장 → 같은 event_id · 같은 생략 상태로 재전송된 2번째는 duplicate (PR #59 리뷰)."""
    c, _ = client
    req_id, items, _mv_id, _, _ = _recommend(c, user_id=1, k=3)
    eid = "7b2c8e9f-0000-4000-8000-000000000031"
    body = c.post("/events/batch", json={"events": [
        {"user_id": 1, "type": "click", "item_id": items[0],
         "request_id": str(req_id), "position": 1, "event_id": eid},          # autofill 적용 저장
        {"user_id": 1, "type": "click", "item_id": items[0],
         "request_id": str(req_id), "position": 1, "event_id": eid},          # 같은 생략 상태 재전송
    ]}).json()
    assert body["created"] == 1 and body["duplicate"] == 1 and body["error"] == 0
