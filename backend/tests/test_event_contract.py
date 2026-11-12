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
