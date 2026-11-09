"""/items 라우터 — PR #21 유보 항목(순서 보존 · 400 · 404) + /items/version (광석 PR #29 리뷰에서 제안)."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.models import Item
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


@pytest.fixture()
def session_factory(tmp_path):
    """/items/version 의 count/max_id 가 삽입 후 바뀌는 걸 확인하기 위해, override 밖에서
    직접 DB 를 만지는 테스트용 세션 공장."""
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'ver.db'}")
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
    yield Session
    app.dependency_overrides.clear()


# ─── GET /items 배치 조회 ────────────────────────────────────────────────────

def test_batch_preserves_input_order(client):
    """검색/추천 결과 순서가 중요 — 입력 id 순서 그대로 돌려줘야 한다."""
    body = client.get("/items", params={"ids": "3,1,2"}).json()
    assert [row["id"] for row in body] == [3, 1, 2]


def test_batch_skips_missing_ids_without_error(client):
    """일부 id 가 없으면 해당 id 는 건너뛰고, 남은 결과는 순서 유지."""
    body = client.get("/items", params={"ids": "3,999999,1"}).json()
    assert [row["id"] for row in body] == [3, 1]


def test_batch_400_on_non_integer_ids(client):
    r = client.get("/items", params={"ids": "3,abc,2"})
    assert r.status_code == 400 and "정수" in r.json()["detail"]


def test_batch_400_on_too_many_ids(client):
    too_many = ",".join(str(i) for i in range(501))
    r = client.get("/items", params={"ids": too_many})
    assert r.status_code == 400 and "500" in r.json()["detail"]


def test_batch_empty_ids_returns_empty_list(client):
    """ids 가 빈 문자열 또는 쉼표만 → 404 가 아니라 빈 배열 (호출자 쪽 특수 처리 부담 줄임)."""
    assert client.get("/items", params={"ids": ""}).json() == []
    assert client.get("/items", params={"ids": ",,"}).json() == []


# ─── GET /items/{item_id} 단건 조회 ─────────────────────────────────────────

def test_single_404_on_missing(client):
    r = client.get("/items/999999")
    assert r.status_code == 404 and "없음" in r.json()["detail"]


def test_single_returns_detail_schema(client):
    """존재하면 상세 필드(description · source_item_id · brand_name)까지 포함."""
    body = client.get("/items/1").json()
    assert body["id"] == 1
    assert "description" in body and "source_item_id" in body and "brand_name" in body


# ─── GET /items/version ────────────────────────────────────────────────────

def test_version_returns_count_and_max_id(client):
    body = client.get("/items/version").json()
    assert set(body) == {"count", "max_id"}
    assert body["count"] == 30                               # seed_sample 이 심는 상품 수
    assert body["max_id"] == 30


def test_version_changes_after_insert(session_factory):
    """상품 추가 후 count · max_id 가 모두 변해야 한다 — 시뮬레이터 bridge 가 변화 감지에 쓴다."""
    client = TestClient(app)
    before = client.get("/items/version").json()

    with session_factory() as s:
        s.add(Item(source="sample", source_item_id="new-1", title="newly added", category="test"))
        s.commit()

    after = client.get("/items/version").json()
    assert after["count"] == before["count"] + 1
    assert after["max_id"] == before["max_id"] + 1


def test_version_empty_db_returns_zero(tmp_path):
    """상품이 하나도 없을 때 max_id 는 0 (null 아님) — 호출자 int 비교 안전."""
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    try:
        body = TestClient(app).get("/items/version").json()
        assert body == {"count": 0, "max_id": 0}
    finally:
        app.dependency_overrides.clear()
