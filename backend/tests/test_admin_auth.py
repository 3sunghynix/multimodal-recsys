"""B4 관리자 쓰기 엔드포인트 인증.

기본 (admin_api_key 비어 있음): 통과 — 기존 테스트 호환 · 로컬 개발 편의.
설정되어 있음: X-Admin-Key 헤더 필수. 없음/오타 → 401. 정확히 일치 → 통과.

검증 대상 쓰기 4개:
    POST /models
    POST /models/{id}/activate
    POST /experiments
    POST /experiments/{id}/stop

읽기(GET /models · /experiments · /experiments/{id}/metrics · /stats)는 보호 안 함 — 대시보드 조회용.
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


@pytest.fixture()
def key(monkeypatch):
    """ADMIN_API_KEY 설정된 상태를 흉내낸다 — 운영 모드."""
    monkeypatch.setattr(settings, "admin_api_key", "test-key-shhh")
    return "test-key-shhh"


# ─── 비활성 (키 비어 있음) — 기존 호환성 ──────────────────────────────────

def test_empty_key_lets_writes_through(client):
    """ADMIN_API_KEY 미설정이면 헤더 없이도 쓰기 성공 (dev · CI · 기존 테스트 호환)."""
    assert settings.admin_api_key == ""                             # 기본값 확인
    r = client.post("/models", json={"name": "popularity", "version": "test-v0", "status": "candidate"})
    assert r.status_code == 201


# ─── 활성 (키 설정) — 헤더 요구 ──────────────────────────────────────────

def test_create_model_rejects_without_key(client, key):
    r = client.post("/models", json={"name": "popularity", "version": "v9", "status": "candidate"})
    assert r.status_code == 401 and "관리자 키" in r.json()["detail"]


def test_create_model_rejects_wrong_key(client, key):
    r = client.post("/models", json={"name": "popularity", "version": "v9", "status": "candidate"},
                    headers={"X-Admin-Key": "wrong"})
    assert r.status_code == 401


def test_create_model_accepts_correct_key(client, key):
    r = client.post("/models", json={"name": "popularity", "version": "v9", "status": "candidate"},
                    headers={"X-Admin-Key": key})
    assert r.status_code == 201


def test_activate_model_requires_key(client, key):
    # 키 켜기 전에 seed 가 만든 모델이 있어야 한다 — seed_sample 은 기본 모델을 심는다.
    existing = client.get("/models").json()
    assert existing, "seed_sample 이 모델을 심어야 한다"
    mid = existing[0]["id"]
    assert client.post(f"/models/{mid}/activate").status_code == 401
    assert client.post(f"/models/{mid}/activate", headers={"X-Admin-Key": key}).status_code == 200


def test_experiments_write_endpoints_require_key(client, key):
    models = client.get("/models").json()
    variants = [{"name": "A", "model_version_id": models[0]["id"], "weight": 1}]
    if len(models) > 1:
        variants.append({"name": "B", "model_version_id": models[1]["id"], "weight": 1})
    else:
        # 한 모델이라도 두 variant 는 같은 id 재사용 가능 (서버는 중복 모델 체크 없음)
        variants.append({"name": "B", "model_version_id": models[0]["id"], "weight": 1})
    body = {"name": "t", "variants": variants}

    # 생성: 키 없으면 401, 있으면 201
    assert client.post("/experiments", json=body).status_code == 401
    r = client.post("/experiments", json=body, headers={"X-Admin-Key": key})
    assert r.status_code == 201
    eid = r.json()["id"]

    # 중지: 키 없으면 401, 있으면 200
    assert client.post(f"/experiments/{eid}/stop").status_code == 401
    assert client.post(f"/experiments/{eid}/stop", headers={"X-Admin-Key": key}).status_code == 200


def test_read_endpoints_stay_public(client, key):
    """GET 은 보호 대상이 아님 — 대시보드가 헤더 없이도 조회 가능."""
    assert client.get("/models").status_code == 200
    assert client.get("/experiments").status_code == 200
