"""API 전체 흐름 확인: 검색 → 추천 → 로그 → 모델 교체 → A/B 할당 · 그룹별 지표. SQLite 임시 DB 사용.

CLIP 모델 로드는 몇 초가 걸리고 테스트와 무관하므로 settings.load_image_encoder를 False로 끈다.
그 상태에서 /search/image 는 503을 반환해야 한다 (vs 운영은 200 + 유사 상품 리스트)."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.services.ab import assign_variant
from scripts.seed_sample import seed


@pytest.fixture(autouse=True)
def _no_clip_during_tests(monkeypatch):
    """CLIP 로드를 끈다. lifespan이 모델을 가져오지 않아 테스트가 가볍다."""
    monkeypatch.setattr(settings, "load_image_encoder", False)


@pytest.fixture()
def client(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    with Session() as s:
        seed(s)   # 샘플은 image_embedding을 채우지 않는다. 실제 데모는 load_dev_subset으로 돈다

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["image_encoder"] is None   # 테스트에선 CLIP 비활성


def test_search_text_and_image(client):
    r = client.get("/search", params={"q": "Alpha dress"})
    assert r.status_code == 200 and all("Alpha" in i["title"] and "Dress" in i["title"] for i in r.json())
    # 형식(415)은 인코더 상태와 무관하게 항상 먼저 거절된다
    r = client.post("/search/image", files={"file": ("a.txt", b"nope", "text/plain")})
    assert r.status_code == 415
    # 올바른 형식이지만 인코더 비활성(테스트) → 503. 운영은 200 + 유사 상품 리스트.
    r = client.post("/search/image", files={"file": ("a.jpg", b"fake", "image/jpeg")})
    assert r.status_code == 503


def test_recommend_uses_active_model(client):
    r = client.get("/recommend/1", params={"k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "popularity" and len(body["items"]) == 5 and body["variant"] is None


def test_event_logging_and_validation(client):
    assert client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3}).status_code == 201
    assert client.post("/events", json={"user_id": 1, "type": "click"}).status_code == 422
    assert client.post("/events", json={"user_id": 999, "type": "search", "query": "x"}).status_code == 404


def test_activate_switches_model(client):
    models = client.get("/models").json()
    brand = next(m for m in models if m["name"] == "brand_popularity")
    assert client.post(f"/models/{brand['id']}/activate").json()["status"] == "active"
    statuses = {m["name"]: m["status"] for m in client.get("/models").json()}
    assert statuses == {"popularity": "candidate", "brand_popularity": "active"}
    assert client.get("/recommend/2", params={"k": 5}).json()["model"] == "brand_popularity"


def test_ab_assignment_and_metrics(client):
    ids = [m["id"] for m in client.get("/models").json()]
    exp = client.post("/experiments", json={"name": "pop vs brand", "variants": [
        {"name": "A", "model_version_id": ids[0], "weight": 1}, {"name": "B", "model_version_id": ids[1], "weight": 1}]}).json()
    seen = {}
    for u in range(1, 11):
        r = client.get(f"/recommend/{u}", params={"k": 3}).json()
        assert r["experiment_id"] == exp["id"] and r["variant"] in ("A", "B")
        assert client.get(f"/recommend/{u}", params={"k": 3}).json()["variant"] == r["variant"]   # 항상 같은 그룹
        seen[r["variant"]] = seen.get(r["variant"], 0) + 1
        for it in r["items"]:
            client.post("/events", json={"user_id": u, "type": "view", "item_id": it,
                                         "experiment_id": exp["id"], "variant": r["variant"]})
        client.post("/events", json={"user_id": u, "type": "click", "item_id": r["items"][0],
                                     "experiment_id": exp["id"], "variant": r["variant"]})
    m = client.get(f"/experiments/{exp['id']}/metrics").json()
    assert sum(x["impressions"] for x in m) == 30 and sum(x["clicks"] for x in m) == 10
    assert all(abs(x["ctr"] - 1 / 3) < 1e-9 for x in m)


def test_assign_variant_split_is_balanced():
    vs = [{"name": "A", "weight": 1}, {"name": "B", "weight": 1}]
    a = sum(assign_variant(u, 7, vs)["name"] == "A" for u in range(10_000))
    assert 4_700 < a < 5_300
