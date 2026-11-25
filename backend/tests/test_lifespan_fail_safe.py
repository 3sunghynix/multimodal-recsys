"""D1: 검색 인코더 로드 실패 시에도 api 는 뜨고, 검색만 503 으로 응답한다.

과거 (PR #63 이전) 는 HF 다운로드 실패 · SSL 검사 차단 · 디스크 부족 시 lifespan 에서 예외가 올라가
`api Exited (3)` 로 컨테이너가 죽었다. D1 에서는 try/except 로 포획하고 인코더만 비활성화한다.
"""
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from scripts.seed_sample import seed


@pytest.fixture()
def seeded_db(tmp_path):
    """TestClient lifespan 이 실제로 돌도록 DB 를 미리 마련해 둔다."""
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
    try:
        yield
    finally:
        app.dependency_overrides.clear()


@contextmanager
def _lifespan_with(monkeypatch, *, image_load=False, text_load=False,
                   image_fail: Exception | None = None, text_fail: Exception | None = None):
    """TestClient 로 lifespan 을 실제 실행하면서 load_encoder/load_text_encoder 호출만 가로챈다."""
    monkeypatch.setattr(settings, "load_image_encoder", image_load)
    monkeypatch.setattr(settings, "load_text_encoder", text_load)
    if image_load:
        import app.encoders.clip_image as clip_image
        monkeypatch.setattr(clip_image, "load_encoder",
                            (lambda *a, **k: (_ for _ in ()).throw(image_fail)) if image_fail
                            else clip_image.load_encoder)
    if text_load:
        import app.encoders.text as text_mod
        monkeypatch.setattr(text_mod, "load_text_encoder",
                            (lambda *a, **k: (_ for _ in ()).throw(text_fail)) if text_fail
                            else text_mod.load_text_encoder)
    with TestClient(app) as client:                                         # lifespan 실행
        yield client


def test_image_encoder_load_failure_keeps_api_up(monkeypatch, seeded_db):
    """HF 다운로드 실패 등으로 이미지 인코더 생성이 예외를 던져도 api 는 뜨고 /health 가 OSError 를 노출한다."""
    with _lifespan_with(monkeypatch, image_load=True, image_fail=OSError("hf hub unreachable")) as client:
        health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["image_encoder"] is None
    assert "OSError" in health["image_encoder_error"] and "hf hub" in health["image_encoder_error"]
    assert "text_encoder_error" not in health                               # 텍스트 쪽은 설정에서 끔


def test_text_encoder_load_failure_keeps_api_up(monkeypatch, seeded_db):
    """텍스트 인코더 로드 실패도 api 를 죽이지 않는다."""
    with _lifespan_with(monkeypatch, text_load=True, text_fail=RuntimeError("SSL verify failed")) as client:
        health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["text_encoder"] is None
    assert "RuntimeError" in health["text_encoder_error"]


def test_search_image_returns_503_when_encoder_failed(monkeypatch, seeded_db):
    """이미지 인코더 로드 실패 후 POST /search/image 는 503 (api crash 가 아니라 503)."""
    with _lifespan_with(monkeypatch, image_load=True, image_fail=OSError("hf offline")) as client:
        r = client.post("/search/image", files={"file": ("a.jpg", b"fake", "image/jpeg")})
    assert r.status_code == 503 and "비활성" in r.json()["detail"]


def test_text_search_returns_503_when_encoder_failed(monkeypatch, seeded_db):
    """텍스트 인코더 로드 실패 후 POST /search 텍스트 질의는 503."""
    with _lifespan_with(monkeypatch, text_load=True, text_fail=OSError("hf offline")) as client:
        r = client.post("/search", data={"q": "dress"})
    assert r.status_code == 503 and "텍스트" in r.json()["detail"]


def test_keyword_search_still_works_when_encoders_failed(monkeypatch, seeded_db):
    """키워드 검색 (GET /search) 은 인코더 없이도 ILIKE 로 동작 — D1 의 핵심."""
    with _lifespan_with(monkeypatch, image_load=True, text_load=True,
                        image_fail=OSError("down"), text_fail=OSError("down")) as client:
        r = client.get("/search", params={"q": "Alpha"})
    assert r.status_code == 200
