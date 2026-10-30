"""통합 검색 POST /search. 모델 대신 가짜 인코더(단어 → 정해진 방향)로 점수 결합 · 번역 · 필터 · 오류를 확인한다.
SQLite에서도 동작한다 (점수는 메모리 색인에서 계산)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.models import Brand, Item

D = 256
DIR = {"dress": 0, "jeans": 1, "shoes": 2}          # 가짜 인코더: 단어마다 한 축


def unit(axis, noise=0.0, seed=0):
    v = np.zeros(D, np.float32)
    v[axis] = 1
    v += np.random.default_rng(seed).normal(scale=noise, size=D).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeText:
    pca_ready = True

    def encode(self, texts):
        return np.stack([unit(next(a for w, a in DIR.items() if w in t)) for t in texts])


class FakeImage(FakeText):
    def encode_texts(self, texts):
        return self.encode(texts)

    def encode_image_bytes(self, data):
        return unit(DIR[data.decode()])               # 업로드 바이트 = 단어 (테스트용)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "load_image_encoder", False)
    monkeypatch.setattr(settings, "load_text_encoder", False)
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'search.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    with Session() as s:
        b1, b2 = Brand(name="acme"), Brand(name="H&M")
        s.add_all([b1, b2])
        s.flush()
        n = 0
        # amazon 40개 · hm 10개. 카테고리마다 축 근처, hm은 같은 카테고리라도 점수가 조금 낮게(다른 방향 섞임)
        for src, brand, count, mix in (("amazon", b1, 40, 0.0), ("hm", b2, 10, 0.6)):
            for i in range(count):
                cat = list(DIR)[i % 3]
                v = unit(DIR[cat], noise=0.05, seed=n)
                v = v + mix * unit(10, seed=n)           # hm만 다른 방향 성분 → 원래 점수로는 amazon이 독차지
                v /= np.linalg.norm(v)
                s.add(Item(source=src, source_item_id=str(i), brand_id=brand.id, title=f"{src} {cat} {i}",
                           category=cat, price=10.0, text_embedding=v.tolist(), image_embedding=v.tolist()))
                n += 1
        s.commit()

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    app.state.text_encoder, app.state.image_encoder = FakeText(), FakeImage()
    yield TestClient(app)
    app.dependency_overrides.clear()
    app.state.text_encoder = app.state.image_encoder = None


def titles(r):
    assert r.status_code == 200, r.text
    return [i["title"] for i in r.json()["items"]]


def test_korean_query_translated_and_sources_mixed(client):
    r = client.post("/search", data={"q": "검은색 원피스", "k": 10})
    body = r.json()
    assert body["query_en"] == "black dress" and body["untranslated"] == []
    assert body["weights"] == {"text": 0.5, "text_image": 0.5}
    t = titles(r)
    assert all("dress" in x for x in t[:5])
    assert any(x.startswith("hm") for x in t)        # per_source_rank: hm도 비율대로 나옴


def test_global_merge_lets_one_source_dominate(client):
    t = titles(client.post("/search", data={"q": "dress", "k": 10, "merge": "global"}))
    assert not any(x.startswith("hm") for x in t)


def test_image_and_text_priority(client):
    img = {"file": ("q.jpg", b"shoes", "image/jpeg")}
    t_img = titles(client.post("/search", data={"q": "jeans", "text_weight": 0.1, "k": 5}, files=img))
    t_txt = titles(client.post("/search", data={"q": "jeans", "text_weight": 0.9, "k": 5}, files=img))
    assert "shoes" in t_img[0] and "jeans" in t_txt[0]
    only_img = client.post("/search", data={"k": 5}, files=img).json()
    assert only_img["weights"] == {"image": 1.0} and "shoes" in only_img["items"][0]["title"]


def test_filters(client):
    t = titles(client.post("/search", data={"q": "jeans", "source": "hm", "k": 5}))
    assert t and all(x.startswith("hm") for x in t)


def test_errors(client):
    assert client.post("/search", data={}).status_code == 422
    assert client.post("/search", data={"q": "dress", "merge": "x"}).status_code == 422
    assert client.post("/search", data={"q": "dress"}, files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415
    app.state.text_encoder = None
    assert client.post("/search", data={"q": "dress"}).status_code == 503


def test_min_rel_threshold(client):
    # hm 상품은 질의와 코사인 약 0.86 (amazon 약 1.0) → 기준 0.95면 hm은 칸을 못 채움, 0이면 기존처럼 섞임
    t = titles(client.post("/search", data={"q": "dress", "k": 10, "min_rel": 0.95}))
    assert not any(x.startswith("hm") for x in t)
    t = titles(client.post("/search", data={"q": "dress", "k": 10, "min_rel": 0}))
    assert any(x.startswith("hm") for x in t)
