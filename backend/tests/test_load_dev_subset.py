"""dev_subset v2 적재: 드라이브와 같은 형식(Amazon + H&M, 모든 표에 source)의 작은 parquet를 SQLite에 넣고 확인.

두 데이터셋의 item_id · user_id가 겹쳐도 (source, id)로 구분되는지, 표기만 다른 브랜드가 하나로 합쳐지는지,
추천 API가 그 데이터로 동작하는지, 통합 임베딩(catalog.parquet) 행을 (source, item_id)로 찾는지 본다.
"""
import zipfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.main import app
from app.models import Brand, Event, Item, User
from scripts.load_dev_subset import _embeddings, load


def make_subset(d):
    items = pd.DataFrame({
        "source": ["amazon", "amazon", "amazon", "hm"], "item_id": [10, 11, 12, 10],          # item_id 10이 두 source에 있음
        "source_item_id": ["B01", "B02", "B03", "0108775015"],
        "brand": ["adidas", "Adidas", "Crocs", "H&M"], "brand_norm": ["adidas", "adidas", "crocs", "h&m"],
        "title": ["Run shoe", "Cap", "Clog", "Strap top"], "categories": ["a", "b", "c", "d"],
        "cat_target": ["Men", "Men", "Women", "Women"], "cat_type": ["Shoes", "Accessories", "Shoes", "Clothing"],
        "features": ["light", None, None, "Black, Solid"], "description": ["", "cotton", None, "Jersey top"],
        "price": [99.0, 20.0, 40.0, 12.5], "price_source": ["real", "imputed", "real", "scaled"]})
    users = pd.DataFrame({"source": ["amazon", "amazon", "hm"], "user_id": [0, 1, 0], "source_user_id": ["UA", "UB", "HC"]})
    events = pd.DataFrame({"source": ["amazon"] * 4 + ["hm"], "user_id": [0, 0, 1, 1, 0], "item_id": [10, 12, 10, 11, 10],
                           "type": "purchase", "rating": [5.0, 2.0, 3.0, 5.0, np.nan],
                           "feedback": [1.0, -1.0, 0.5, 1.0, 1.0], "ts": pd.to_datetime(["2023-01-01"] * 5),
                           "split": ["train"] * 5})
    brands = items.groupby(["brand_norm", "brand"]).size().rename("n_items").reset_index()
    for n, df in {"items": items, "brands": brands, "users": users, "events": events}.items():
        df.to_parquet(d / f"{n}.parquet", index=False)
    with zipfile.ZipFile(d / "images.zip", "w") as z:
        for s, i in (("amazon", 10), ("amazon", 11), ("amazon", 12), ("hm", 10)):
            z.writestr(f"{s}/{i}.jpg", f"{s}{i}".encode())


@pytest.fixture()
def session(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    app_db.init_db(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def test_load_v2_and_recommend(tmp_path, session):
    data = tmp_path / "dev_subset"
    data.mkdir()
    make_subset(data)
    with session() as db:
        out = load(db, data)
        assert out == {"sources": ["amazon", "hm"], "brands": 3, "items": 4, "users": 3, "events": 5,
                       "embeddings": 0, "images": True}
        assert {b.name for b in db.scalars(select(Brand))} == {"adidas", "Crocs", "H&M"}     # adidas · Adidas 하나로
        hm_item = db.scalars(select(Item).where(Item.source == "hm")).one()
        amz_10 = db.scalars(select(Item).where(Item.source == "amazon", Item.source_item_id == "B01")).one()
        assert hm_item.id != amz_10.id and hm_item.image_path.endswith("hm/10.jpg")
        assert (data / "images" / "hm" / "10.jpg").read_bytes() == b"hm10"
        assert hm_item.category == "Women > Clothing" and amz_10.price == 99.0
        assert db.scalar(select(func.count()).select_from(Event)) == 5
        hm_user = db.scalars(select(User).where(User.source == "hm")).one()
        assert [e.item_id for e in db.scalars(select(Event).where(Event.user_id == hm_user.id))] == [hm_item.id]
        with pytest.raises(SystemExit):               # 두 번 넣으면 막는다
            load(db, data)

    def override():
        s = session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    try:
        r = TestClient(app).get(f"/recommend/{hm_user.id}", params={"k": 3}).json()
        assert r["model"] == "popularity" and len(r["items"]) == 3
    finally:
        app.dependency_overrides.clear()


def test_embeddings_v2_lookup(tmp_path):
    pd.DataFrame({"row": [0, 1, 2], "source": ["amazon", "amazon", "hm"], "item_id": [10, 12, 10]}).to_parquet(
        tmp_path / "catalog.parquet", index=False)
    text = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.save(tmp_path / "text_pca256.npy", text)
    np.save(tmp_path / "image_pca256.npy", text + 100)
    np.save(tmp_path / "has_image.npy", np.array([True, False, True]))
    vec = _embeddings(tmp_path, [("amazon", 10), ("amazon", 12), ("hm", 10), ("amazon", 99)])
    assert vec[("hm", 10)] == ([4.0, 5.0], [104.0, 105.0])
    assert vec[("amazon", 12)] == ([2.0, 3.0], None)          # 이미지 없는 상품
    assert ("amazon", 99) not in vec                           # 카탈로그에 없는 상품은 건너뜀


@pytest.mark.skipif(not hasattr(Item, "text_embedding"), reason="임베딩 열은 PR #14 병합 후")
def test_load_with_embeddings(tmp_path, session):
    data = tmp_path / "dev_subset"
    data.mkdir()
    make_subset(data)
    emb = tmp_path / "v2"
    emb.mkdir()
    pd.DataFrame({"row": [0, 1, 2, 3], "source": ["amazon", "amazon", "amazon", "hm"],
                  "item_id": [10, 11, 12, 10]}).to_parquet(emb / "catalog.parquet", index=False)
    np.save(emb / "text_pca256.npy", np.eye(4, 256, dtype=np.float32))
    np.save(emb / "image_pca256.npy", np.eye(4, 256, k=1, dtype=np.float32))
    np.save(emb / "has_image.npy", np.array([True, True, False, True]))
    with session() as db:
        assert load(db, data, emb, extract_images=False)["embeddings"] == 4
        hm_item = db.scalars(select(Item).where(Item.source == "hm")).one()
        assert hm_item.text_embedding[3] == 1.0 and len(hm_item.text_embedding) == 256
        crocs = db.scalars(select(Item).where(Item.source_item_id == "B03")).one()
        assert crocs.image_embedding is None                    # 이미지 없는 상품
