"""dev_subset 적재: 드라이브와 같은 형식의 작은 parquet를 만들어 SQLite에 넣고, 추천 API가 그 데이터로 동작하는지 확인."""
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
from scripts.load_dev_subset import load


def make_subset(d):
    brands = pd.DataFrame({"brand_id": [3, 7], "brand": ["Nike", "Crocs"]})
    items = pd.DataFrame({"item_id": [10, 11, 12], "source_item_id": ["B01", "B02", "B03"], "brand_id": [3, 3, 7],
                          "brand": ["Nike", "Nike", "Crocs"], "title": ["Run shoe", "Cap", "Clog"],
                          "categories": ["Shoes", None, "Shoes"], "features": ["light", None, None],
                          "description": ["", "cotton", None], "price": [99.0, np.nan, 40.0]})
    users = pd.DataFrame({"user_id": [0, 1], "source_user_id": ["UA", "UB"]})
    events = pd.DataFrame({"user_id": [0, 0, 1, 1], "item_id": [10, 12, 10, 11], "type": "purchase",
                           "rating": [5.0, 4.0, 3.0, 5.0], "ts": pd.to_datetime(["2023-01-01"] * 4)})
    for n, df in {"items": items, "brands": brands, "users": users, "events": events}.items():
        df.to_parquet(d / f"{n}.parquet", index=False)
    with zipfile.ZipFile(d / "images.zip", "w") as z:
        for i in (10, 11, 12):
            z.writestr(f"{i}.jpg", b"jpg")


@pytest.fixture()
def session(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    app_db.init_db(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def test_load_and_recommend(tmp_path, session):
    data = tmp_path / "dev_subset"
    data.mkdir()
    make_subset(data)
    with session() as db:
        out = load(db, data)
        assert out == {"brands": 2, "items": 3, "users": 2, "events": 4, "embeddings": False, "images": True}
        assert db.scalar(select(func.count()).select_from(Event)) == 4
        assert {b.name for b in db.scalars(select(Brand))} == {"Nike", "Crocs"}
        item = db.scalars(select(Item).where(Item.source_item_id == "B01")).one()
        assert item.price == 99.0 and item.description == "light" and item.image_path.endswith("10.jpg")
        assert (data / "images" / "12.jpg").exists()
        with pytest.raises(SystemExit):               # 두 번 넣으면 막는다
            load(db, data)
        user = db.scalars(select(User).where(User.source_user_id == "UA")).one()

    def override():
        s = session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    try:
        r = TestClient(app).get(f"/recommend/{user.id}", params={"k": 3}).json()
        assert r["model"] == "popularity" and len(r["items"]) == 3
    finally:
        app.dependency_overrides.clear()
