"""색상 변형 묶기 (services/variants.py, 다음작업 C1): 같은 옷(H&M product_code)은 추천 목록에 한 번만, 나머지는 variant_ids."""
import numpy as np
import pytest

from app.services.catalog_index import CatalogIndex
from app.services.rerank import diversify
from app.services.variants import first_per_variant, variant_key


def test_variant_key_only_for_hm_article_ids():
    assert variant_key("hm", "0372860001") == variant_key("hm", "0372860069") == "hm:0372860"
    assert variant_key("hm", "0372861001") != variant_key("hm", "0372860001")
    assert variant_key("amazon", "B07XYZ1234") == ""          # 변형 정보 없는 데이터셋
    assert variant_key("hm", "123") == "" and variant_key("hm", None) == "" and variant_key(None, "0372860001") == ""


def test_first_per_variant_keeps_best_and_fills_at_end():
    ids, keys = [1, 2, 3, 4, 5], ["a", "a", "", "b", "a"]
    assert first_per_variant(ids, keys, 3) == [1, 3, 4]
    assert first_per_variant(ids, keys, 4) == [1, 3, 4, 2]        # 모자라면 건너뛴 변형을 점수 순으로
    assert first_per_variant(ids, keys, 3, per_variant=2) == [1, 2, 3]


def test_diversify_one_per_variant_then_category_cap():
    # 점수 순 0..7: 0 · 1 · 2는 같은 옷, 3 · 4도 같은 옷
    var = np.array(["a", "a", "a", "b", "b", "", "", ""], object)
    cats = np.array(["T", "T", "T", "T", "P", "P", "S", "S"], object)
    out = diversify(np.arange(8), 4, cats=cats, variants=var, params={"max_category_share": 0.5, "min_cap": 2})
    assert out.tolist() == [0, 3, 5, 6]
    assert diversify(np.arange(8), 4, variants=var, params={"max_category_share": None}).tolist() == [0, 3, 5, 6]
    assert diversify(np.arange(3), 3, variants=var[:3], params={"max_category_share": None}).tolist() == [0, 1, 2]   # 다른 후보가 없으면 채움
    off = {"max_category_share": None, "max_per_variant": None}
    assert diversify(np.arange(8), 4, variants=var, params=off).tolist() == [0, 1, 2, 3]                            # 끄면 예전 그대로


def test_catalog_index_variant_ids():
    z = np.zeros((4, 2), np.float32)
    ix = CatalogIndex(np.array([10, 11, 12, 13]), np.array(["hm"] * 3 + ["amazon"], object), np.full(4, -1), z, z,
                      np.array([""] * 4, object), np.array(["hm:1", "hm:1", "hm:2", ""], object))
    assert ix.variant_ids(10) == [11] and ix.variant_ids(11) == [10]
    assert ix.variant_ids(12) == [] and ix.variant_ids(13) == [] and ix.variant_ids(999) == []


# ---------------------------------------------------------------- API

@pytest.fixture()
def env(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from sqlalchemy.orm import sessionmaker

    from app import db as app_db
    from app.config import settings
    from app.main import app
    from app.models import Event, Item, User
    from scripts.seed_sample import seed
    monkeypatch.setattr(settings, "load_image_encoder", False)
    monkeypatch.setattr(settings, "load_text_encoder", False)
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    with Session() as s:
        seed(s)
        # 같은 옷 3색(0372860xxx) + 다른 옷 1개, 가장 인기 있게 구매를 몰아 준다 (시드 상품은 임베딩 없음 → 색인에서 빠짐)
        hm = [Item(source="hm", source_item_id=sid, title=f"Basic socks {sid[-3:]}", category="양말",
                   text_embedding=[float(j + 1), 0.0, 0.0, 0.0])
              for j, sid in enumerate(["0372860001", "0372860002", "0372860069", "0999999001"])]
        s.add_all(hm)
        s.flush()
        buyer = s.query(User).first()
        for n, it in zip((40, 39, 38, 37), hm):
            s.add_all([Event(user_id=buyer.id, item_id=it.id, type="purchase") for _ in range(n)])
        s.commit()
        ids = [it.id for it in hm]

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    yield TestClient(app), ids
    app.dependency_overrides.clear()


def test_popularity_recommendation_shows_one_color_with_variant_ids(env):
    client, (a, b, c, other) = env
    body = client.get("/recommend/1", params={"k": 5}).json()
    recs = body["recommendations"]
    shown = [int(r["product_id"]) for r in recs]
    assert shown[:2] == [a, other]                                 # 2 · 3위였던 같은 옷의 다른 색상은 빠지고 다른 옷이 올라온다
    assert b not in shown and c not in shown and len(shown) == 5 == len(set(shown))
    assert recs[0]["variant_ids"] == [str(b), str(c)]
    assert recs[1]["variant_ids"] == [] and all(r["variant_ids"] == [] for r in recs[2:])
