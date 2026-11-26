"""상품 키워드 배지 (D4, services/badges.py · scripts/load_badges.py · /items · /search 응답)."""
from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.models import Event, Item, ItemBadge, User
from app.services import badges as badges_mod

T0 = datetime(2020, 9, 1)


@pytest.fixture(autouse=True)
def _no_encoders(monkeypatch):
    monkeypatch.setattr(settings, "load_image_encoder", False)
    monkeypatch.setattr(settings, "load_text_encoder", False)
    badges_mod._log_cache.clear()


@pytest.fixture()
def env(tmp_path):
    """H&M 같은 카테고리 상품 20개. 상품 0: 최근 7일 9건 (그 전 7일 1건) → 최근 인기 · 급상승. 상품 1: 최근 5건 · 그 전 3건 → 없음
    (카테고리 상위 3% 아님).
    시뮬레이터 고객 구매(상품 2, 10건)는 세지 않는다. Amazon 상품 1개에는 리뷰 배지 4개."""
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'b.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    with Session() as s:
        hm = [Item(source="hm", source_item_id=f"h{i}", title=f"h{i}", category="여성 > 상의 > 티셔츠") for i in range(20)]
        az = Item(source="amazon", source_item_id="B0001", title="dress", category="여성 > 원피스·세트 > 원피스")
        u, sim = User(source="hm"), User(source="sim")
        s.add_all(hm + [az, u, sim])
        s.flush()
        buys = [(hm[0], d) for d in (1, 2, 3, 4, 5, 6, 1.5, 2.5, 3.5)] + [(hm[0], 10)] + [(hm[1], d) for d in (1, 2, 3, 4, 5, 8, 9, 10)]
        s.add_all([Event(user_id=u.id, item_id=it.id, type="purchase", created_at=T0 - timedelta(days=d)) for it, d in buys])
        s.add_all([Event(user_id=sim.id, item_id=hm[2].id, type="purchase", created_at=T0) for _ in range(10)])
        s.add_all([ItemBadge(item_id=az.id, label=lab, kind=k, origin="review", rank=r, n_mention=40 + r, n_reviews=400, lift=2.5)
                   for r, (lab, k) in enumerate([("작게 나와요", "caution"), ("착용감 좋음", "good"), ("체형 보정", "vibe"),
                                                 ("출근룩", "occasion")])])
        s.commit()
        ids = {"hm": [i.id for i in hm], "amazon": az.id}

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    yield Session, ids, TestClient(app)
    app.dependency_overrides.clear()


def test_log_badges_hot_and_rising_service_scope_only(env):
    Session, ids, _ = env
    with Session() as db:
        b = badges_mod.log_badges(db, ids["hm"])
    labels = {i: [x["label"] for x in v] for i, v in b.items()}
    assert labels[ids["hm"][0]] == ["최근 인기", "급상승"]                     # 최근 7일 9건 · 그 전 1건
    assert ids["hm"][1] not in b                                               # 5건이지만 카테고리 상위 3% 아님, 그 전 3건
    assert ids["hm"][2] not in b                                               # 시뮬레이터 구매는 세지 않음
    assert b[ids["hm"][0]][1]["evidence"] == "최근 7일 구매 9건 (그 전 7일 1건)"


def test_review_badges_ordered_with_evidence_and_card_limit(env):
    Session, ids, _ = env
    with Session() as db:
        full = badges_mod.badges_for(db, [ids["amazon"]], limit=None)[ids["amazon"]]
        card = badges_mod.badges_for(db, [ids["amazon"]])[ids["amazon"]]
    assert [x["label"] for x in full] == ["작게 나와요", "착용감 좋음", "체형 보정", "출근룩"]
    assert len(card) == 3 and card[0]["evidence"] == "리뷰 400건 중 40건, 같은 카테고리의 2.5배"


def test_items_and_detail_responses_carry_badges(env):
    _, ids, client = env
    a, h0 = ids["amazon"], ids["hm"][0]
    items = client.get(f"/items?ids={h0},{a},{ids['hm'][5]}").json()
    assert [x["label"] for x in items[0]["badges"]] == ["최근 인기", "급상승"]
    assert len(items[1]["badges"]) == 3 and items[2]["badges"] == []
    assert len(client.get(f"/items/{a}").json()["badges"]) == 4                # 상세는 전부


def test_cache_refreshes_when_new_purchase_arrives(env):
    Session, ids, _ = env
    h5 = ids["hm"][5]
    with Session() as db:
        assert h5 not in badges_mod.log_badges(db, [h5])
        u = db.query(User).filter_by(source="hm").first()
        db.add_all([Event(user_id=u.id, item_id=h5, type="purchase", created_at=T0 - timedelta(hours=h)) for h in range(10)])
        db.commit()
        assert [x["label"] for x in badges_mod.log_badges(db, [h5])[h5]] == ["최근 인기", "급상승"]


def test_load_badges_replaces_review_rows(env):
    from scripts.load_badges import load
    Session, ids, _ = env
    df = pd.DataFrame({"parent_asin": ["B0001", "NOPE"], "badge": ["비침 주의", "x"], "kind": ["caution", "good"],
                       "n_reviews": [100, 1], "n_mention": [9, 1], "share": [0.09, 1.0], "lift": [3.0, 1.0], "rank": [0, 0]})
    with Session() as db:
        r = load(db, df)
        assert r == {"inserted": 1, "items": 1, "missing_items": 1}
        assert [b.label for b in db.query(ItemBadge).all()] == ["비침 주의"]    # 기존 리뷰 배지 4개는 지우고 다시
        assert load(db, df)["inserted"] == 1 and db.query(ItemBadge).count() == 1   # 다시 실행해도 안전
