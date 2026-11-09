"""--reset 재적재로 최대 id가 예전과 같아져도 상품 색인 · 인기 캐시가 새 데이터로 바뀌는지 (#40 리뷰).

API를 재시작하지 않은 채 다른 데이터를 같은 개수로 다시 넣으면, 최대 id만 보는 변경 감지는 예전 캐시를 그대로 쓴다.
"""
import numpy as np
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.models import Base, Event, Item, User
from app.recommenders.ume import _popularity_z
from app.services.catalog_index import get_index

D = 8


def _load(engine, seed, buyer_item):
    """상품 3개 · 고객 1명 · 구매 3건을 넣는다. seed가 다르면 원본 id와 임베딩이 다르다. 구매는 buyer_item번째 상품에."""
    rng = np.random.default_rng(seed)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Session() as s:
        items = [Item(source="amazon", source_item_id=f"s{seed}-{i}", title=f"t{i}",
                      text_embedding=rng.normal(size=D).tolist(), image_embedding=rng.normal(size=D).tolist())
                 for i in range(3)]
        user = User(source="sim")
        s.add_all(items + [user])
        s.flush()
        s.add_all([Event(user_id=user.id, item_id=items[buyer_item].id, type="purchase") for _ in range(3)])
        s.commit()
    return Session


def _reset(engine):
    Base.metadata.drop_all(bind=engine)          # load_dev_subset --reset 과 같은 방식
    app_db.init_db(engine)


def test_index_and_popularity_refresh_after_reset_with_same_max_id(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'reload.db'}")
    app_db.init_db(engine)

    Session = _load(engine, seed=1, buyer_item=0)
    with Session() as s:
        ix1 = get_index(s)
        text1 = ix1.text.copy()
        pop1 = _popularity_z(s, "amazon", ix1.ids, None).copy()
    assert pop1.argmax() == 0

    _reset(engine)
    Session = _load(engine, seed=2, buyer_item=2)  # 같은 개수 → 상품 · 이벤트 최대 id가 예전과 같다
    with Session() as s:
        ix2 = get_index(s)
        assert ix2.ids.tolist() == ix1.ids.tolist()                    # id는 그대로지만
        assert not np.allclose(ix2.text, text1)                         # 색인은 새 임베딩
        pop2 = _popularity_z(s, "amazon", ix2.ids, None)
        assert pop2.argmax() == 2                                       # 인기도 새 구매 기준


def test_index_kept_when_nothing_changed(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'same.db'}")
    app_db.init_db(engine)
    Session = _load(engine, seed=1, buyer_item=0)
    with Session() as s:
        assert get_index(s) is get_index(s)                             # 바뀐 게 없으면 다시 만들지 않는다


def test_popularity_cache_ignores_setup_and_non_purchase_events(tmp_path):
    """#47 리뷰: 초기 이력(is_setup=True)이나 조회 · 클릭만 늘면 인기 집계를 다시 하지 않는다 (같은 배열 객체를 그대로 돌려줌).
    비 setup 구매가 늘면 다시 집계한다."""
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'setup.db'}")
    app_db.init_db(engine)
    Session = _load(engine, seed=1, buyer_item=0)

    with Session() as s:
        ids = get_index(s).ids
        z0 = _popularity_z(s, "amazon", ids, None)
        user = s.query(User).first()
        # 초기 이력 구매 5건 (인기 1위를 바꿀 만큼) · 일반 조회 · 클릭 추가 → 다시 집계하지 않음
        s.add_all([Event(user_id=user.id, item_id=int(ids[2]), type="purchase", is_setup=True) for _ in range(5)])
        s.add_all([Event(user_id=user.id, item_id=int(ids[2]), type=t) for t in ("view", "click")])
        s.commit()
        assert _popularity_z(s, "amazon", ids, None) is z0
        assert z0.argmax() == 0
        # 비 setup 구매 추가 → 다시 집계해서 1위가 바뀜
        s.add_all([Event(user_id=user.id, item_id=int(ids[2]), type="purchase") for _ in range(5)])
        s.commit()
        z2 = _popularity_z(s, "amazon", ids, None)
        assert z2 is not z0 and z2.argmax() == 2
