"""서비스 인기 통계 · 재학습 카운터가 어떤 이벤트를 세는지 (services/event_scope.py).

- 추천 노출 view(request_id 있음)는 인기 추천에 안 들어간다 — 보여 준 만큼 인기가 오르는 되먹임 방지. 상세 조회 view는 들어간다.
- 시뮬레이터 고객(source="sim") 이벤트는 인기 추천 · UME 인기 점수에서 빠진다 (설정으로 켤 수 있음).
- trainer 재학습 카운터(recsys.retrain.new_log_count)는 click · add_to_cart · purchase만 센다
  (CT 루프는 /metrics/log-delta?types=click,add_to_cart,purchase로 같은 수를 본다, #70).
"""
import uuid
from datetime import datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.models import Event, Item, User
from app.recommenders.popularity import _popular_scored
from app.recommenders.ume import _pop_cache, _popularity_z

T0 = datetime(2020, 9, 1)


@pytest.fixture()
def Session(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'scope.db'}")
    app_db.init_db(engine)
    S = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    _pop_cache.clear()
    yield S
    _pop_cache.clear()


def _setup(S, n_items=4):
    with S() as s:
        items = [Item(source="hm", source_item_id=str(i), title=f"t{i}") for i in range(n_items)]
        real, sim = User(source="hm"), User(source="sim")
        s.add_all(items + [real, sim])
        s.commit()
        return [it.id for it in items], real.id, sim.id


def _add(S, *events):
    with S() as s:
        s.add_all(list(events))
        s.commit()


def test_impression_views_do_not_count_but_detail_views_do(Session):
    ids, real, _ = _setup(Session)
    # 상품 0: 노출 view 30건 (6.0점이 될 뻔) / 상품 1: 클릭 1건 (1.0) / 상품 2: 상세 view 10건 (2.0)
    _add(Session, *[Event(user_id=real, item_id=ids[0], type="view", request_id=uuid.uuid4()) for _ in range(30)],
         Event(user_id=real, item_id=ids[1], type="click"),
         *[Event(user_id=real, item_id=ids[2], type="view") for _ in range(10)])
    with Session() as s:
        scored = dict(_popular_scored(s, 4, per_variant=None))
    assert scored[ids[0]] == 0.0                     # 노출만 받은 상품은 점수 없음 (id 순 채우기)
    assert scored[ids[1]] == pytest.approx(1.0)
    assert scored[ids[2]] == pytest.approx(2.0)


def test_sim_customers_excluded_from_popularity_unless_enabled(Session, monkeypatch):
    ids, real, sim = _setup(Session)
    _add(Session, *[Event(user_id=sim, item_id=ids[3], type="purchase") for _ in range(5)],
         Event(user_id=real, item_id=ids[1], type="purchase"))
    with Session() as s:
        assert _popular_scored(s, 1, per_variant=None)[0][0] == ids[1]
    monkeypatch.setattr(settings, "popularity_include_sim", True)
    with Session() as s:
        assert _popular_scored(s, 1, per_variant=None)[0][0] == ids[3]


def test_sim_purchases_do_not_move_hm_recent_window(Session):
    """시뮬레이터 구매(현재 시각)가 H&M '마지막 구매 기준 최근 7일' 창을 차지하지 않는다."""
    ids, real, sim = _setup(Session)
    _add(Session, *[Event(user_id=real, item_id=ids[0], type="purchase", created_at=T0) for _ in range(3)],
         Event(user_id=real, item_id=ids[1], type="purchase", created_at=T0 - timedelta(days=1)),
         *[Event(user_id=sim, item_id=ids[2], type="purchase", created_at=T0 + timedelta(days=2000)) for _ in range(9)])
    with Session() as s:
        z = _popularity_z(s, "hm", np.array(ids), 7)
    assert z.argmax() == 0 and z[2] == z[3]          # sim 구매 상품 2는 구매 없는 상품 3과 같다


def test_trainer_counts_training_types_only(Session):
    from recsys.retrain import new_log_count
    ids, real, sim = _setup(Session)
    _add(Session, *[Event(user_id=sim, item_id=ids[0], type="view", request_id=uuid.uuid4()) for _ in range(10)],
         Event(user_id=sim, query="dress", type="search"),
         Event(user_id=sim, item_id=ids[0], type="click"), Event(user_id=real, item_id=ids[1], type="add_to_cart"),
         Event(user_id=real, item_id=ids[1], type="purchase"),
         Event(user_id=real, item_id=ids[2], type="purchase", is_setup=True))
    with Session() as s:
        assert new_log_count(s, 0) == (3, 15)    # sim 클릭은 학습 데이터라 센다, view · search · setup 제외
