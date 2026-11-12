"""재정렬 마지막 단계: MAB 탐색 슬롯 · 신규 상품 노출 · 같은 소분류 연속 제한 (services/explore.py).

앞부분은 DB 없는 순수 함수, 뒷부분은 SQLite 샘플 DB로 /recommend 응답까지 확인한다."""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.services import explore as E
from app.services.explore import Rec


# ---------------------------------------------------------------- 순수 함수

def test_config_off_by_default_and_merges_defaults():
    assert E.config({}) is None and E.config({"explore": None}) is None and E.config({"explore": False}) is None
    cfg = E.config({"explore": {"c": 2.0}})
    assert cfg["c"] == 2.0 and cfg["algorithm"] == "ucb" and cfg["new_days"] == 7
    assert E.config({"explore": True})["slots"] == 2


def test_slot_count_follows_spec_one_or_two():
    cfg = E.config({"explore": True})
    assert E.n_slots(20, cfg) == 2 and E.n_slots(10, cfg) == 2
    assert E.n_slots(5, cfg) == 1 and E.n_slots(2, cfg) == 1 and E.n_slots(1, cfg) == 0


def test_ucb_tries_unseen_first_then_balances_ctr_and_uncertainty():
    assert E.ucb_index(0, 0, 100, 1.0) == math.inf
    # 같은 CTR이면 노출이 적은 쪽(불확실성 큼)이 높다
    assert E.ucb_index(10, 1, 1000, 1.0) > E.ucb_index(500, 50, 1000, 1.0)
    # c = 0이면 CTR 그대로
    assert E.ucb_index(10, 3, 1000, 0.0) == pytest.approx(0.3)
    assert E.ucb_index(5, 9, 100, 0.0) == 1.0                      # 노출보다 많은 클릭은 노출 수로 자름


def test_choose_ucb_prefers_unseen_new_items():
    cfg = E.config({"explore": True})
    stats = {1: (100, 30), 2: (100, 1), 3: (0, 0), 4: (0, 0)}
    picks = E.choose([1, 2, 3, 4], stats, 2, cfg, np.random.default_rng(0), new_ids={4})
    assert picks == [4, 3]                                         # 둘 다 노출 0 → 신규 먼저
    picks = E.choose([1, 2], stats, 1, {**cfg, "c": 0.0}, np.random.default_rng(0))
    assert picks == [1]                                            # c=0: CTR 높은 쪽


def test_choose_epsilon_greedy_exploits_with_zero_epsilon_and_randomizes_with_one():
    cfg = E.config({"explore": {"algorithm": "epsilon_greedy", "epsilon": 0.0}})
    stats = {1: (100, 5), 2: (100, 40), 3: (100, 10)}
    assert E.choose([1, 2, 3], stats, 2, cfg, np.random.default_rng(0)) == [2, 3]
    cfg1 = {**cfg, "epsilon": 1.0}
    seen = {tuple(E.choose([1, 2, 3], stats, 1, cfg1, np.random.default_rng(s))) for s in range(30)}
    assert seen == {(1,), (2,), (3,)}


def test_choose_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        E.choose([1], {}, 1, {**E.DEFAULT, "algorithm": "thompson"}, np.random.default_rng(0))


def test_place_keeps_length_and_puts_exploration_at_positions():
    exploit = [Rec(i, 1.0 - i / 100) for i in range(10)]
    out = E.place(exploit, [100, 101], [3, 8], 10)
    assert len(out) == 10 and [r.item_id for r in out if r.is_exploration] == [100, 101]
    assert [i for i, r in enumerate(out) if r.is_exploration] == [3, 8]
    assert [r.item_id for r in out if not r.is_exploration] == list(range(8))   # 원래 상위 8개, 순서 유지


def test_place_clips_positions_for_short_lists_and_skips_duplicates():
    exploit = [Rec(i, 1.0) for i in range(5)]
    out = E.place(exploit, [100], [3, 8], 5)
    assert [i for i, r in enumerate(out) if r.is_exploration] == [3] and len(out) == 5
    out = E.place(exploit, [2], [3], 5)                            # 이미 목록에 있는 상품은 탐색으로 넣지 않음
    assert not any(r.is_exploration for r in out) and len(out) == 5


def test_break_runs_removes_three_in_a_row_when_possible():
    cats = {1: "T", 2: "T", 3: "T", 4: "T", 5: "D", 6: "J"}
    recs = [Rec(i, 1.0) for i in (1, 2, 3, 4, 5, 6)]
    out = E.break_runs(recs, cats)
    assert E.max_run_length(out, cats) <= 2 and sorted(r.item_id for r in out) == [1, 2, 3, 4, 5, 6]
    assert [r.item_id for r in out][:2] == [1, 2]                  # 앞쪽 순서는 최대한 유지


def test_break_runs_leaves_list_when_no_other_category_and_ignores_empty():
    cats = {1: "T", 2: "T", 3: "T"}
    assert [r.item_id for r in E.break_runs([Rec(i, 1.0) for i in (1, 2, 3)], cats)] == [1, 2, 3]
    cats = {1: "", 2: "", 3: "", 4: "D"}
    assert [r.item_id for r in E.break_runs([Rec(i, 1.0) for i in (1, 2, 3, 4)], cats)] == [1, 2, 3, 4]


def test_break_runs_keeps_exploration_flags():
    cats = {1: "T", 2: "T", 3: "T", 9: "D"}
    out = E.break_runs([Rec(1, 1.0), Rec(2, 1.0), Rec(3, 1.0), Rec(9, 0.0, True)], cats)
    assert [r.item_id for r in out] == [1, 2, 9, 3] and out[2].is_exploration


def test_leaf_category():
    assert E.leaf("여성 > 상의 > 셔츠·블라우스") == "셔츠·블라우스" and E.leaf(None) == "" and E.leaf("Dress") == "Dress"


# ---------------------------------------------------------------- DB · API

@pytest.fixture()
def env(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from sqlalchemy.orm import sessionmaker

    from app import db as app_db
    from app.config import settings
    from app.main import app
    from scripts.seed_sample import seed
    monkeypatch.setattr(settings, "load_image_encoder", False)
    monkeypatch.setattr(settings, "load_text_encoder", False)
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
    yield TestClient(app), Session
    app.dependency_overrides.clear()


def _enable_explore(Session, cfg=True):
    from sqlalchemy import select

    from app.models import ModelVersion
    with Session() as s:
        mv = s.scalars(select(ModelVersion).where(ModelVersion.status == "active")).first()
        mv.params = {**(mv.params or {}), "explore": cfg}
        s.commit()


def _cats(client, recs):
    return {int(r["product_id"]): client.get(f"/items/{r['product_id']}").json()["category"] for r in recs}


def test_recommend_without_explore_has_no_exploration_but_no_triple_runs(env):
    client, _ = env
    recs = client.get("/recommend/1", params={"k": 10}).json()["recommendations"]
    assert len(recs) == 10 and not any(r["is_exploration"] for r in recs)
    cats = _cats(client, recs)
    assert E.max_run_length([Rec(int(r["product_id"]), 0) for r in recs], cats) <= 2


def test_recommend_with_explore_has_two_slots_and_spec_fields(env):
    client, Session = env
    _enable_explore(Session)
    body = client.get("/recommend/1", params={"k": 10}).json()
    recs = body["recommendations"]
    explo = [r for r in recs if r["is_exploration"]]
    assert len(recs) == 10 and len(explo) == 2
    assert {r["reason"] for r in explo} == {"mab_exploration"}
    assert {r["score_kind"] for r in explo} == {"rank"} and {r["score_basis"] for r in explo} == {"rank_placeholder"}
    assert len({r["product_id"] for r in recs}) == 10                           # 중복 없음
    assert body["items"] == [int(r["product_id"]) for r in recs]
    assert body["pipeline_latency"]["reranking_ms"] >= 0
    small = client.get("/recommend/1", params={"k": 5}).json()["recommendations"]
    assert sum(r["is_exploration"] for r in small) == 1


def _make_new(Session, ids, user=None, shown=()):
    """ids를 막 등록된 상품(이벤트 없음 · registered_at 하루 안)으로 만든다."""
    from sqlalchemy import delete

    from app.models import Event
    from scripts.mark_new_items import mark
    with Session() as s:
        s.execute(delete(Event).where(Event.item_id.in_(ids)))
        s.commit()
        return mark(s, ids=ids, days=1)


def test_new_items_get_one_slot_and_preferred_category_the_other(env):
    """두 슬롯 = 신규 상품 1개(노출 보장) + 선호 소분류 상품 1개."""
    client, Session = env
    from sqlalchemy import select

    from app.models import Event
    shown = {int(r["product_id"]) for r in client.get("/recommend/3", params={"k": 10}).json()["recommendations"]}
    with Session() as s:
        bought = set(s.scalars(select(Event.item_id).where(Event.user_id == 3, Event.type == "purchase")))
    new = set(_make_new(Session, [i for i in range(30, 0, -1) if i not in shown | bought][:2]))
    _enable_explore(Session)
    explo = [int(r["product_id"]) for r in client.get("/recommend/3", params={"k": 10}).json()["recommendations"]
             if r["is_exploration"]]
    assert len(explo) == 2 and len(set(explo) & new) == 1


def test_new_items_rotate_across_requests(env):
    """노출 0인 신규 상품이 여럿이면 요청(고객 · 시각)마다 다른 신규 상품이 먼저 온다 — 같은 상품만 반복 노출 X."""
    _, Session = env
    import numpy as np

    from app.services.explore import candidate_pool, choose
    new = set(_make_new(Session, list(range(11, 21))))
    cfg = E.config({"explore": True})
    firsts = set()
    with Session() as s:
        for seed in range(10):
            cands, new_ids = candidate_pool(s, 1, set(), cfg, rng=np.random.default_rng(seed))
            assert new_ids <= new
            firsts |= set(choose([i for i in cands if i in new_ids], {}, 1, cfg, np.random.default_rng(seed), new_ids))
    assert len(firsts) >= 3


def test_old_registration_is_not_new(env):
    client, Session = env
    from app.services.explore import candidate_pool
    from scripts.mark_new_items import mark
    now = datetime(2026, 11, 12, 12, 0)
    with Session() as s:
        mark(s, ids=[5], days=0, now=now - timedelta(days=8))     # 8일 전 → 신규 아님
        mark(s, ids=[6], days=0, now=now - timedelta(days=2))     # 2일 전 → 신규
        _, new_ids = candidate_pool(s, 1, set(), E.config({"explore": True}), now=now)
    assert new_ids == {6}


def test_exploration_excludes_purchased_and_already_recommended(env):
    client, Session = env
    from sqlalchemy import select

    from app.models import Event
    from app.services.explore import candidate_pool
    with Session() as s:
        bought = set(s.scalars(select(Event.item_id).where(Event.user_id == 2, Event.type == "purchase")))
        cands, _ = candidate_pool(s, 2, {1, 2, 3}, E.config({"explore": True}))
    assert not (set(cands) & (bought | {1, 2, 3}))


def test_item_stats_counts_views_and_clicks_excluding_setup(env):
    _, Session = env
    from app.models import Event
    from app.services.explore import item_stats
    with Session() as s:
        before = item_stats(s, [7]).get(7, (0, 0))
        s.add_all([Event(user_id=1, item_id=7, type="view"), Event(user_id=1, item_id=7, type="click"),
                   Event(user_id=1, item_id=7, type="view", is_setup=True)])
        s.commit()
        assert item_stats(s, [7])[7] == (before[0] + 1, before[1] + 1)
