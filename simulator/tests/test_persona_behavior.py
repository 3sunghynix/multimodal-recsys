"""server_run 페르소나 행동 (A1 후속) — 작은 카탈로그 · 가짜 서버로 로직만 확인."""
import json

import pandas as pd
import pytest

from simulator import server_run
from simulator.persona_behavior import PersonaBehavior

N = 14


@pytest.fixture
def catalog(tmp_path):
    rows, events = [], []
    for source, year in (("amazon", 2023), ("hm", 2020)):
        for i in range(8):
            rows.append(dict(source=source, item_id=i, source_item_id=f"{source}-{i}", brand_norm="a" if i < 4 else "b",
                             cat_target="Women", cat_type="Clothing", category_l1="여성", category_l2="상의",
                             category_l3="티셔츠", price=10.0 + i, price_source="real"))
            events.append(dict(source=source, item_id=i, split="train", feedback=1.0, ts=f"{year}-08-{i + 1:02d}", type="purchase"))
    pd.DataFrame(rows).to_parquet(tmp_path / "items.parquet")
    pd.DataFrame(events).to_parquet(tmp_path / "events.parquet")
    snap = {"kind": "simulator_db_item_snapshot_v1", "items": [
        dict(id=100 + n, source=r["source"], source_item_id=r["source_item_id"]) for n, r in enumerate(rows)]}
    (tmp_path / "db_items.json").write_text(json.dumps(snap))
    return tmp_path


def _behavior(catalog, n=N, seed=42):
    return PersonaBehavior(catalog, catalog / "db_items.json", n, seed=seed)


def _recs(items):
    return [(p, s, sid) for p, (s, sid) in enumerate(items, start=1)]


def test_customers_personas_and_setup_ids(catalog):
    b = _behavior(catalog)
    assert len(b.customers) == N and len({c["customer_id"] for c in b.customers}) == N
    assert set(b.summary()["customers_by_persona"]) <= set(b.personas)
    for ci in range(N):
        ids = b.setup_purchases(ci)
        assert all(100 <= i < 116 for i in ids)                         # 서버 DB id로 변환
        assert len(ids) == len(b.customers[ci]["initial_purchase_keys"])


def test_actions_are_deterministic_and_funnel_ordered(catalog):
    recs = _recs([("hm", f"hm-{i}") for i in range(8)])
    a = [_behavior(catalog).actions(ci, si, recs) for ci in range(N) for si in range(3)]
    b = [_behavior(catalog).actions(ci, si, recs) for ci in range(N) for si in range(3)]
    assert a == b
    for out in a:
        for acts in out.values():
            assert acts == ["click", "add_to_cart", "purchase"][:len(acts)]   # 클릭 없이 구매 없음


def test_same_item_same_rank_same_outcome_regardless_of_other_items(catalog):
    """공통 난수: 같은 고객 · 세션 · 순위에서 같은 상품이면 목록의 다른 상품과 무관하게 같은 결과 (그룹 간 공정 비교)."""
    b = _behavior(catalog)
    x = b.actions(0, 0, _recs([("hm", "hm-0"), ("hm", "hm-1"), ("amazon", "amazon-2")]))
    y = b.actions(0, 0, _recs([("hm", "hm-0"), ("hm", "hm-5"), ("amazon", "amazon-7")]))
    assert x[1] == y[1]


def test_unknown_items_are_view_only(catalog):
    b = _behavior(catalog)
    out = b.actions(0, 0, _recs([("hm", "nope"), ("other", "x")]))
    assert out == {1: [], 2: []} and b.stats["unknown_items"] == 2


def test_preference_raises_click_rate_for_brand_loyal(catalog):
    """선호 점수가 클릭 확률을 바꾼다: brand_loyal(브랜드 가중 0.8)은 선호 브랜드 상품을 더 자주 클릭한다.
    (페르소나 전체 평균으로는 보장되지 않는다 — novelty · 가격 가중이 큰 페르소나는 처음 보는 브랜드를 더 누른다)"""
    b = _behavior(catalog, n=70)
    pref, other = [0, 0], [0, 0]
    loyal = [(ci, c) for ci, c in enumerate(b.customers) if c["persona"] == "brand_loyal" and len(c["preferred_brands"]) == 1]
    assert loyal
    for ci, c in loyal:
        src = "hm" if any(k.startswith("hm") for k in c["initial_purchase_keys"]) else "amazon"
        for si in range(200):
            for p, acts in b.actions(ci, si, _recs([(src, f"{src}-{i}") for i in range(8)])).items():
                bucket = pref if ("a" if p <= 4 else "b") in c["preferred_brands"] else other
                bucket[0] += 1
                bucket[1] += "click" in acts
    assert pref[1] / pref[0] > 1.5 * other[1] / other[0]


def test_missing_setup_item_in_snapshot_fails_early(catalog):
    snap = json.loads((catalog / "db_items.json").read_text())
    snap["items"] = snap["items"][:3]
    (catalog / "db_items.json").write_text(json.dumps(snap))
    with pytest.raises(ValueError, match="서버 DB에 없음"):
        _behavior(catalog)


class FakeClient:
    """서버 흉내: 고객 id 발급 · 추천(고정 목록) · 이벤트 저장 (event_id 멱등)."""
    store: dict = {}
    users: dict = {}

    def __init__(self, base_url, timeout=15.0):
        pass

    event_id = server_run.ServerClient.event_id

    def ensure_customer(self, run_id, idx, persona=None):
        return FakeClient.users.setdefault((run_id, idx), (len(FakeClient.users) + 1, persona))[0]

    def recommend(self, user_id, session_id, k=10):
        items = [("hm", f"hm-{i}") for i in range(k)]
        return server_run.RecommendResponse(request_id=f"r-{user_id}-{session_id}", model="ume", model_version_id=1,
                                            experiment_id=None, variant=None,
                                            impressions=[(p, 108 + p - 1) for p in range(1, k + 1)],
                                            products=_recs(items))

    def send_events_batch(self, events):
        out = {"created": 0, "duplicate": 0, "error": 0}
        for e in events:
            key = e["event_id"]
            out["duplicate" if key in FakeClient.store else "created"] += 1
            FakeClient.store[key] = e
        return out


def test_run_persona_mode_end_to_end(catalog, monkeypatch):
    monkeypatch.setattr(server_run, "ServerClient", FakeClient)
    FakeClient.store, FakeClient.users = {}, {}
    b = _behavior(catalog)
    r = server_run.run("http://x", N, 2, 5, "t1", behavior=b)
    ev = list(FakeClient.store.values())
    setup = [e for e in ev if e.get("is_setup")]
    assert len(setup) == sum(len(c["initial_purchase_keys"]) for c in b.customers)
    assert all(e["type"] == "purchase" and "request_id" not in e for e in setup)
    views = [e for e in ev if e["type"] == "view"]
    assert len(views) == N * 2 * 5 and r["events_error"] == 0
    acts = [e for e in ev if e["type"] in ("click", "add_to_cart", "purchase") and not e.get("is_setup")]
    assert all(e["request_id"] and e["position"] for e in acts)
    assert r["behavior"]["kind"] == "persona_v0.3" and {p for _, p in FakeClient.users.values()} <= set(b.personas)
    # 행동 이벤트는 세션마다 서버에 먼저 보낸다 (다음 세션 추천이 이 세션 클릭을 보도록) — 초기 이력은 첫 추천 전
    first_rec = min(i for i, e in enumerate(ev) if e["type"] == "view")
    assert all(e.get("is_setup") for e in ev[:first_rec]) and first_rec > 0


def test_run_rejects_customer_count_mismatch(catalog, monkeypatch):
    monkeypatch.setattr(server_run, "ServerClient", FakeClient)
    with pytest.raises(ValueError, match="페르소나 고객"):
        server_run.run("http://x", N + 1, 1, 5, "t2", behavior=_behavior(catalog))
