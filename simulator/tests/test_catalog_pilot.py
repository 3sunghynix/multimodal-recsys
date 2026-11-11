"""summarize() of the real-catalog A/B pilot. run() needs the API container (DB + search API) and is checked by running it."""
import importlib
import sys

import pytest

from simulator.catalog_pilot import summarize


def customer(cid, persona, history=1):
    return {"customer_id": cid, "persona": persona, "initial_purchase_keys": [f"hm:{i}" for i in range(history)]}


@pytest.fixture
def pilot():
    customers = [customer("c1", "brand_loyal"), customer("c2", "brand_loyal"),
                 customer("c3", "window_shopper", history=0), customer("c4", "window_shopper", history=0),
                 customer("c5", "window_shopper")]
    assignments = {"c1": "A", "c2": "B", "c3": "A", "c4": "B", "c5": "B"}
    responses = [dict(customer_id=cid, variant=arm, items=items, raw_count=len(items), fallback_reason=reason)
                 for cid, arm, items, reason in [("c1", "A", ["hm:1", "hm:2"], None),
                                                 ("c2", "B", ["hm:3", "hm:4"], None),
                                                 ("c3", "A", ["hm:1", "hm:2"], None),
                                                 ("c4", "B", ["hm:1", "hm:2"], "popular"),
                                                 ("c5", "B", ["hm:5", "hm:6"], None)]]
    searches = [dict(type="search", customer_id=cid, variant=arm, search_type="image" if cid == "c1" else "text")
                for cid, arm in assignments.items()]
    actions = [dict(type="view", customer_id=r["customer_id"], variant=r["variant"]) for r in responses for _ in r["items"]]
    actions += [dict(type="click", customer_id="c2", variant="B"), dict(type="click", customer_id="c2", variant="B"),
                dict(type="click", customer_id="c3", variant="A"), dict(type="purchase", customer_id="c3", variant="A")]
    return customers, assignments, responses, searches, actions


def test_overall_comparison_counts_customers_not_clicks(pilot):
    s = summarize(*pilot, seed=42, catalog_products=10, min_per_arm=1)
    assert s["kind"] == "real_catalog_ab_pilot" and s["customers"] == 5
    c = s["comparison"]
    assert (c["A"]["customers"], c["A"]["clicked_customers"]) == (2, 1)
    assert (c["B"]["customers"], c["B"]["clicked_customers"]) == (3, 1)   # c2 clicked twice, counted once
    assert c["difference_B_minus_A"] == pytest.approx(1 / 3 - 1 / 2)


def test_counts_fallback_cold_start_and_distinct_items(pilot):
    s = summarize(*pilot, seed=42, catalog_products=10, min_per_arm=1)
    assert s["groups"]["A"] == {"view": 4, "click": 1, "purchase": 1, "search": 2}
    assert s["groups"]["B"] == {"view": 6, "click": 2, "search": 3}
    assert s["ume_requests"] == 3 and s["ume_fallback_count"] == 1
    assert s["cold_start_customers"] == 2
    assert s["distinct_recommended_items"] == {"A": 2, "B": 6}   # B: hm:1-6 incl. c4 popular fallback
    assert s["search_types"] == {"image": 1, "text": 4}
    assert s["excluded_recommendations"] == 0


def test_per_persona_breakdown(pilot):
    s = summarize(*pilot, seed=42, catalog_products=10, min_per_arm=1)
    assert set(s["by_persona"]) == {"brand_loyal", "window_shopper"}
    loyal, window = s["by_persona"]["brand_loyal"], s["by_persona"]["window_shopper"]
    assert loyal["comparison"]["A"]["rate"] == 0 and loyal["comparison"]["B"]["rate"] == 1
    assert (loyal["cold_start_customers"], loyal["ume_fallback_count"]) == (0, 0)
    assert (window["cold_start_customers"], window["ume_fallback_count"]) == (2, 1)
    assert "unadjusted" in window["comparison"]["interpretation"]


def test_small_groups_report_no_p_value(pilot):
    s = summarize(*pilot, seed=42, catalog_products=10)                    # default min_per_arm=50
    assert s["comparison"]["p_value"] is None
    assert s["comparison"]["status"] == "insufficient_sample_or_expected_counts"


def test_persona_with_one_arm_has_no_comparison(pilot):
    customers, assignments, responses, searches, actions = pilot
    customers = customers + [customer("c6", "practical")]
    assignments = {**assignments, "c6": "A"}
    s = summarize(customers, assignments, responses, searches, actions, seed=42, catalog_products=10, min_per_arm=1)
    assert s["by_persona"]["practical"]["comparison"] is None


def test_module_imports_without_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "app", None)                          # backend package unavailable
    importlib.reload(importlib.import_module("simulator.catalog_pilot"))
