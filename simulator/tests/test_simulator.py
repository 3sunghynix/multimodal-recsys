import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from simulator.config import load_config
from simulator.data import prepare
from simulator.engine import customer_category_plan, click_probability, item_features, local_popularity_responses, make_customers, simulate, summarize
from simulator.__main__ import run

CONFIG = Path(__file__).parents[1] / "personas.yaml"


@pytest.fixture
def config():
    return load_config(CONFIG)


@pytest.fixture
def tables():
    rows, events = [], []
    for source, year in (("amazon", 2023), ("hm", 2020)):
        for i in range(6):
            rows.append(dict(source=source, item_id=i, brand_norm="a" if i < 3 else "b",
                             cat_target="Women", cat_type="Clothing", price=10.0 + i, price_source="real"))
            events.append(dict(source=source, item_id=i, split="train", feedback=1.0,
                               ts=f"{year}-08-{i + 1:02d}", type="purchase"))
        events.append(dict(source=source, item_id=0, split="test", feedback=1.0, ts=f"{year}-09-01", type="purchase"))
    return pd.DataFrame(rows), pd.DataFrame(events)


@pytest.fixture
def prepared(tables):
    return prepare(*tables)


def test_source_keys_do_not_collide(prepared):
    assert len(set(prepared.items.item_key)) == 12
    assert {"amazon:0", "hm:0"} <= set(prepared.items.item_key)


def test_setup_is_deterministic_train_only(tables, config):
    items, events = tables
    first = prepare(items, events)
    c1, s1 = make_customers(first, config, count=3)
    events.loc[events.split == "test", ["feedback", "ts"]] = [99999, "invalid future timestamp"]
    second = prepare(items, events)
    c2, s2 = make_customers(second, config, count=3)
    assert (c1, s1) == (c2, s2)
    pd.testing.assert_frame_equal(first.items, second.items)
    assert len(c1) == 12


def test_nonpositive_train_events_not_initial_history(tables, config):
    items, events = tables
    events.loc[(events.source == "amazon") & (events.item_id == 0), "feedback"] = -1
    data = prepare(items, events)
    customers, _ = make_customers(data, config, count=10)
    assert all("amazon:0" not in c["initial_purchase_keys"] for c in customers)


def test_popularity_uses_each_source_calendar_window(tables):
    items, events = tables
    extra = events.iloc[[0]].copy()
    extra.loc[:, "ts"] = "2023-07-30"  # before Aug 6's seven-calendar-day window
    extra.loc[:, "item_id"] = 0
    data = prepare(items, pd.concat([events, extra], ignore_index=True))
    assert data.stats["popularity_windows"]["hm"]["end_exclusive"].startswith("2020-08-07")
    assert data.stats["popularity_windows"]["amazon"]["end_exclusive"].startswith("2023-08-07")
    assert data.stats["popularity_windows"]["amazon"]["positive_rows"] == 6
    assert np.allclose(data.items.popularity_affinity, 0.5)


def test_unknown_brands_cannot_form_customers(tables, config):
    items, events = tables
    items.loc[:, "brand_norm"] = ""
    with pytest.raises(ValueError, match="No positive train"):
        prepare(items, events)


def test_missing_brand_pool_reports_skip_instead_of_switching(tables, config):
    items, events = tables
    items.loc[:, "brand_norm"] = [f"b{i}" for i in range(len(items))]
    customers, skipped = make_customers(prepare(items, events), config, count=2)
    assert customers == [] and len(skipped) == 8
    assert all("no_eligible_category" in s["reason"] for s in skipped)


def test_feature_definitions(prepared):
    products = {i["item_key"]: i for i in prepared.items.to_dict("records")}
    c = dict(initial_purchase_keys=["amazon:0"], preferred_brands=["a"], budget=20)
    f = item_features(c, products["amazon:0"], products)
    assert f == dict(brand_affinity=1, price_affinity=0.5, novelty_affinity=0, repurchase_affinity=1)
    assert item_features(c, products["amazon:1"], products)["novelty_affinity"] == 0  # new item, known brand
    assert item_features(c, products["amazon:3"], products)["novelty_affinity"] == 1


def test_click_formula_neutral_and_bounds(config):
    assert click_probability(0.5, config) == pytest.approx(0.04)
    assert click_probability(1, config) > click_probability(0, config)
    with pytest.raises(ValueError):
        click_probability(float("nan"), config)


def test_events_preserve_attribution_and_funnel(prepared, config):
    customers, _ = make_customers(prepared, config, count=1)
    responses = local_popularity_responses(customers, prepared, 10)
    responses[0].update(experiment_id=9, variant="B", model_version_id=7, request_id="server-request")
    certain = copy.deepcopy(config)
    certain["behavior"]["click"].update(base_probability=1, minimum_probability=1, maximum_probability=1)
    certain["behavior"]["add_to_cart"]["probability"] = 1
    certain["behavior"]["purchase"]["probability"] = 1
    decisions, ev = simulate(customers, responses[:1], prepared, certain, "run")
    assert len(ev) == 40
    assert [e["type"] for e in ev[:4]] == ["view", "click", "add_to_cart", "purchase"]
    assert all(e["experiment_id"] == 9 and e["variant"] == "B" and e["model_version_id"] == 7 and e["request_id"] == "server-request" for e in ev)
    assert len({e["local_event_id"] for e in ev}) == len(ev)
    assert all(e["delivery_status"] == "local_only_not_sent" for e in ev)
    assert ev[0]["local_impression_id"] == ev[3]["local_impression_id"]


def test_model_variant_does_not_change_behavior(prepared, config):
    customers, _ = make_customers(prepared, config, count=1)
    original = local_popularity_responses(customers, prepared, 10)
    changed = copy.deepcopy(original)
    for r in changed:
        r.update(model="ume", model_version_id=9, experiment_id=1, variant="B", score=9999)
    d1, _ = simulate(customers, original, prepared, config, "a")
    d2, _ = simulate(customers, changed, prepared, config, "b")
    assert [(d["actions"], d["uniform_draws"], d["preference_score"]) for d in d1] == [(d["actions"], d["uniform_draws"], d["preference_score"]) for d in d2]


@pytest.mark.parametrize("bad", ["duplicate_item", "db_id", "unknown_product", "duplicate_response", "variant_change", "too_many"])
def test_invalid_recommendations_are_rejected(prepared, config, bad):
    customers, _ = make_customers(prepared, config, count=1)
    r = local_popularity_responses(customers, prepared, 10)[:1]
    if bad == "duplicate_item":
        r[0]["items"] = ["amazon:0", "amazon:0"]
    elif bad == "db_id":
        r[0]["items"] = [1]
    elif bad == "unknown_product":
        r[0]["items"] = ["amazon:999"]
    elif bad == "duplicate_response":
        r.append(copy.deepcopy(r[0]))
    elif bad == "too_many":
        r[0]["items"] = prepared.items.item_key.tolist()
    else:
        r[0].update(experiment_id=1, variant="A", model_version_id=1)
        other = copy.deepcopy(r[0])
        other.update(model="ume", recommendation_id="new", variant="B", model_version_id=2)
        r.append(other)
    with pytest.raises(ValueError):
        simulate(customers, r, prepared, config, "run")


def test_empty_response_does_not_fabricate_views(prepared, config):
    customers, _ = make_customers(prepared, config, count=1)
    responses = local_popularity_responses(customers, prepared, 10)[:1]
    responses[0]["items"] = []
    d, e = simulate(customers, responses, prepared, config, "run")
    summary = summarize(customers, [], responses, e, config)
    row = summary["groups"][0]
    assert not d and not e and row["empty_responses"] == 1
    assert row["views"] == 0 and row["ctr"] is None
    assert row["dominant_brand_share_purchase"] is None


def test_zero_baseline_uplift_undefined(prepared, config):
    customers, _ = make_customers(prepared, config, count=1)
    responses = local_popularity_responses(customers, prepared, 10)[:1]
    never = copy.deepcopy(config)
    never["behavior"]["click"].update(base_probability=0, minimum_probability=0, maximum_probability=0)
    _, events = simulate(customers, responses, prepared, never, "run")
    row = summarize(customers, [], responses, events, never)["groups"][0]
    assert row["baseline_comparison"]["zero_baseline"]
    assert row["baseline_comparison"]["relative_ctr_uplift"] is None


@pytest.mark.parametrize("change", ["write_events", "weight_sum", "deferred"])
def test_config_rejects_unsupported_settings(tmp_path, config, change):
    if change == "write_events":
        config["pilot"]["write_events"] = True
    elif change == "weight_sum":
        config["personas"][0]["preference_weights"]["brand_affinity"] = 0.9
    else:
        config["pilot"]["active_personas"].append("new_item_seeking")
    f = tmp_path / "config.yaml"
    f.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(f)


def test_end_to_end_repeatability_and_no_overwrite(tmp_path, tables):
    items, events = tables
    folder = tmp_path / "data"
    folder.mkdir()
    items.to_parquet(folder / "items.parquet")
    events.to_parquet(folder / "events.parquet")
    one, two = tmp_path / "one", tmp_path / "two"
    run(folder, one, CONFIG, count=2)
    run(folder, two, CONFIG, count=2)
    for name in ("customers.json", "recommendations.jsonl", "decisions.jsonl", "events.jsonl", "summary.json"):
        assert (one / name).read_bytes() == (two / name).read_bytes()
    manifest = json.loads((one / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["server_writes"] is False
    with pytest.raises(FileExistsError):
        run(folder, one, CONFIG, count=2)


def test_category_plan_excludes_sparse_and_weights_train_rows(tables, config):
    items, events = tables
    # A second eligible category with twice the purchase count, plus an ineligible category.
    extra_items = items.loc[items.source == "amazon"].copy()
    extra_items["item_id"] += 20
    extra_items["cat_type"] = "Shoes"
    extra_events = events.loc[(events.source == "amazon") & (events.split == "train")].copy()
    extra_events["item_id"] += 20
    sparse = extra_items.iloc[[0]].copy()
    sparse["item_id"] = 99
    sparse["cat_type"] = "Sparse"
    sparse_event = extra_events.iloc[[0]].copy()
    sparse_event["item_id"] = 99
    data = prepare(pd.concat([items, extra_items, sparse], ignore_index=True),
                   pd.concat([events, extra_events, extra_events, extra_events, extra_events, sparse_event], ignore_index=True))
    _, plan = customer_category_plan(data, 3)
    by_name = {row["category"]: row for row in plan}
    assert set(by_name) == {"Women > Clothing", "Women > Shoes"}
    assert by_name["Women > Clothing"]["probability"] == pytest.approx(1 / 3)
    assert by_name["Women > Shoes"]["probability"] == pytest.approx(2 / 3)
    customers, skipped = make_customers(data, config, count=4)
    assert len(customers) == 16 and not skipped
    assert all(c["preferred_category"] != "Women > Sparse" for c in customers)
    assert "Women > Sparse" in data.stats["customer_category_sampling"]["excluded_categories"]
    assert sum(p["probability"] for p in plan) == pytest.approx(1)


def test_setup_ignores_input_row_order(tables, config):
    items, events = tables
    one, _ = make_customers(prepare(items, events), config, count=3)
    two, _ = make_customers(prepare(items.sample(frac=1, random_state=2), events.sample(frac=1, random_state=3)), config, count=3)
    assert one == two
