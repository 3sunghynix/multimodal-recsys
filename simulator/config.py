"""Load the team's persona YAML; reject unsupported or inconsistent settings."""
from pathlib import Path
import math

import yaml


FEATURES = {"brand_affinity", "price_affinity", "novelty_affinity", "repurchase_affinity"}


def probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name}: finite probability in [0, 1] required")
    return float(value)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    pilot, behavior = config["pilot"], config["behavior"]
    if pilot.get("write_events") is not False:
        raise ValueError("This local runner requires write_events: false; API writes are not implemented")
    active = pilot["active_personas"]
    ids = [p["id"] for p in config["personas"]]
    if len(set(ids)) != len(ids) or len(set(active)) != len(active) or not active:
        raise ValueError("Persona IDs must be unique; active list must not be empty")
    by_id = {p["id"]: p for p in config["personas"]}
    for name in active:
        if name not in by_id or not by_id[name]["pilot_enabled"]:
            raise ValueError(f"Inactive or unknown persona: {name}")
        p = by_id[name]
        weights = p["preference_weights"]
        if not set(weights) <= FEATURES or not set(p["required_features"]) <= set(weights):
            raise ValueError(f"Unsupported or missing features for {name}; deferred personas cannot run yet")
        if not math.isclose(sum(probability(w, name) for w in weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"Preference weights must sum to 1: {name}")
    for key in ("customers_per_persona", "sessions_per_customer", "recommendations_per_session"):
        if type(pilot[key]) is not int or pilot[key] <= 0:
            raise ValueError(f"pilot.{key}: positive integer required")
    if pilot["sessions_per_customer"] != 1:
        raise ValueError("v0.1 supports one frozen-history session per customer only")
    c = behavior["click"]
    for key in ("base_probability", "minimum_probability", "maximum_probability"):
        probability(c[key], key)
    if c["minimum_probability"] > c["maximum_probability"]:
        raise ValueError("Invalid click probability limits")
    if not math.isfinite(c["preference_strength"]) or c["preference_strength"] < 0:
        raise ValueError("preference_strength must be finite and nonnegative")
    for action in ("add_to_cart", "purchase"):
        probability(behavior[action]["probability"], action)
    if behavior["add_to_cart"]["conditional_on"] != "click_on_same_item_in_current_session" or behavior["purchase"]["conditional_on"] != "add_to_cart_on_same_item_in_current_session":
        raise ValueError("Only the view -> click -> add_to_cart -> purchase funnel is supported")
    if any(behavior.get(k) is not False for k in ("use_model_score_as_behavior_input", "use_variant_as_behavior_input", "use_model_name_as_behavior_input")):
        raise ValueError("Behavior must not depend on model score, name, or variant")
    if config["features"]["brand_affinity"]["use_popularity_weighted_brand_sampling"]:
        raise ValueError("Brand sampling must be uniform")
    for key in ("brand_affinity", "price_affinity", "novelty_affinity", "repurchase_affinity"):
        if key not in config["features"]:
            raise ValueError(f"Missing feature definition: {key}")
    brands = config["features"]["brand_affinity"]
    if type(brands["min_train_items_per_brand_in_category"]) is not int or brands["min_train_items_per_brand_in_category"] <= 0:
        raise ValueError("Minimum train products per brand must be a positive integer")
    if not brands["preferred_brand_count_candidates"] or any(type(n) is not int or not 1 <= n <= 3 for n in brands["preferred_brand_count_candidates"]):
        raise ValueError("Preferred brand counts must be 1..3")
    price = config["features"]["price_affinity"]
    qs, ws = price["budget_quantiles"], price["budget_sampling_weights"]
    if not qs or len(qs) != len(ws):
        raise ValueError("Budget quantiles and sampling weights must match")
    for q in qs:
        probability(q, "budget_quantile")
    if not math.isclose(sum(probability(w, "budget_sampling_weight") for w in ws), 1, abs_tol=1e-9):
        raise ValueError("Budget sampling weights must sum to 1")
    if type(pilot["seed"]) is not int or pilot["seed"] < 0:
        raise ValueError("Seed must be a nonnegative integer")
    return config
