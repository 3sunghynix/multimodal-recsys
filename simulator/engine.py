"""Pure local behavior engine and descriptive reports; no HTTP/DB side effects."""
from collections import Counter, defaultdict
import math
import uuid

import numpy as np

from recsys.eval.online import dominant_brand_share

from .data import derived_seed


def customer_category_plan(data, minimum):
    """Eligible categories only; weight by positive train purchase rows, not catalog size."""
    setup = data.items.loc[data.items.item_key.isin(data.positive.item_key)]
    categories = []
    for category, pool in setup.groupby("category", sort=True):
        sizes = pool.groupby("brand_norm").item_key.nunique()
        brands = sorted(sizes[sizes >= minimum].index)
        if brands:
            count = int(data.positive.item_key.isin(pool.item_key).sum())
            categories.append({"category": str(category), "brands": brands, "positive_train_purchases": count})
    total = sum(row["positive_train_purchases"] for row in categories)
    for row in categories:
        row["probability"] = row["positive_train_purchases"] / total
    return setup, categories


def make_customers(data, config, count=None, history_size=10):
    if type(history_size) is not int or not 3 <= history_size <= 50:
        raise ValueError("history_size must be 3..50 (local pilot proposal)")
    count = config["pilot"]["customers_per_persona"] if count is None else count
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer per persona")
    seed = config["pilot"]["seed"]
    features = config["features"]
    minimum = features["brand_affinity"]["min_train_items_per_brand_in_category"]
    setup, plan = customer_category_plan(data, minimum)
    data.stats["customer_category_sampling"] = {
        "rule": "eligible_categories_weighted_by_positive_train_purchase_count",
        "eligibility": "at_least_one_brand_with_minimum_distinct_positive_train_items",
        "min_train_items_per_brand_in_category": minimum,
        "scope": "provided usable catalog; positive train purchases only; source-specific IDs retained",
        "categories": plan,
        "excluded_categories": sorted(set(setup.category) - {p["category"] for p in plan}),
    }
    customers, skipped = [], []
    for persona in config["pilot"]["active_personas"]:
        for index in range(count):
            cid = f"sim-{persona}-{index:04d}"
            rng = np.random.default_rng(derived_seed(seed, "customer", cid))
            if not plan:
                skipped.append({"customer_id": cid, "persona": persona, "category": None,
                                "reason": "no_eligible_category_with_minimum_train_items"})
                continue
            selection = plan[int(rng.choice(len(plan), p=[row["probability"] for row in plan]))]
            category, brands = selection["category"], selection["brands"]
            pool = setup.loc[setup.category == category]
            n = min(int(rng.choice(features["brand_affinity"]["preferred_brand_count_candidates"])), len(brands))
            chosen = sorted(rng.choice(brands, size=n, replace=False).tolist())
            eligible = pool.loc[pool.brand_norm.isin(chosen)]
            # At least one item per selected brand; no fabricated product or cross-source identity.
            history = [str(rng.choice(sorted(eligible.loc[eligible.brand_norm == b, "item_key"]))) for b in chosen]
            rest = sorted(set(eligible.item_key) - set(history))
            remaining = min(max(history_size - len(history), 0), len(rest))
            history += rng.choice(rest, size=remaining, replace=False).tolist()
            quantiles = features["price_affinity"]["budget_quantiles"]
            weights = features["price_affinity"]["budget_sampling_weights"]
            q = float(rng.choice(quantiles, p=weights))
            budget = float(pool.price.quantile(q))
            customers.append({"customer_id": cid, "source": "sim", "persona": persona,
                              "preferred_category": category, "category_sampling_probability": selection["probability"],
                              "preferred_brands": chosen,
                              "budget": budget, "budget_quantile": q, "initial_purchase_keys": history,
                              "setup_seed": derived_seed(seed, "customer", cid), "setup_kind": "synthetic_train_only"})
    return customers, skipped


def item_features(customer, item, item_by_key):
    history = customer["initial_purchase_keys"]
    seen_brands = {item_by_key[k]["brand_norm"] for k in history}
    return {
        "brand_affinity": float(item["brand_norm"] in customer["preferred_brands"]),
        "price_affinity": float(np.clip(1 - item["price"] / customer["budget"], 0, 1)),
        "novelty_affinity": float(item["item_key"] not in history and item["brand_norm"] not in seen_brands),
        "repurchase_affinity": float(item["item_key"] in history),
    }


def click_probability(score, config):
    c = config["behavior"]["click"]
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Preference score must be in [0, 1]")
    # Clamping in log space avoids overflow for a large (but finite) strength.
    log_p = math.log(c["base_probability"]) + c["preference_strength"] * (score - 0.5) if c["base_probability"] else -math.inf
    p = math.exp(min(0.0, log_p))
    return min(c["maximum_probability"], max(c["minimum_probability"], p))


def local_popularity_responses(customers, data, k):
    """Train-only local fixture baseline, not the production DB popularity model."""
    items = data.items.copy()
    items["rank_score"] = items.groupby("source").positive_train_count.rank(method="average", pct=True)
    top = items.sort_values(["rank_score", "source", "item_id"], ascending=[False, True, True]).item_key.head(k).tolist()
    return [{"customer_id": c["customer_id"], "session_id": "s0000", "recommendation_id": f"local-{c['customer_id']}",
             "model": "popularity", "provider": "local_train_positive_counts_per_source_rank",
             "model_version_id": None, "experiment_id": None, "variant": None, "items": top}
            for c in customers]


def simulate(customers, responses, data, config, run_id):
    """Replay source-qualified recommendation lists; preserve attribution verbatim.

    Draws depend on seed/customer/session/rank/action, never model name/score/variant.
    Replaying multiple models is counterfactual/offline, not randomized server A/B.
    """
    people = {c["customer_id"]: c for c in customers}
    products = {row["item_key"]: row for row in data.items.to_dict("records")}
    personas = {p["id"]: p for p in config["personas"]}
    decisions, events, seen, request_ids, assignments = [], [], set(), set(), {}
    k = config["pilot"]["recommendations_per_session"]
    for response in responses:
        cid = response["customer_id"]
        if cid not in people:
            raise ValueError(f"Unknown synthetic customer: {cid}")
        customer = people[cid]
        if response.get("session_id") != "s0000":
            raise ValueError("Only the single pilot session s0000 is supported")
        if not isinstance(response.get("recommendation_id"), str) or not response["recommendation_id"]:
            raise ValueError("A nonempty local recommendation_id is required")
        request_key = (cid, response["recommendation_id"])
        if request_key in request_ids:
            raise ValueError("recommendation_id must be unique per customer across replayed models")
        request_ids.add(request_key)
        if not isinstance(response.get("model"), str) or not response["model"]:
            raise ValueError("A model label is required for reporting, not for behavior")
        for key in ("model_version_id", "experiment_id", "variant"):
            if key not in response:
                raise ValueError(f"Missing attribution field {key}; use null for local lists")
        for key in ("model_version_id", "experiment_id"):
            if response[key] is not None and (type(response[key]) is not int or response[key] <= 0):
                raise ValueError(f"{key}: positive integer or null required")
        exp, variant = response["experiment_id"], response["variant"]
        if exp is not None:
            if not isinstance(variant, str) or not variant or response["model_version_id"] is None:
                raise ValueError("Experiment responses need a variant and model_version_id")
            group = (variant, response["model_version_id"])
            assignment_key = (cid, exp)
            if assignment_key in assignments and assignments[assignment_key] != group:
                raise ValueError("Same customer/experiment cannot change variant or model version")
            assignments[assignment_key] = group
        elif variant is not None:
            raise ValueError("A variant without experiment_id is ambiguous")
        identity = (cid, response["session_id"], response["model"], response["model_version_id"], exp, variant)
        if identity in seen:
            raise ValueError("Duplicate customer/session/model recommendation; refusing double-counting")
        seen.add(identity)
        keys = response["items"]
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise ValueError("items must be source-qualified raw keys such as amazon:93813, not DB IDs")
        if len(keys) > k:
            raise ValueError(f"More than {k} displayed items; truncate explicitly before replay")
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate products within a recommendation")
        if set(keys) - set(products):
            raise ValueError(f"Unknown or unusable products: {sorted(set(keys) - set(products))}")
        for rank, key in enumerate(keys, start=1):
            item = products[key]
            f = item_features(customer, item, products)
            score = sum(w * f[name] for name, w in personas[customer["persona"]]["preference_weights"].items())
            probs = {"click": click_probability(score, config),
                     "add_to_cart": config["behavior"]["add_to_cart"]["probability"],
                     "purchase": config["behavior"]["purchase"]["probability"]}
            draws = {a: float(np.random.default_rng(derived_seed(config["pilot"]["seed"], "behavior", cid, "s0000", rank, a)).random()) for a in probs}
            actions = ["view"]
            if draws["click"] < probs["click"]:
                actions.append("click")
                if draws["add_to_cart"] < probs["add_to_cart"]:
                    actions.append("add_to_cart")
                    if draws["purchase"] < probs["purchase"]:
                        actions.append("purchase")
            exposure = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}/{response['recommendation_id']}/{cid}/{rank}/{key}"))
            common = {"run_id": run_id, "customer_id": cid, "persona": customer["persona"], "session_id": "s0000",
                      "recommendation_id": response["recommendation_id"], "local_impression_id": exposure,
                      "item_key": key, "source": item["source"], "item_id": int(item["item_id"]),
                      "brand_norm": item["brand_norm"], "price_source": str(item["price_source"]), "rank": rank,
                      "model": response["model"], "model_version_id": response["model_version_id"],
                      "experiment_id": exp, "variant": variant,
                      "request_id": response.get("request_id"), "server_user_id": response.get("server_user_id")}
            decisions.append({**common, "features": f, "preference_score": score, "probabilities": probs,
                              "uniform_draws": draws, "actions": actions})
            for action in actions:
                events.append({**common, "type": action,
                               "local_event_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{exposure}/{action}")),
                               "logical_order": len(events), "delivery_status": "local_only_not_sent"})
    return decisions, events


def summarize(customers, skipped, responses, events, config):
    def group_of(row):
        return (row["persona"], row["model"], row["model_version_id"], row["experiment_id"], row["variant"])
    people = {c["customer_id"]: c for c in customers}
    groups = defaultdict(list)
    for r in responses:
        g = group_of({**r, "persona": people[r["customer_id"]]["persona"]})
        groups[g].append(r)
    event_groups = defaultdict(list)
    for e in events:
        event_groups[group_of(e)].append(e)
    rows = []
    for g, requests in groups.items():
        ev = event_groups[g]
        counts = Counter(e["type"] for e in ev)
        views = [e for e in ev if e["type"] == "view"]
        n = len(views)
        customer_ids = sorted({r["customer_id"] for r in requests})
        row = dict(zip(("persona", "model", "model_version_id", "experiment_id", "variant"), g))
        row.update({"customers": len(customer_ids), "requests": len(requests), "empty_responses": sum(not r["items"] for r in requests),
                    "views": n, "clicks": counts["click"], "carts": counts["add_to_cart"], "purchases": counts["purchase"],
                    "ctr": counts["click"] / n if n else None,
                    "purchase_per_view": counts["purchase"] / n if n else None,
                    "session_cart_rate": sum(any(e["customer_id"] == cid and e["type"] == "add_to_cart" for e in ev) for cid in customer_ids) / len(requests),
                    "session_purchase_rate": sum(any(e["customer_id"] == cid and e["type"] == "purchase" for e in ev) for cid in customer_ids) / len(requests),
                    "source_view_counts": dict(Counter(e["source"] for e in views)),
                    "price_source_view_counts": dict(Counter(e["price_source"] for e in views)),
                    "estimated_price_view_share": sum(e["price_source"] != "real" for e in views) / n if n else None,
                    "customer_ids": customer_ids})
        for action in ("view", "click", "purchase"):
            by_customer = defaultdict(list)
            for e in ev:
                if e["type"] == action:
                    by_customer[e["customer_id"]].append(e["brand_norm"])
            result = dominant_brand_share(by_customer, event_type=action)
            row[f"dominant_brand_share_{action}"] = result["value"]
            row[f"dominant_brand_share_{action}_customers"] = result["n_users"]
        rows.append(row)
    baseline = config["pilot"]["baseline_model_name"]
    for row in rows:
        bases = [b for b in rows if b["persona"] == row["persona"] and b["model"] == baseline and b["experiment_id"] == row["experiment_id"]]
        row["baseline_comparison"] = {"status": "no_unique_baseline"}
        if len(bases) == 1 and bases[0]["ctr"] is not None and row["ctr"] is not None:
            b = bases[0]
            paired = row["customer_ids"] == b["customer_ids"]
            diff = row["ctr"] - b["ctr"]
            row["baseline_comparison"] = {"status": "descriptive_only", "same_customer_set": paired,
                                          "ctr_difference_pp": 100 * diff,
                                          "relative_ctr_uplift": diff / b["ctr"] if b["ctr"] else None,
                                          "zero_baseline": b["ctr"] == 0}
    return {"kind": "local_synthetic_exploratory_not_server_ab", "calibrated": False,
            "notes": ["No HTTP/DB writes; synthetic events are not real-user evidence",
                      "Compare models within each persona, not raw CTR across personas",
                      "Brand-loyal is a stress test; source/date windows frozen from train",
                      "Shared customer/rank/action draws couple replay comparisons; not independent AB samples",
                      "No confidence intervals or automatic winner selection in this pilot"],
            "generated_customers": len(customers), "skipped_customers": skipped,
            "disabled_personas": config["pilot"]["deferred_personas"], "groups": rows}
