"""Run one randomized simulator session per customer against the loaded catalog."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .ab import compare_customers
from .config import load_config
from .data import derived_seed, item_key, prepare
from .engine import make_customers, simulate
from .queries import ITEM_WORDS, image_keys, image_query_item, mark_has_image, text_query


def run(catalog, output, count, seed, api):
    # Backend imports stay inside run(): the API container provides them, and summarize() stays testable without it.
    import httpx
    from sqlalchemy import select

    from app import recommenders
    from app.db import SessionLocal
    from app.models import Event, Item, ModelVersion, User

    started = time.perf_counter()
    config = load_config(Path(__file__).with_name("personas.yaml"))
    config["pilot"].update(planned_customers=count, seed=seed)
    items = pd.read_parquet(catalog / "items.parquet")
    events = pd.read_parquet(catalog / "events.parquet")
    if "split" not in events:
        raise ValueError("Catalog events must have a split column to verify train-only history")
    data = prepare(items, events)
    customers, skipped = make_customers(data, config)
    if len(customers) != count:
        raise ValueError(f"Generated {len(customers)} of {count} customers: {skipped[:3]}")

    assignments = {}
    offset = 0
    for persona in config["pilot"]["active_personas"]:
        ids = [c["customer_id"] for c in customers if c["persona"] == persona]
        rng = np.random.default_rng(derived_seed(seed, "assignment", persona))
        for i, cid in enumerate(rng.permutation(ids)):
            assignments[cid] = "AB"[(offset + i) % 2]
        offset += len(ids)

    key_by_external = {(r.source, str(r.source_item_id)): item_key(r.source, r.item_id)
                       for r in items.itertuples()}
    usable = set(data.items.item_key)
    # #47: image_query_item needs has_image; mark it once from the extracted images folder (or images.zip).
    marked = mark_has_image(data.items, image_keys(catalog))
    query_items = {leaf: group for leaf, group in marked[marked.has_image].groupby("category_l3")}
    responses, searches = [], []
    with SessionLocal() as db, httpx.Client(base_url=api, timeout=120) as client:
        models = {}
        for arm, name in (("A", "popularity"), ("B", "ume")):
            stmt = select(ModelVersion).where(ModelVersion.name == name)
            if arm == "B":
                stmt = stmt.where(ModelVersion.status == "active")
            model = db.scalars(stmt.order_by(ModelVersion.id.desc())).first()
            if model is None:
                raise ValueError(f"Missing model version for {name}")
            models[arm] = model
        db_to_key = {}
        for db_id, source, external_id in db.execute(select(Item.id, Item.source, Item.source_item_id)):
            key = key_by_external.get((source, external_id))
            if key is not None:
                db_to_key[db_id] = key
        key_to_db = {key: db_id for db_id, key in db_to_key.items()}
        if len(key_to_db) != len(items):
            raise ValueError("The database and catalog product IDs do not match")

        for c in customers:
            cid = c["customer_id"]
            arm = assignments[cid]
            model = models[arm]
            user = User(source="sim", persona=c["persona"])
            db.add(user)
            db.flush()
            history = c["initial_purchase_keys"]
            db.add_all(Event(user_id=user.id, item_id=key_to_db[key], type="purchase", is_setup=True)
                       for key in history)
            db.flush()
            recommender = recommenders.get(model.name, model.params)
            k = config["pilot"]["recommendations_per_session"]
            # Score through the production recommenders. Keep only products the behavior engine can score.
            scored = recommender.recommend_scored(db, user.id, k)
            keys = [db_to_key.get(db_id) for db_id, _ in scored]
            displayed = [key for key in keys if key in usable]
            if not displayed:
                raise ValueError(f"No usable recommendations for {cid} ({model.name})")
            responses.append(dict(customer_id=cid, session_id="s0000", recommendation_id=cid,
                                  variant=arm, experiment_id=None, model=model.name,
                                  model_version_id=model.id, items=displayed,
                                  raw_count=len(keys), fallback_reason=getattr(recommender, "fallback_reason", None)))

            rng = np.random.default_rng(derived_seed(seed, "search", cid))
            leaf = c["preferred_category"].split(" > ")[-1]
            search_leaf = leaf if leaf in ITEM_WORDS else str(rng.choice(sorted(query_items.keys() & ITEM_WORDS.keys())))
            pool = query_items.get(search_leaf)
            source = history[0].split(":")[0] if history else None
            image_key = image_query_item(rng, pool, search_leaf, source) if pool is not None and rng.random() < .25 else None
            if image_key:
                src, raw_id = image_key
                path = catalog / "images" / src / f"{raw_id}.jpg"
                with path.open("rb") as photo:
                    response = client.post("/search", files={"file": (path.name, photo, "image/jpeg")}, data={"k": 6})
                exclude = str(key_to_db[item_key(src, raw_id)])
                query = None
            else:
                query = text_query(rng, c["persona"], search_leaf, brand=c["preferred_brands"][0])["query"]
                response = client.post("/search", data={"q": query, "k": 5})
                exclude = None
            response.raise_for_status()
            hits = [hit["product_id"] for hit in response.json()["results"] if hit["product_id"] != exclude][:5]
            if not hits:
                raise ValueError(f"Search returned no products for {cid}")
            searches.append(dict(type="search", customer_id=cid, persona=c["persona"],
                                 variant=arm, session_id="s0000", search_type="image" if image_key else "text",
                                 query=query, category_leaf=search_leaf, image_key=image_key, product_ids=hits))
        db.rollback()

    replay = [{**row, "variant": None} for row in responses]
    decisions, actions = simulate(customers, replay, data, config, output.name)
    for row in decisions + actions:
        row["variant"] = assignments[row["customer_id"]]
    summary = summarize(customers, assignments, responses, searches, actions,
                        seed=seed, catalog_products=len(items))
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    summary["seconds_per_customer"] = round(summary["elapsed_seconds"] / count, 3)
    output.mkdir(parents=True, exist_ok=False)
    for name, records in (("customers", customers), ("recommendations", responses),
                          ("events", searches + actions), ("decisions", decisions)):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "catalog_ab_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def summarize(customers, assignments, responses, searches, actions, *, seed, catalog_products, min_per_arm=50):
    """Overall and per-persona first-session click tests plus counts. Pure: no database or API access.

    Per-persona tests are unadjusted for multiple comparisons; read them as a breakdown, not as separate claims."""
    persona = {c["customer_id"]: c["persona"] for c in customers}
    history = {c["customer_id"]: len(c["initial_purchase_keys"]) for c in customers}
    clicked = {r["customer_id"] for r in actions if r["type"] == "click"}

    def test(ids, interpretation):
        rows = [dict(customer_id=cid, variant=assignments[cid], clicked_first_session=cid in clicked) for cid in ids]
        if {r["variant"] for r in rows} != {"A", "B"}:
            return None
        result = compare_customers(rows, min_per_arm=min_per_arm)
        result["interpretation"] = interpretation
        return result

    comparison = test(sorted(assignments), "Overall pilot. Simulated customer behavior on real catalog products.")
    fallback = {r["customer_id"] for r in responses if r["variant"] == "B" and r.get("fallback_reason") == "popular"}
    by_persona = {}
    for p in sorted(set(persona.values())):
        ids = sorted(cid for cid in assignments if persona[cid] == p)
        by_persona[p] = dict(
            comparison=test(ids, "Per-persona breakdown, unadjusted for multiple comparisons."),
            cold_start_customers=sum(history[cid] == 0 for cid in ids),
            ume_fallback_count=sum(cid in fallback for cid in ids),
        )
    groups = {}
    for arm in "AB":
        counts = Counter(r["type"] for r in actions if r["variant"] == arm)
        counts["search"] = sum(r["variant"] == arm for r in searches)
        groups[arm] = dict(counts)
    distinct = {arm: len({i for r in responses if r["variant"] == arm for i in r["items"]}) for arm in "AB"}
    b = [r for r in responses if r["variant"] == "B"]
    return dict(kind="real_catalog_ab_pilot", seed=seed, catalog_products=catalog_products,
                customers=len(assignments), comparison=comparison, by_persona=by_persona, groups=groups,
                distinct_recommended_items=distinct,
                cold_start_customers=sum(n == 0 for n in history.values()),
                ume_fallback_count=len(fallback), ume_requests=len(b),
                search_types=dict(Counter(r["search_type"] for r in searches)),
                excluded_recommendations=sum(r["raw_count"] - len(r["items"]) for r in responses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("/data/full_catalog"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--customers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.customers < 2 or args.out.exists():
        parser.error("Use at least two customers and a new output folder")
    print(json.dumps(run(args.catalog, args.out, args.customers, args.seed, args.api), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
