"""Run one randomized simulator session per customer against the loaded catalog.

A = 인기, B = UME, (선택) C = UME + MAB 탐색 슬롯. 모든 그룹에 GET /recommend와 같은 마지막 재정렬
(services/explore.py: 탐색 슬롯 · 같은 소분류 3개 연속 금지)을 적용한다."""
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


ARM_MODELS = {"A": ("popularity", None), "B": ("ume", None), "C": ("ume", "explore")}


def run(catalog, output, count, seed, api, arms="AB", batches=1, new_items=0, explore_cfg=None):
    """arms: "AB" (인기 vs UME) 또는 "ABC" (+ C = UME + MAB 탐색 슬롯).
    batches > 1이면 고객을 무작위 순서로 나눠 배치마다 행동을 재생하고, 그 노출 · 클릭을 메모리 집계에 더해
    다음 배치의 탐색 선택에 쓴다 (서비스에서 이벤트가 쌓이며 MAB가 배우는 것과 같은 흐름).
    new_items: 이 실행 안에서만 신규 상품(registered_at = 최근 7일)으로 지정할 상품 수 (끝에 rollback)."""
    # Backend imports stay inside run(): the API container provides them, and summarize() stays testable without it.
    from datetime import datetime, timedelta, timezone

    import httpx
    from sqlalchemy import select, update

    from app import recommenders
    from app.db import SessionLocal
    from app.models import Event, Item, ModelVersion, User
    from app.services import explore

    if set(arms) - set(ARM_MODELS) or "A" not in arms or "B" not in arms or len(set(arms)) != len(arms):
        raise ValueError("arms must be AB or ABC")
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
            assignments[cid] = arms[(offset + i) % len(arms)]
        offset += len(ids)
    order = customers if batches <= 1 else \
        [customers[i] for i in np.random.default_rng(derived_seed(seed, "batch_order")).permutation(len(customers))]
    chunks = [order] if batches <= 1 else [list(c) for c in np.array_split(np.array(order, dtype=object), batches)]

    key_by_external = {(r.source, str(r.source_item_id)): item_key(r.source, r.item_id)
                       for r in items.itertuples()}
    usable = set(data.items.item_key)
    # #47: image_query_item needs has_image; mark it once from the extracted images folder (or images.zip).
    marked = mark_has_image(data.items, image_keys(catalog))
    query_items = {leaf: group for leaf, group in marked[marked.has_image].groupby("category_l3")}
    responses, searches, decisions, actions, batch_log = [], [], [], [], []
    seen = {}                                                  # 메모리 집계 {db id: [노출, 클릭]} — 배치마다 갱신
    k = config["pilot"]["recommendations_per_session"]
    with SessionLocal() as db, httpx.Client(base_url=api, timeout=120) as client:
        models = {}
        for arm in arms:
            name, _ = ARM_MODELS[arm]
            stmt = select(ModelVersion).where(ModelVersion.name == name)
            if name == "ume":
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
        new_db_ids = []
        if new_items:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            new_rng = np.random.default_rng(derived_seed(seed, "new_items"))
            new_db_ids = sorted(int(i) for i in new_rng.choice(sorted(key_to_db.values()), size=new_items, replace=False))
            for i in new_db_ids:                                 # rollback 되므로 서비스 DB에는 남지 않는다
                db.execute(update(Item).where(Item.id == i)
                           .values(registered_at=now - timedelta(seconds=float(new_rng.uniform(0, 6 * 86400)))))
            db.flush()
        new_keys = {db_to_key[i] for i in new_db_ids}

        def stats_fn(ids):
            base = explore.item_stats(db, ids)
            return {i: (base.get(i, (0, 0))[0] + seen.get(i, [0, 0])[0], base.get(i, (0, 0))[1] + seen.get(i, [0, 0])[1])
                    for i in ids}

        for b, chunk in enumerate(chunks):
            batch_responses = []
            for c in chunk:
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
                # Score through the production recommenders + the same final re-ranking step as GET /recommend.
                scored = recommender.recommend_scored(db, user.id, k)
                params = {**(model.params or {}),
                          "explore": (explore_cfg or True) if ARM_MODELS[arm][1] == "explore" else None}
                recs = explore.apply(db, user.id, scored, k, params, stats_fn=stats_fn,
                                     rng=np.random.default_rng(derived_seed(seed, "explore", cid)))
                keys = [db_to_key.get(r.item_id) for r in recs]
                displayed = [key for key in keys if key in usable]
                if not displayed:
                    raise ValueError(f"No usable recommendations for {cid} ({model.name})")
                row = dict(customer_id=cid, session_id="s0000", recommendation_id=cid,
                           variant=arm, experiment_id=None, model=model.name,
                           model_version_id=model.id, items=displayed, batch=b,
                           exploration_items=[db_to_key[r.item_id] for r in recs
                                              if r.is_exploration and db_to_key.get(r.item_id) in usable],
                           raw_count=len(keys), fallback_reason=getattr(recommender, "fallback_reason", None))
                responses.append(row)
                batch_responses.append(row)

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

            replay = [{key: v for key, v in row.items() if key not in ("batch", "exploration_items")} | {"variant": None}
                      for row in batch_responses]
            d, a = simulate([c for c in chunk], replay, data, config, output.name)
            for r in d + a:
                r["variant"] = assignments[r["customer_id"]]
            decisions += d
            actions += a
            for r in d:                                          # 다음 배치의 탐색 선택에 반영 (서비스의 이벤트 누적)
                db_id = key_to_db.get(r["item_key"])
                if db_id is not None:
                    v = seen.setdefault(db_id, [0, 0])
                    v[0] += 1
                    v[1] += "click" in r["actions"]
            batch_log.append(_batch_stats(b, batch_responses, d))
        db.rollback()

    summary = summarize(customers, assignments, responses, searches, actions,
                        seed=seed, catalog_products=len(items), decisions=decisions, new_keys=new_keys)
    summary.update(arms=arms, batches=batch_log if "C" in arms else len(chunks), new_items=len(new_db_ids))
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


def _exploration_impressions(responses, decisions):
    """(탐색 슬롯 노출, 그중 클릭) — C 그룹 결정 중 탐색 상품만."""
    explo = {(r["customer_id"], key) for r in responses for key in r.get("exploration_items", [])}
    shown = [d for d in decisions if (d["customer_id"], d["item_key"]) in explo]
    return len(shown), sum("click" in d["actions"] for d in shown)


def _batch_stats(b, responses, decisions):
    views, clicks = _exploration_impressions(responses, decisions)
    c_ids = {r["customer_id"] for r in responses if r["variant"] == "C"}
    c_dec = [d for d in decisions if d["customer_id"] in c_ids]
    return dict(batch=b, customers=len(responses), exploration_views=views, exploration_clicks=clicks,
                exploration_ctr=clicks / views if views else None,
                c_ctr=sum("click" in d["actions"] for d in c_dec) / len(c_dec) if c_dec else None)


def summarize(customers, assignments, responses, searches, actions, *, seed, catalog_products, min_per_arm=50,
              decisions=None, new_keys=None):
    """Overall and per-persona first-session click tests plus counts. Pure: no database or API access.

    comparison = B vs A (인기 vs UME). C가 있으면 comparisons에 C vs B (탐색 슬롯 효과) · C vs A도 넣는다.
    Per-persona tests are unadjusted for multiple comparisons; read them as a breakdown, not as separate claims."""
    persona = {c["customer_id"]: c["persona"] for c in customers}
    history = {c["customer_id"]: len(c["initial_purchase_keys"]) for c in customers}
    clicked = {r["customer_id"] for r in actions if r["type"] == "click"}
    arms = sorted(set(assignments.values()))

    def test(ids, interpretation, control="A", treatment="B"):
        rows = [dict(customer_id=cid, variant={control: "A", treatment: "B"}[assignments[cid]],
                     clicked_first_session=cid in clicked) for cid in ids if assignments[cid] in (control, treatment)]
        if {r["variant"] for r in rows} != {"A", "B"}:
            return None
        result = compare_customers(rows, min_per_arm=min_per_arm)
        result["interpretation"] = interpretation
        if (control, treatment) != ("A", "B"):
            result["arms"] = {"A": control, "B": treatment}       # compare_customers 출력의 A · B가 가리키는 그룹
        return result

    pairs = [("A", "B")] + ([("B", "C"), ("A", "C")] if "C" in arms else [])
    overall = "Overall pilot. Simulated customer behavior on real catalog products."
    comparisons = {f"{t}_vs_{c}": test(sorted(assignments), overall, c, t) for c, t in pairs}
    fallback = {r["customer_id"] for r in responses if r["variant"] == "B" and r.get("fallback_reason") == "popular"}
    by_persona = {}
    for p in sorted(set(persona.values())):
        ids = sorted(cid for cid in assignments if persona[cid] == p)
        unadjusted = "Per-persona breakdown, unadjusted for multiple comparisons."
        entry = dict(comparison=test(ids, unadjusted),
                     cold_start_customers=sum(history[cid] == 0 for cid in ids),
                     ume_fallback_count=sum(cid in fallback for cid in ids))
        if "C" in arms:
            entry["comparisons"] = {f"{t}_vs_{c}": test(ids, unadjusted, c, t) for c, t in pairs[1:]}
        by_persona[p] = entry
    groups = {}
    for arm in arms:
        counts = Counter(r["type"] for r in actions if r["variant"] == arm)
        counts["search"] = sum(r["variant"] == arm for r in searches)
        groups[arm] = dict(counts)
    distinct = {arm: len({i for r in responses if r["variant"] == arm for i in r["items"]}) for arm in arms}
    b = [r for r in responses if r["variant"] == "B"]
    out = dict(kind="real_catalog_ab_pilot", seed=seed, catalog_products=catalog_products,
               customers=len(assignments), comparison=comparisons["B_vs_A"], by_persona=by_persona, groups=groups,
               distinct_recommended_items=distinct,
               cold_start_customers=sum(n == 0 for n in history.values()),
               ume_fallback_count=len(fallback), ume_requests=len(b),
               search_types=dict(Counter(r["search_type"] for r in searches)),
               excluded_recommendations=sum(r["raw_count"] - len(r["items"]) for r in responses))
    if "C" in arms:
        out["comparisons"] = comparisons
        c_rows = [r for r in responses if r["variant"] == "C"]
        explo_items = [key for r in c_rows for key in r.get("exploration_items", [])]
        views, clicks = _exploration_impressions(c_rows, decisions or [])
        c_dec = [d for d in (decisions or []) if d["customer_id"] in {r["customer_id"] for r in c_rows}]
        exploit = [d for d in c_dec if d["item_key"] not in
                   {key for r in c_rows if r["customer_id"] == d["customer_id"] for key in r.get("exploration_items", [])}]
        new_keys = new_keys or set()
        out["mab"] = dict(
            exploration_slots=len(explo_items), distinct_exploration_items=len(set(explo_items)),
            exploration_views=views, exploration_clicks=clicks, exploration_ctr=clicks / views if views else None,
            exploit_ctr_in_c=(sum("click" in d["actions"] for d in exploit) / len(exploit)) if exploit else None,
            new_item_slots=sum(key in new_keys for key in explo_items),
            new_items_shown=len(set(explo_items) & new_keys),
            new_items_shown_in_other_arms=len({i for r in responses if r["variant"] != "C" for i in r["items"]} & new_keys),
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("/data/full_catalog"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--customers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--arms", default="AB", choices=["AB", "ABC"],
                        help="AB = 인기 vs UME, ABC = + C (UME + MAB 탐색 슬롯)")
    parser.add_argument("--batches", type=int, default=1,
                        help="고객을 나눠 배치마다 행동을 재생하고 노출 · 클릭을 다음 배치 탐색에 반영 (MAB 학습)")
    parser.add_argument("--new-items", type=int, default=0, help="이 실행 안에서만 신규 상품으로 지정할 상품 수")
    parser.add_argument("--explore-config", type=json.loads, default=None,
                        help='C 그룹 탐색 설정 JSON (기본: services/explore.py DEFAULT), 예: \'{"algorithm":"epsilon_greedy"}\'')
    args = parser.parse_args()
    if args.customers < 2 or args.out.exists():
        parser.error("Use at least two customers and a new output folder")
    if args.batches < 1 or args.new_items < 0:
        parser.error("--batches >= 1, --new-items >= 0")
    print(json.dumps(run(args.catalog, args.out, args.customers, args.seed, args.api, args.arms, args.batches,
                         args.new_items, args.explore_config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
