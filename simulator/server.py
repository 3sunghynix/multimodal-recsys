"""Read-only server bridge: python -m simulator.server. Never sends events or creates users."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import pandas as pd

from .__main__ import write_json, write_jsonl
from .data import file_hash, item_key


def positive_id(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Server IDs must be positive integers")
    return value


class ReadOnlyClient:
    def __init__(self, base_url, timeout=15):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("Use an http(s) API base URL without credentials, query or fragment")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("API path must start with one slash")
        request = Request(self.base_url + path, headers={"Accept": "application/json"}, method="GET")
        with urlopen(request, timeout=self.timeout) as response:
            payload = response.read(8 * 1024 * 1024 + 1)
        if len(payload) > 8 * 1024 * 1024:
            raise ValueError("API JSON response exceeds 8 MiB")
        return json.loads(payload)


def properties(spec, model):
    return spec.get("components", {}).get("schemas", {}).get(model, {}).get("properties", {})


def readiness(spec):
    """Report advertised capabilities only; OpenAPI presence does not prove delivery semantics."""
    paths = spec.get("paths", {})
    event = properties(spec, "EventIn")
    rec = properties(spec, "RecommendOut")
    checks = {
        "recommend_get": "get" in paths.get("/recommend/{user_id}", {}),
        "create_user_post": "post" in paths.get("/users", {}),
        "event_post": "post" in paths.get("/events", {}),
        "event_dedup_key_advertised": bool({"event_id", "idempotency_key"} & set(event)),
        "recommendation_link_id_advertised": bool({"request_id", "impression_id"} & set(rec)),
        "setup_event_marker_advertised": bool({"is_setup", "event_origin", "phase"} & set(event)),
    }
    return {"checks": checks, "missing": [k for k, v in checks.items() if not v],
            "delivery_enabled": False,
            "reason": "This bridge is GET-only regardless of advertised capabilities. Creation, initial-history loading and event delivery require a separately reviewed backend contract.",
            "openapi_is_not_runtime_idempotency_verification": True}


def map_items(items, snapshot):
    required = {"source", "item_id", "source_item_id"}
    if required - set(items.columns):
        raise ValueError(f"items.parquet missing mapping columns: {sorted(required - set(items.columns))}")
    if snapshot.get("kind") != "simulator_db_item_snapshot_v1" or not isinstance(snapshot.get("items"), list):
        raise ValueError("Expected item snapshot exported from the target database")
    db, server_ids = {}, set()
    for row in snapshot["items"]:
        sid = positive_id(row["id"])
        key = (row["source"], str(row["source_item_id"]))
        if sid in server_ids or key in db:
            raise ValueError("Duplicate database ID or (source, source_item_id); mapping is ambiguous")
        if row["source_item_id"] is None or not key[1]:
            raise ValueError("Missing source_item_id in database snapshot")
        server_ids.add(sid)
        db[key] = sid
    mapping, external = {}, set()
    for row in items.to_dict("records"):
        if pd.isna(row["source_item_id"]) or pd.isna(row["item_id"]):
            raise ValueError("Missing local item_id/source_item_id")
        raw = row["item_id"]
        if isinstance(raw, bool) or float(raw) < 0 or not float(raw).is_integer():
            raise ValueError("Local item_id must be a nonnegative integer")
        key = item_key(row["source"], raw)
        source_key = (row["source"], str(row["source_item_id"]))
        if key in mapping or source_key in external:
            raise ValueError("Duplicate local item identity; mapping must be one-to-one")
        external.add(source_key)
        mapping[key] = db.get(source_key)
    matched = {k: v for k, v in mapping.items() if v is not None}
    return matched, sorted(k for k, v in mapping.items() if v is None)


def fetch_recommendations(client, bindings, mapping, k):
    """Capture once per explicitly bound server user; fail on unknown products, never drop them."""
    if type(k) is not int or not 1 <= k <= 100:
        raise ValueError("k must be 1..100")
    reverse = {v: key for key, v in mapping.items()}
    if len(reverse) != len(mapping):
        raise ValueError("Mapping contains repeated database IDs")
    seen_customers, seen_users, captured = set(), set(), []
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("Bindings must be a nonempty list")
    # Validate all bindings before any request.
    for row in bindings:
        cid, uid = row.get("customer_id"), positive_id(row.get("server_user_id"))
        if not isinstance(cid, str) or not cid or cid in seen_customers or uid in seen_users:
            raise ValueError("Bindings require unique customer IDs and unique server users")
        seen_customers.add(cid)
        seen_users.add(uid)
    for row in bindings:
        cid, uid = row["customer_id"], row["server_user_id"]
        response = client.get(f"/recommend/{uid}?{urlencode({'k': k})}")
        if response.get("user_id") != uid:
            raise ValueError("Recommendation user_id differs from requested server user")
        ids = response.get("items")
        if not isinstance(ids, list) or len(ids) > k or any(type(i) is not int or i <= 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid or duplicated recommendation item IDs")
        unknown = [i for i in ids if i not in reverse]
        if unknown:
            raise ValueError(f"Recommendations contain unmapped products {unknown[:10]}; do not truncate or replace IDs")
        if not isinstance(response.get("model"), str) or not response["model"]:
            raise ValueError("Missing recommendation model")
        positive_id(response.get("model_version_id"))
        exp, variant = response.get("experiment_id"), response.get("variant")
        if exp is not None:
            positive_id(exp)
            if not isinstance(variant, str) or not variant:
                raise ValueError("Experiment requires nonempty variant")
        elif variant is not None:
            raise ValueError("Variant without experiment_id")
        captured.append({"customer_id": cid, "server_user_id": uid, "server_response": response,
                         "mapped_item_keys": [reverse[i] for i in ids],
                         "initial_history_verified": False,
                         "purpose": "read_only_diagnostic_not_behavior_replay_or_ab"})
    return captured


def inspect_server(data_dir, snapshot_path, out, base_url, bindings_path=None, k=10, client=None):
    data_dir, snapshot_path, out = Path(data_dir), Path(snapshot_path), Path(out)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}; choose a new folder")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    mapping, missing = map_items(pd.read_parquet(data_dir / "items.parquet"), snapshot)
    client = client or ReadOnlyClient(base_url)
    health = client.get("/health")
    if health.get("status") != "ok":
        raise ValueError("Backend health is not ok")
    spec = client.get("/openapi.json")
    report = readiness(spec)
    captures = []
    if bindings_path:
        if not report["checks"]["recommend_get"]:
            raise ValueError("GET /recommend/{user_id} unavailable")
        bindings = json.loads(Path(bindings_path).read_text(encoding="utf-8"))
        captures = fetch_recommendations(client, bindings, mapping, k)
    report.update({"kind": "simulator_server_read_only_preflight_v1", "base_url": base_url,
                   "checked_at_utc": datetime.now(timezone.utc).isoformat(), "health": health,
                   "mapped_products": len(mapping), "unmapped_products": missing,
                   "recommendations_captured": len(captures), "server_writes": False,
                   "history_verified": False, "snapshot_captured_at_utc": snapshot.get("captured_at_utc"),
                   "inputs_sha256": {"items.parquet": file_hash(data_dir / "items.parquet"),
                                     "db_items.json": file_hash(snapshot_path),
                                     "bindings": file_hash(bindings_path) if bindings_path else None}})
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "readiness.json", report)
    write_json(out / "item_mapping.json", mapping)
    write_jsonl(out / "server_recommendations.jsonl", captures)
    write_json(out / "checksums.json", {p.name: file_hash(p) for p in sorted(out.iterdir())})
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--db-items", type=Path, required=True, help="Read-only database item snapshot JSON")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bindings", type=Path, help="Optional explicit existing-user bindings; captures GET recommendations only")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    try:
        result = inspect_server(args.data, args.db_items, args.out, args.base_url, args.bindings, args.k)
    except (ValueError, KeyError, OSError, TypeError) as error:
        ap.exit(2, f"Read-only bridge stopped: {error}\n")
    print(f"Mapped products: {result['mapped_products']}; unmapped: {len(result['unmapped_products'])}")
    print(f"Missing advertised capabilities: {', '.join(result['missing']) or 'none (semantics still need verification)'}")
    print(f"Saved to: {args.out}\nGET-only diagnostic. No users created, history loaded, or server events sent.")


if __name__ == "__main__":
    main()
