import copy
import json
from pathlib import Path
from urllib.request import Request

import pandas as pd
import pytest

from simulator.server import ReadOnlyClient, fetch_recommendations, inspect_server, map_items, readiness


def catalog():
    return pd.DataFrame([{"source": "amazon", "item_id": 7, "source_item_id": "B007"},
                         {"source": "hm", "item_id": 7, "source_item_id": "000007"}])


def snapshot():
    return {"kind": "simulator_db_item_snapshot_v1", "items": [
        {"id": 99, "source": "amazon", "source_item_id": "B007"},
        {"id": 100, "source": "hm", "source_item_id": "000007"}]}


def spec():
    return {"paths": {"/recommend/{user_id}": {"get": {}}, "/events": {"post": {}}},
            "components": {"schemas": {"EventIn": {"properties": {"type": {}}},
                                         "RecommendOut": {"properties": {"items": {}}}}}}


class FakeClient:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or {"user_id": 5, "model": "ume", "model_version_id": 3,
                                     "experiment_id": 2, "variant": "B", "items": [100, 99]}

    def get(self, path):
        self.calls.append(path)
        if path == "/health":
            return {"status": "ok"}
        if path == "/openapi.json":
            return spec()
        return copy.deepcopy(self.response)


def test_source_qualified_mapping_preserves_leading_zero_external_ids():
    assert map_items(catalog(), snapshot()) == ({"amazon:7": 99, "hm:7": 100}, [])


@pytest.mark.parametrize("where", ["database_identity", "database_id", "local_identity", "local_external"])
def test_ambiguous_mapping_fails(where):
    items, snap = catalog(), snapshot()
    if where == "database_identity":
        snap["items"].append({"id": 101, "source": "hm", "source_item_id": "000007"})
    elif where == "database_id":
        snap["items"][1]["id"] = 99
    elif where == "local_identity":
        items = pd.concat([items, items.iloc[[0]]], ignore_index=True)
    else:
        items = pd.concat([items, pd.DataFrame([{"source": "hm", "item_id": 8, "source_item_id": "000007"}])])
    with pytest.raises(ValueError, match="Duplicate"):
        map_items(items, snap)


def test_unmapped_products_reported_without_guessing_db_ids():
    snap = snapshot()
    snap["items"] = snap["items"][:1]
    assert map_items(catalog(), snap) == ({"amazon:7": 99}, ["hm:7"])


def test_readiness_blocks_delivery_even_if_all_advertised():
    s = spec()
    s["paths"]["/users"] = {"post": {}}
    s["components"]["schemas"]["EventIn"]["properties"].update(event_id={}, is_setup={})
    s["components"]["schemas"]["RecommendOut"]["properties"]["request_id"] = {}
    r = readiness(s)
    assert not r["missing"] and r["delivery_enabled"] is False


def test_capture_keeps_server_order_and_attribution_without_history_claims():
    c = FakeClient()
    result = fetch_recommendations(c, [{"customer_id": "sim-a", "server_user_id": 5}], {"amazon:7": 99, "hm:7": 100}, 2)
    row = result[0]
    assert row["mapped_item_keys"] == ["hm:7", "amazon:7"]
    assert row["server_response"] == c.response and not row["initial_history_verified"]
    assert c.calls == ["/recommend/5?k=2"]
    assert "actions" not in row and "ctr" not in row


@pytest.mark.parametrize("patch", [{"items": [999]}, {"items": [99, 99]}, {"user_id": 6},
                                   {"variant": None}, {"model_version_id": None}])
def test_invalid_server_responses_fail_closed(patch):
    c = FakeClient()
    c.response.update(patch)
    with pytest.raises(ValueError):
        fetch_recommendations(c, [{"customer_id": "sim-a", "server_user_id": 5}], {"amazon:7": 99}, 2)


def test_invalid_binding_batch_rejected_before_network():
    c = FakeClient()
    with pytest.raises(ValueError):
        fetch_recommendations(c, [{"customer_id": "a", "server_user_id": 5},
                                 {"customer_id": "b", "server_user_id": 5}], {}, 2)
    assert c.calls == []


def test_preflight_saves_hashes_and_refuses_overwrite(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    catalog().to_parquet(data / "items.parquet")
    snap = tmp_path / "db.json"
    snap.write_text(json.dumps(snapshot()), encoding="utf-8")
    c = FakeClient()
    out = tmp_path / "out"
    r = inspect_server(data, snap, out, "http://example.test", client=c)
    assert r["mapped_products"] == 2 and not r["server_writes"]
    assert "create_user_post" in r["missing"]
    assert set(c.calls) == {"/health", "/openapi.json"}
    assert len(json.loads((out / "checksums.json").read_text(encoding="utf-8"))) == 3
    with pytest.raises(FileExistsError):
        inspect_server(data, snap, out, "http://example.test", client=c)
    assert len(c.calls) == 2


def test_transport_uses_get_only(monkeypatch):
    class Reply:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return b'{"status":"ok"}'
    seen = []
    def open_fake(request, timeout):
        assert isinstance(request, Request)
        seen.append(request.get_method())
        return Reply()
    monkeypatch.setattr("simulator.server.urlopen", open_fake)
    assert ReadOnlyClient("http://127.0.0.1:8000").get("/health")["status"] == "ok"
    assert seen == ["GET"]
