import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.data import (confidence_interval, delta, get_api, load_json, offline_report,
                            parquet_inventory, simulator_summary)

ROOT = Path(__file__).parents[1]


@pytest.fixture
def native():
    return {"source": "hm", "split": "test", "eval": {"split": "test", "users": 40},
            "models": {"popularity": {"map@12": 0.0}, "ume": {"map@12": 0.1}},
            "vs_popularity": {"ume": {"map@12": {"diff": 0.1, "ci95": [0.05, 0.15]}}}}


def encode(value):
    return json.dumps(value).encode()


def test_document_values_have_provenance():
    report = offline_report((ROOT / "data/reported_offline.json").read_bytes(), "document")
    assert report["reported"]
    assert report["provenance"]["commit"].startswith("b05ff92")
    hm, amazon = report["sources"]
    assert hm["models"]["ume+popscore2.0"]["map@12"] == 0.0246
    assert amazon["models"]["ume+popscore2.0+brand2.0"]["brand_alignment@12"] == 0.107
    assert confidence_interval(hm, "ume+popscore2.0", "map@12") is None


def test_native_results_keep_same_cohort_and_ci(native):
    report = offline_report(encode(native), "hm_test_pca.json")
    source = report["sources"][0]
    assert not report["reported"]
    assert source["eval"]["users"] == 40
    assert confidence_interval(source, "ume", "map@12") == (0.05, 0.15)
    assert delta(source, "ume", "map@12") == {"difference": 0.1, "relative": None}


def test_single_baseline_does_not_invent_comparison():
    obj = {"source": "amazon", "model": "only_model", "eval": {"split": "valid", "users": 2}, "metrics": {"recall@300": 0.2}}
    source = offline_report(encode(obj), "result")["sources"][0]
    assert delta(source, "only_model", "recall@300") is None
    assert "hitrate@50" not in source["models"]["only_model"]


@pytest.mark.parametrize("bad", [b"[]", b"{broken", b'{"x":NaN}', b'{"x":Infinity}'])
def test_bad_json(bad):
    with pytest.raises(ValueError): load_json(bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5, True, float("nan")])
def test_reject_bad_metric(native, bad):
    native["models"]["ume"]["map@12"] = bad
    with pytest.raises(ValueError): offline_report(encode(native), "bad")


def test_bad_ci(native):
    native["vs_popularity"]["ume"]["map@12"]["ci95"] = [0.8, 0.1]
    source = offline_report(encode(native), "badci")["sources"][0]
    with pytest.raises(ValueError): confidence_interval(source, "ume", "map@12")


@pytest.fixture
def sim():
    return {"kind": "local_synthetic_exploratory_not_server_ab", "calibrated": False,
            "generated_customers": 1, "skipped_customers": [], "notes": [], "groups": [
                {"persona": "brand_loyal", "model": "popularity", "customers": 1, "views": 10, "clicks": 1,
                 "carts": 0, "purchases": 0, "ctr": 0.1}]}


def test_simulation_denominators_and_no_server_claim(sim):
    report = simulator_summary(encode(sim))
    assert report["kind"] == "local_synthetic_exploratory_not_server_ab"
    sim["groups"][0]["ctr"] = 0.5
    with pytest.raises(ValueError, match="CTR"): simulator_summary(encode(sim))


def test_empty_simulation_exposure_is_unavailable(sim):
    sim["groups"][0].update(views=0, clicks=0, ctr=None)
    assert simulator_summary(encode(sim))["groups"][0]["ctr"] is None


def test_api_only_reads_and_handles_connection_failure():
    with pytest.raises(ValueError): get_api("http://unused", "/models/1/activate")
    with patch("dashboard.data.urlopen", side_effect=OSError("offline")):
        with pytest.raises(ValueError, match="API"): get_api("http://unused", "/models")


def test_parquet_inventory_reads_metadata_only(tmp_path):
    import pandas as pd
    pd.DataFrame({"item_id": [1, 2]}).to_parquet(tmp_path / "items.parquet")
    assert parquet_inventory(tmp_path) == [{"파일": "items.parquet", "행 수": 2, "열 수": 1}]


def app(monkeypatch, tmp_path):
    monkeypatch.setenv("OFFLINE_RESULTS_DIR", str(tmp_path / "empty_results"))
    monkeypatch.setenv("SIM_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DEV_DATA_DIR", str(tmp_path / "data"))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()


def test_lab_default_and_dataset_change(monkeypatch, tmp_path):
    at = app(monkeypatch, tmp_path)
    assert not at.exception
    assert at.title[0].value == "모델 실험실"
    assert at.warning and "문서 보고값" in at.warning[0].value
    assert at.metric[0].value == "0.0246"
    at.selectbox(key="dataset").set_value("amazon").run()
    assert not at.exception
    assert at.metric[0].value == "0.0091"
    assert at.button[0].disabled


def test_lab_empty_model_selection(monkeypatch, tmp_path):
    at = app(monkeypatch, tmp_path)
    at.multiselect(key="models").set_value([]).run()
    assert not at.exception
    assert any("하나 이상" in i.value for i in at.info)


@pytest.mark.parametrize("page", ["개요", "A/B 테스트", "알림·재학습", "시뮬레이터", "데이터 현황"])
def test_unconnected_pages_have_no_fake_numbers(monkeypatch, tmp_path, page):
    at = app(monkeypatch, tmp_path)
    at.radio(key="page").set_value(page).run()
    assert not at.exception
    assert at.title[0].value == ("운영 개요" if page == "개요" else "시뮬레이터 결과" if page == "시뮬레이터" else page)
    if page in ("A/B 테스트", "알림·재학습", "시뮬레이터", "데이터 현황"):
        assert not at.metric


def test_simulator_page_uses_real_file(monkeypatch, tmp_path, sim):
    folder = tmp_path / "runs/pilot_001"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_bytes(encode(sim))
    at = app(monkeypatch, tmp_path)
    at.radio(key="page").set_value("시뮬레이터").run()
    assert not at.exception
    assert at.metric[0].value == "1"
    assert any("한 모델" in i.value for i in at.info)
    assert any("로컬 합성" in w.value for w in at.warning)
