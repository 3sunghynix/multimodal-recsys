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
    assert at.title[0].value == "A/B 테스트"
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


@pytest.mark.parametrize("page", ["시뮬레이터", "모니터링 페이지"])
def test_unconnected_pages_have_no_fake_numbers(monkeypatch, tmp_path, page):
    at = app(monkeypatch, tmp_path)
    assert at.radio(key="page").options == ["A/B 테스트", "시뮬레이터", "모니터링 페이지"]
    at.radio(key="page").set_value(page).run()
    assert not at.exception
    assert at.title[0].value == ("시뮬레이터 결과" if page == "시뮬레이터" else page)
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


def test_original_test_result_is_default_and_ci_displayed(monkeypatch, tmp_path, native):
    folder = tmp_path / "empty_results"
    folder.mkdir()
    (folder / "hm_test_pca.json").write_bytes(encode(native))
    at = app(monkeypatch, tmp_path)
    assert not at.exception
    assert not at.warning
    assert at.metric[0].value == "0.1000"
    assert any("+0.05000, +0.15000" in text.value for text in at.markdown)
    # Missing Amazon file falls back for Amazon only, not to the H&M result.
    at.selectbox(key="dataset").set_value("amazon").run()
    assert not at.exception and at.warning
    assert at.metric[3].label == "브랜드 일치도@12"
    assert at.metric[3].value == "10.7%"


def test_invalid_original_is_not_silently_replaced(monkeypatch, tmp_path, native):
    folder = tmp_path / "empty_results"
    folder.mkdir()
    native["source"] = "amazon"
    (folder / "hm_test_pca.json").write_bytes(encode(native))
    at = app(monkeypatch, tmp_path)
    assert not at.exception
    assert at.error and "조건" in at.error[0].value
    assert not at.metric


def test_original_result_appearing_after_first_render_is_preferred(monkeypatch, tmp_path, native):
    at = app(monkeypatch, tmp_path)
    assert at.warning
    folder = tmp_path / "empty_results"
    folder.mkdir()
    (folder / "hm_test_pca.json").write_bytes(encode(native))
    at.run()
    assert not at.exception and not at.warning
    assert at.metric[0].value == "0.1000"


def test_new_scale_ab_and_monitoring_pages(monkeypatch, tmp_path):
    folder=tmp_path/'runs/scale_001';folder.mkdir(parents=True)
    result={'metric':'first_session_customer_click_rate','A':{'rate':.1},'B':{'rate':.2},'p_value':.01,'ci95':[.01,.2]}
    report={'kind':'synthetic_scale_fixture','ab_by_persona':{'window_shopper':result},
            'products':50000,'customers':10000,'category_levels':3,'events':1000000,'event_counts':{'view':900000},
            'groups':[{'persona':'window_shopper','variant':'A','views':100,'clicks':10},
                      {'persona':'window_shopper','variant':'B','views':100,'clicks':20}], 'requirements_met':{}}
    (folder/'scale_summary.json').write_bytes(encode(report))
    at=app(monkeypatch,tmp_path)
    assert not at.exception
    assert any('눈팅형' in select.options for select in at.selectbox)
    at.radio(key='page').set_value('시뮬레이터').run()
    assert not at.exception
    assert not any('규모 검증 결과' in select.label for select in at.selectbox)
    at.radio(key='page').set_value('A/B 테스트').run()
    assert any('합성 · 임시 결과 (실제 카탈로그 A/B로 교체 예정)' in c.value for c in at.caption)
    state={'latest':{'values':{'ctr':.01,'hitrate':None}},'last_window_end':'2026-10-31T01:00:00Z',
           'identity':['ume-v1','amazon','fixed-users','v1'],'alerts':[],'jobs':[{'version':'candidate-000001','status':'requested'}]}
    (folder/'monitor_state.json').write_bytes(encode(state))
    at.radio(key='page').set_value('모니터링 페이지').run()
    assert not at.exception
    assert at.metric[0].value=='1.00%'
    assert at.metric[1].value=='계산 불가'


def test_real_catalog_pilot_replaces_synthetic_ab(monkeypatch, tmp_path):
    folder = tmp_path / 'runs/catalog_pilot_1k'; folder.mkdir(parents=True)
    test = lambda a, b: {'A': {'customers': 50, 'clicked_customers': a, 'rate': a / 50},
                         'B': {'customers': 50, 'clicked_customers': b, 'rate': b / 50},
                         'difference_B_minus_A': (b - a) / 50, 'ci95': [-.1, .2], 'p_value': .5}
    pilot = {'kind': 'real_catalog_ab_pilot', 'comparison': test(14, 16),
             'groups': {'A': {'view': 500, 'click': 22}, 'B': {'view': 500, 'click': 22}},
             'ume_fallback_count': 2, 'ume_requests': 50, 'cold_start_customers': 4,
             'distinct_recommended_items': {'A': 10, 'B': 256}, 'elapsed_seconds': 44.5, 'seconds_per_customer': .445,
             'by_persona': {'window_shopper': {'comparison': test(20, 10), 'cold_start_customers': 4, 'ume_fallback_count': 2},
                            'practical': {'comparison': None, 'cold_start_customers': 0, 'ume_fallback_count': 0}}}
    (folder / 'catalog_ab_summary.json').write_bytes(encode(pilot))
    at = app(monkeypatch, tmp_path)
    assert not at.exception
    assert any('실제 상품과 추천 모델을 사용한 모의 고객 실험' in c.value for c in at.caption)
    assert not any('합성 · 임시 결과' in c.value for c in at.caption)
    assert any('A 10 · B 256' in m.value for m in at.markdown)
    persona_tables = [df.value for df in at.dataframe if '페르소나' in df.value.columns]
    assert len(persona_tables) == 1 and list(persona_tables[0]['페르소나']) == ['눈팅형']   # no-comparison persona skipped


def test_pilot_with_mab_arm_shows_c_section(monkeypatch, tmp_path):
    folder = tmp_path / 'runs/catalog_pilot_abc'; folder.mkdir(parents=True)
    test = lambda a, b: {'A': {'customers': 50, 'clicked_customers': a, 'rate': a / 50},
                         'B': {'customers': 50, 'clicked_customers': b, 'rate': b / 50},
                         'difference_B_minus_A': (b - a) / 50, 'ci95': [-.1, .2], 'p_value': .5}
    pilot = {'kind': 'real_catalog_ab_pilot', 'comparison': test(14, 16),
             'comparisons': {'B_vs_A': test(14, 16), 'C_vs_B': test(16, 18), 'C_vs_A': test(14, 18)},
             'groups': {a: {'view': 500, 'click': 22} for a in 'ABC'},
             'ume_fallback_count': 2, 'ume_requests': 50, 'cold_start_customers': 4,
             'distinct_recommended_items': {'A': 10, 'B': 256, 'C': 300}, 'new_items': 100,
             'mab': {'exploration_slots': 100, 'distinct_exploration_items': 80, 'exploration_views': 100,
                     'exploration_clicks': 3, 'exploration_ctr': .03, 'exploit_ctr_in_c': .05,
                     'new_item_slots': 40, 'new_items_shown': 35, 'new_items_shown_in_other_arms': 0},
             'batches': [{'batch': b, 'exploration_ctr': .02 + b / 100, 'c_ctr': .05} for b in range(3)],
             'by_persona': {}}
    (folder / 'catalog_ab_summary.json').write_bytes(encode(pilot))
    at = app(monkeypatch, tmp_path)
    assert not at.exception
    assert any('C · UME + MAB 탐색 슬롯' in m.value for m in at.markdown)
    tables = [df.value for df in at.dataframe if '비교' in df.value.columns]
    assert len(tables) == 1 and len(tables[0]) == 3
    assert any('신규 상품: 탐색 슬롯 40회' in m.value for m in at.markdown)
