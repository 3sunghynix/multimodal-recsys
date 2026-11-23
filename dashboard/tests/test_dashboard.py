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


# ───── 모델 버전 · 재학습 후보 비교 · 교체 (모니터링 페이지) ─────

MODELS = [{"id": 3, "name": "ume", "version": "v1", "status": "active", "params": {}},
          {"id": 4, "name": "ume", "version": "retrain-1", "status": "candidate", "params": {"warm_start_from_version_id": 3}},
          {"id": 1, "name": "popularity", "version": "v1", "status": "candidate", "params": {}}]


def snap(i, mv, metric, value, n=5000, t="2026-11-17T07:21:43"):
    return {"id": i, "model_version_id": mv, "metric": metric, "value": value, "n": n, "created_at": t}


SNAPS = [snap(8, 4, "amazon/holdout/map@12", 0.0041), snap(7, 3, "amazon/holdout/map@12", 0.0039),
         snap(6, 4, "amazon/holdout/recall@20", 0.0192), snap(5, 3, "amazon/holdout/recall@20", 0.0198),
         snap(4, 4, "hm/holdout/map@12", 0.0428), snap(3, 3, "hm/holdout/map@12", 0.0425),
         snap(2, 4, "hm/holdout/recall@20", 0.1516), snap(1, 3, "hm/holdout/recall@20", 0.1530),
         snap(0, 3, "hm/holdout/map@12", 0.0999, t="2026-11-10T00:00:00")]          # 다른 실행의 기준 기록 → 쓰지 않음


def test_candidate_comparison_pairs_same_run_and_applies_trainer_rule():
    from dashboard.data import candidate_comparison
    rows = candidate_comparison(MODELS, SNAPS)
    assert {(r["데이터셋"], r["지표"]) for r in rows} == {("amazon", "map@12"), ("amazon", "recall@20"),
                                                         ("hm", "map@12"), ("hm", "recall@20")}
    hm_map = next(r for r in rows if r["데이터셋"] == "hm" and r["지표"] == "map@12")
    assert hm_map["기준값"] == 0.0425 and hm_map["후보값"] == 0.0428                 # 가장 가까운 시각의 기준 기록
    assert all(r["채택 규칙"] == "통과" for r in rows)                                 # 11-17 실측: 둘 다 통과
    worse = [dict(s, value=0.0400) if s["id"] == 4 else s for s in SNAPS]             # H&M MAP@12 −5.9%
    assert {r["채택 규칙"] for r in candidate_comparison(MODELS, worse) if r["데이터셋"] == "hm"} == {"미통과"}


def test_candidate_comparison_ignores_versions_without_retrain_base():
    from dashboard.data import candidate_comparison
    assert candidate_comparison([MODELS[0], MODELS[2]], SNAPS) == []


# ─── A5: 서버 /experiments/{id}/stats · /metrics/window wrappers ──────

def test_get_experiment_stats_validates_id_and_hits_endpoint():
    from dashboard.data import get_experiment_stats
    for bad in (0, -1, "3", True, 3.0):
        with pytest.raises(ValueError):
            get_experiment_stats("http://unused", bad)
    stats = {"experiment_id": 3, "experiment_name": "pop vs ume",
             "variants": [{"variant": "A", "users": 10, "impressions": 100, "clicks": 5,
                           "purchases": 0, "ctr": 0.05, "ctr_ci_95": [0.02, 0.11]}],
             "comparisons": []}
    seen = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, _n): return encode(stats)

    def fake(request, timeout):
        seen.append(request.full_url); return Resp()

    with patch("dashboard.data.urlopen", side_effect=fake):
        got = get_experiment_stats("http://api:8000", 3)
    assert got == stats and seen == ["http://api:8000/experiments/3/stats"]


def test_get_metrics_window_builds_query_with_url_encoded_timestamps():
    """start · end 가 '+' 를 포함하면 '%2B' 로 인코딩돼야 한다 (PR #48 2차 리뷰: '+' 를 공백으로 디코딩 방지)."""
    from dashboard.data import get_metrics_window
    seen = []
    body = {"window_id": "w1", "views": 10, "clicks": 1, "evaluated_users": 1, "hit_users": 1,
            "cumulative_new_logs": 100}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, _n): return encode(body)

    def fake(request, timeout):
        seen.append(request.full_url); return Resp()

    with patch("dashboard.data.urlopen", side_effect=fake):
        got = get_metrics_window("http://api:8000/", "2026-11-23T00:00:00Z", "2026-11-23T01:00:00Z",
                                  model_version="live", source="all", model_version_id=3)
    assert got == body
    assert seen == ["http://api:8000/metrics/window?start=2026-11-23T00%3A00%3A00Z"
                     "&end=2026-11-23T01%3A00%3A00Z&model_version=live&source=all&model_version_id=3"]


def test_get_metrics_timeseries_yields_oldest_to_newest_buckets():
    """지난 N 시간 1시간 단위 = N 버킷. 반환은 **과거 → 최근** 순 (Altair 라인차트가 시간축 왼쪽부터 그리므로)."""
    from datetime import datetime, timezone

    from dashboard.data import get_metrics_timeseries
    bodies = iter([{"views": 10, "clicks": 1}, {"views": 20, "clicks": 2}, {"views": 30, "clicks": 6}])

    class Resp:
        def __init__(self, b): self._b = encode(b)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, _n): return self._b

    def fake(request, timeout):
        return Resp(next(bodies))

    with patch("dashboard.data.urlopen", side_effect=fake):
        series = get_metrics_timeseries("http://api:8000", hours=3, step_hours=1,
                                         now_utc=datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc))
    assert len(series) == 3
    # 과거 → 최근
    assert series[0]["start"] < series[1]["start"] < series[2]["start"]
    assert series[2]["end"] == datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)
    # CTR 계산 — 시간 역순으로 응답이 왔으므로 가장 최근 (12:00) = 응답 1 (views 10, clicks 1)
    assert series[2]["views"] == 10 and series[2]["clicks"] == 1 and abs(series[2]["ctr"] - 0.1) < 1e-9


def test_get_metrics_timeseries_skips_failed_buckets():
    """한 bucket API 가 실패해도 나머지 bucket 은 반환돼야 (UI 끊김 방지)."""
    from datetime import datetime, timezone

    from dashboard.data import get_metrics_timeseries
    calls = {"n": 0}

    def fake(request, timeout):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("timeout")
        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, _n): return encode({"views": 5, "clicks": 1})
        return Resp()

    with patch("dashboard.data.urlopen", side_effect=fake):
        series = get_metrics_timeseries("http://api:8000", hours=3, step_hours=1,
                                         now_utc=datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc))
    assert len(series) == 2                                # 3 bucket 중 2 성공


def test_activate_model_posts_only_to_activate_endpoint():
    from dashboard.data import activate_model
    for bad in (0, -1, "4", True, 4.0):
        with pytest.raises(ValueError): activate_model("http://unused", bad)
    seen = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, _n): return encode({"id": 4, "name": "ume", "version": "retrain-1", "status": "active"})

    def fake(request, timeout):
        seen.append((request.get_method(), request.full_url)); return Resp()

    with patch("dashboard.data.urlopen", side_effect=fake):
        assert activate_model("http://api:8000/", 4)["status"] == "active"
    assert seen == [("POST", "http://api:8000/models/4/activate")]
    with patch("dashboard.data.urlopen", side_effect=OSError("down")):
        with pytest.raises(ValueError, match="교체 실패"): activate_model("http://unused", 4)


def test_monitoring_page_shows_versions_comparison_and_guarded_swap(monkeypatch, tmp_path):
    calls = []

    class Resp:
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, _n): return encode(self.body) if not isinstance(self.body, list) else json.dumps(self.body).encode()

    def fake(request, timeout):
        calls.append((request.get_method(), request.full_url))
        if request.full_url.endswith("/models"): return Resp(MODELS)
        if "/metrics/snapshots" in request.full_url: return Resp(SNAPS)
        if request.full_url.endswith("/activate"): return Resp({**MODELS[1], "status": "active"})
        raise OSError("unexpected")

    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:8000")
    with patch("dashboard.data.urlopen", side_effect=fake):
        at = app(monkeypatch, tmp_path)
        at.radio(key="page").set_value("모니터링 페이지").run()
        assert not at.exception
        assert any("운영 중: 3 · ume v1" in c.value for c in at.caption)
        assert len(at.dataframe) >= 2                                              # 버전 표 + 후보 비교 표
        assert at.button(key="swap_run").disabled                                  # 확인 전에는 교체 불가
        assert not any(m == "POST" for m, _ in calls)
        at.selectbox(key="swap_target").set_value(MODELS[1]).run()
        at.checkbox(key="swap_confirm").check().run()
        at.button(key="swap_run").click().run()
        assert not at.exception
        assert ("POST", "http://127.0.0.1:8000/models/4/activate") in calls
        assert any("교체 완료" in s.value for s in at.success)


def test_monitoring_page_without_api_shows_no_versions(monkeypatch, tmp_path):
    with patch("dashboard.data.urlopen", side_effect=OSError("offline")):
        at = app(monkeypatch, tmp_path)
        at.radio(key="page").set_value("모니터링 페이지").run()
        assert not at.exception
        assert any("서버 API 미연결" in i.value for i in at.info)
        assert not at.metric
