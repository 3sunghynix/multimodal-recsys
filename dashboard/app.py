"""Streamlit model laboratory: read-only evaluation and local simulator reports."""
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import altair as alt
import pandas as pd
import streamlit as st

from dashboard.data import (KNOWN_METRICS, confidence_interval, delta, get_api, offline_report,
                            parquet_inventory, simulator_summary)


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
RESULTS = Path(os.getenv("OFFLINE_RESULTS_DIR", str(ROOT / "results/ume")))
RUNS = Path(os.getenv("SIM_RUNS_DIR", str(ROOT / "simulator/runs")))
DATA = Path(os.getenv("DEV_DATA_DIR", str(ROOT / "data/dev_subset")))
PAGES = ["A/B 테스트", "시뮬레이터", "모니터링 페이지"]
LABELS = {"map@12": "MAP@12", "ndcg@12": "NDCG@12", "recall@12": "Recall@12", "recall@300": "Recall@300",
          "coverage@12": "Coverage@12", "brand_alignment@12": "브랜드 일치도@12", "intra_list_diversity@12": "목록 내 다양성@12",
          "hitrate@50": "HitRate@50", "ndcg@50": "NDCG@50", "mrr@10": "MRR@10"}
PERSONAS = {"trendsetter":"트렌드세터", "practical":"실용주의자", "value_seeker":"가성비추구",
            "brand_loyal":"브랜드충성", "impulse_buyer":"충동구매", "careful_explorer":"신중탐색", "window_shopper":"눈팅형",
            "price_sensitive":"이전 버전 · 가격 민감형", "explorer":"이전 버전 · 탐색형", "repeat_purchaser":"이전 버전 · 재구매형"}


def read_small(path):
    with path.open("rb") as f:
        return f.read(8 * 1024 * 1024 + 1)


def choose_offline():
    default = HERE / "data/reported_offline.json"
    files = sorted(RESULTS.glob("*.json")) if RESULTS.is_dir() else []
    labels = {"원본 결과 자동 선택 · 없으면 문서 보고값": None}
    labels.update({f"로컬 결과 · {p.name}": p for p in files})
    labels["문서 보고값 · PR #23"] = default
    label = st.selectbox("결과 파일", list(labels), key="offline_file")
    uploaded = st.file_uploader("평가 JSON 가져오기", type=["json"], key="offline_upload",
                                help="recsys.ume 또는 recsys.run_eval의 결과. 서로 다른 파일의 고객 집합을 임의로 합치지 않습니다.")
    try:
        if uploaded:
            report = offline_report(uploaded.getvalue(), uploaded.name)
        elif labels[label] is None:
            source_name = st.selectbox("데이터셋", ["hm", "amazon"],
                                      format_func=lambda s: "H&M" if s == "hm" else "Amazon", key="dataset")
            path = RESULTS / f"{source_name}_test_pca.json"
            if path.is_file():
                report = offline_report(read_small(path), path.name)
                if report["reported"] or len(report["sources"]) != 1 or report["sources"][0]["source"] != source_name or report["sources"][0]["split"] != "test":
                    raise ValueError("자동 선택 파일의 source/test 조건이 일치하지 않습니다.")
            else:
                st.caption(f"{path.name} 미설치 → 문서 보고값 사용. 원본 JSON을 results/ume/에 넣으면 다음 화면 갱신부터 우선 표시합니다.")
                report = offline_report(read_small(default), "문서 보고값 · PR #23")
            return report, next(s for s in report["sources"] if s["source"] == source_name)
        else:
            report = offline_report(read_small(labels[label]), label)
        sources = sorted({s["source"] for s in report["sources"]}, key=lambda s: (s != "hm", s))
        source_name = st.selectbox("데이터셋", sources, format_func=lambda s: "H&M" if s == "hm" else "Amazon", key="dataset")
        candidates = [s for s in report["sources"] if s["source"] == source_name]
        if len(candidates) != 1:
            raise ValueError("같은 source의 평가가 중복되었습니다. 한 평가 집합만 제공하세요.")
        return report, candidates[0]
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"평가 파일 확인 필요: {error}")
        st.stop()


def sidebar():
    with st.sidebar:
        st.markdown("### MSRS<span style='color:#e34b48'>.</span>", unsafe_allow_html=True)
        st.caption("ADMIN CONSOLE / LOCAL")
        page = st.radio("관리 메뉴", PAGES, index=0, label_visibility="collapsed", key="page")
        st.divider()
        st.caption("읽기 전용 v0.1 · 서버 변경 없음")
        st.caption("합성 실험·저장 결과를 구분합니다. 서버 변경은 수행하지 않습니다.")
    return page


def report_banner(report, source):
    if report["reported"]:
        st.warning("문서 보고값 · 반올림 수치 / 실시간 결과가 아닙니다.")
    else:
        st.info("평가 파일에서 읽은 기록값입니다. 이번 화면에서 모델을 재평가하지 않습니다.")
    n = source.get("eval", {}).get("users")
    st.caption(f"{source['source'].upper()} · {source['split']} · 평가 고객 {int(n):,}명" if n is not None else f"{source['source']} · {source['split']} · 평가 고객 수 미제공")


def lab():
    st.subheader("검색·추천 오프라인 평가")
    st.caption("같은 평가 집합에서 모델을 비교하고, 정확도와 노출 범위의 차이를 확인합니다.")
    with st.expander("결과 파일과 데이터셋 선택", expanded=True):
        report, source = choose_offline()
    report_banner(report, source)
    names = list(source["models"])
    service_name = "ume+popscore2.0" if source["source"] == "hm" else "ume+popscore2.0+brand2.0"
    if service_name in names:
        names = [n for n in ("popularity", service_name) if n in names] + [n for n in names if n not in {"popularity", service_name}]
    selected = st.multiselect("비교 모델 · 최대 4개", names, default=names[:4], max_selections=4, key="models")
    if not selected:
        st.info("비교할 모델을 하나 이상 선택하세요.")
        return
    candidate = next((n for n in selected if n != "popularity"), selected[0])
    st.subheader("주요 지표")
    st.caption(f"카드 표시: {candidate} / 차이는 같은 파일의 popularity 대비 절대 차이입니다.")
    card_metrics = ["map@12", "recall@300", "coverage@12"]
    if source["source"] == "amazon":
        card_metrics.append("brand_alignment@12")
    columns = st.columns(len(card_metrics))
    for column, metric in zip(columns, card_metrics):
        value = source["models"][candidate].get(metric)
        change = delta(source, candidate, metric)
        with column:
            display = "미제공" if value is None else (f"{value:.1%}" if metric == "brand_alignment@12" else f"{value:.4f}")
            difference = None
            if change and candidate != "popularity":
                difference = f"{change['difference'] * 100:+.2f}%p" if metric == "brand_alignment@12" else f"{change['difference']:+.4f}"
            st.metric(LABELS[metric], display,
                      delta=difference,
                      delta_color="off", border=True)
    st.caption("Coverage 증가는 노출 범위 확대이며 항상 성능 개선을 뜻하지 않습니다. 브랜드 일치도는 고객별 최다 브랜드 비중과 다릅니다.")
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("모델별 지표 비교")
        available = [m for m in KNOWN_METRICS if any(m in source["models"][n] for n in selected)]
        metric = st.selectbox("차트 지표", available, format_func=lambda m: LABELS[m], key="chart_metric")
        rows = [{"모델": n, "값": source["models"][n].get(metric)} for n in selected if source["models"][n].get(metric) is not None]
        chart = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("모델:N", sort=selected, axis=alt.Axis(labelAngle=-15)), y=alt.Y("값:Q", title=LABELS[metric]),
            color=alt.Color("모델:N", scale=alt.Scale(scheme="tableau10"), legend=None), tooltip=["모델", alt.Tooltip("값:Q", format=".5f")]).properties(height=270)
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.subheader("비교 해석")
        change = delta(source, candidate, metric)
        if change:
            st.write(f"{candidate}의 {LABELS[metric]} 차이: **{change['difference']:+.5f}**")
            st.write("상대 변화: 계산 불가(기준값 0)" if change["relative"] is None else f"상대 변화: **{change['relative']:+.1%}**")
        else:
            st.info("같은 평가 파일에 popularity 기준값이 없어 개선율을 계산하지 않습니다.")
        try:
            ci = confidence_interval(source, candidate, metric)
            st.write(f"파일 제공 차이의 95% 구간: [{ci[0]:+.5f}, {ci[1]:+.5f}]" if ci else "95% 구간: 파일에 미제공")
        except ValueError as error:
            st.error(str(error))
        st.caption("서로 다른 데이터셋·분할·고객 집합·K의 지표를 같은 실험으로 합치지 않습니다. 문서 반올림값의 변화율은 근사값입니다.")
        st.button("이 모델로 교체 · 후속 단계", disabled=True, help="평가 파일의 모델명과 서버 model_version_id의 매핑·확인 절차가 필요합니다.")
    st.subheader("상세 결과")
    table = pd.DataFrame([{ "모델": n, **{LABELS[m]: source["models"][n].get(m) for m in available}} for n in selected])
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("표 CSV 다운로드", table.to_csv(index=False).encode("utf-8-sig"), "model_comparison.csv", "text/csv")
    with st.expander("출처·조건·미연결 지표"):
        if report["provenance"]:
            st.json(report["provenance"])
        for note in report["notes"]:
            st.write(f"• {note}")
        st.write("학교 과제용 HitRate@50·NDCG@50·검색 MRR@10 등은 해당 결과 파일이 준비된 뒤 표시합니다. Recall을 HitRate로 대체하지 않습니다.")


def simulator():
    st.title("시뮬레이터 결과")
    st.warning("로컬 합성 행동 결과입니다. 서버 A/B 실험·실제 고객 CTR이 아닙니다.")
    files = sorted(RUNS.glob("*/summary.json")) if RUNS.is_dir() else []
    chosen = st.selectbox("로컬 실행 결과", files, format_func=lambda p: p.parent.name) if files else None
    upload = st.file_uploader("summary.json 가져오기", type=["json"], key="sim_upload")
    if not upload and chosen is None:
        st.info("아직 결과가 없습니다. simulator/runs/<실행명>/summary.json을 연결하거나 업로드하세요.")
        return
    try:
        report = simulator_summary(upload.getvalue() if upload else read_small(chosen))
    except (ValueError, OSError, KeyError, TypeError) as error:
        st.error(f"시뮬레이터 파일 확인 필요: {error}")
        return
    x, y, z = st.columns(3)
    x.metric("생성 고객", f"{report['generated_customers']:,}", border=True)
    y.metric("생성 제외", f"{len(report['skipped_customers']):,}", border=True)
    z.metric("서버 이벤트 전송", "없음 · 로컬 재생", border=True)
    personas = sorted({g["persona"] for g in report["groups"]})
    if not personas:
        st.info("표시할 모델 그룹이 없습니다.")
        return
    persona = st.selectbox("페르소나 · 같은 유형 안에서 모델 비교", personas, format_func=lambda p: PERSONAS.get(p, p))
    if persona == "brand_loyal":
        st.info("브랜드 충성형은 강한 브랜드 선호의 스트레스 테스트이며 실제 고객 분포를 재현하지 않습니다.")
    rows = [g for g in report["groups"] if g["persona"] == persona]
    table = pd.DataFrame([{ "모델": r["model"], "모델 버전 ID": r.get("model_version_id"), "실험 ID": r.get("experiment_id"), "그룹": r.get("variant"),
                            "고객": r["customers"], "노출": r["views"], "클릭": r["clicks"], "장바구니": r["carts"], "구매": r["purchases"], "CTR": r["ctr"] } for r in rows])
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("CTR = 클릭 / 노출. 노출이 없으면 계산 불가입니다. 페르소나 사이 CTR 크기를 모델 우열로 해석하지 않습니다.")
    if len({r["model"] for r in rows}) == 1:
        st.info("현재 한 모델의 결과만 있습니다. UME 대비 개선율이나 A/B 유의성을 표시하지 않습니다.")
    with st.expander("제외 사유·시뮬레이터 한계"):
        counts = pd.Series([s.get("reason", "unknown") for s in report["skipped_customers"]]).value_counts()
        st.dataframe(counts.rename_axis("제외 사유").reset_index(name="고객 수"), hide_index=True)
        for note in report.get("notes", []):
            st.write(note)
    st.button("시뮬레이터 실행 · 후속 단계", disabled=True, help="현재 화면은 결과를 읽으며 작업 실행 API를 호출하지 않습니다.")


def ab_page():
    st.title("A/B 테스트")
    offline_tab, simulated_tab, search_tab = st.tabs(["추천 평가", "A/B 시뮬레이션", "검색 품질"])
    with offline_tab:
        lab()
    with simulated_tab:
        st.caption("첫 세션에서 한 번 이상 클릭한 고객 비율의 Z-test입니다. 전체 세션 CTR은 별도로 표시하며 검정에 사용하지 않습니다.")
        from dashboard.data import load_json
        pilot_files = sorted(RUNS.glob("*/catalog_ab_summary.json")) if RUNS.is_dir() else []
        if pilot_files:
            path = st.selectbox("실제 카탈로그 파일럿", pilot_files, format_func=lambda p: p.parent.name)
            try:
                pilot = load_json(read_small(path))
                if pilot["kind"] != "real_catalog_ab_pilot":
                    raise ValueError("실제 카탈로그 파일럿 요약이 필요합니다.")
                result = pilot["comparison"]
                st.caption("실제 상품과 추천 모델을 사용한 모의 고객 실험입니다.")
                st.bar_chart(pd.DataFrame({
                    "그룹": ["A · Popularity", "B · UME"],
                    "첫 세션 고객 클릭률": [result[arm]["rate"] for arm in "AB"],
                }).set_index("그룹"))
                lo, hi = result["ci95"]
                p = result["p_value"]
                st.write(f"B − A의 95% 신뢰구간: [{lo:+.2%}, {hi:+.2%}]")
                st.write(f"p-value: {p:.4g}" if p is not None else "p-value: 표본 또는 기대 빈도 부족")
                st.dataframe(pd.DataFrame([
                    {"그룹": arm, "고객": result[arm]["customers"],
                     "전체 CTR": pilot["groups"][arm].get("click", 0) / pilot["groups"][arm]["view"]}
                    for arm in "AB"
                ]), hide_index=True)
                st.write(f"UME 인기 모델 대체: {pilot['ume_fallback_count']} / {pilot['ume_requests']}"
                         f" · 이력 0개 신규 고객 (두 그룹 합계): {pilot.get('cold_start_customers', '—')}명")
                if "distinct_recommended_items" in pilot:
                    d = pilot["distinct_recommended_items"]
                    st.write(f"추천된 서로 다른 상품 수: A {d['A']:,} · B {d['B']:,}")
                if "elapsed_seconds" in pilot:
                    st.caption(f"실행 시간 {pilot['elapsed_seconds']:,.0f}초 (고객당 {pilot['seconds_per_customer']}초)")
                rows = []
                for persona, item in pilot.get("by_persona", {}).items():
                    c = item["comparison"]
                    if c is None:
                        continue
                    rows.append({"페르소나": PERSONAS.get(persona, persona),
                                 "고객 A/B": f"{c['A']['customers']}/{c['B']['customers']}",
                                 "A 클릭률": c["A"]["rate"], "B 클릭률": c["B"]["rate"],
                                 "B − A": c["difference_B_minus_A"],
                                 "95% 구간": f"[{c['ci95'][0]:+.2%}, {c['ci95'][1]:+.2%}]",
                                 "p-value": c["p_value"],
                                 "신규 고객": item["cold_start_customers"],
                                 "UME 대체": item["ume_fallback_count"]})
                if rows:
                    st.markdown("**페르소나별**")
                    st.dataframe(pd.DataFrame(rows), hide_index=True)
                    st.caption("페르소나별 검정은 다중 비교 미보정입니다. 표본 부족이면 p-value가 비어 있습니다. 자동 승자는 선택하지 않습니다.")
            except (ValueError, KeyError, TypeError, OSError) as error:
                st.error(str(error))
        else:
            st.caption("합성 · 임시 결과 (실제 카탈로그 A/B로 교체 예정). 현재 가상 상품으로 만든 인기 vs 카테고리 인기 비교이며, 최종 목표는 실제 카탈로그와 추천 API를 사용한 UME vs 인기 비교입니다.")
            files = sorted(RUNS.glob("*/scale_summary.json")) if RUNS.is_dir() else []
            chosen = st.selectbox("규모·A/B 실행 결과", files, format_func=lambda p:p.parent.name, key="ab_file") if files else None
            upload = st.file_uploader("scale_summary.json", type=["json"], key="ab_upload")
            if upload or chosen:
                try:
                    from dashboard.data import load_json
                    obj=load_json(upload.getvalue() if upload else read_small(chosen))
                    if obj.get("kind")!="synthetic_scale_fixture": raise ValueError("규모 실험 결과 파일이 필요합니다.")
                    persona=st.selectbox("검정 페르소나", list(obj['ab_by_persona']),format_func=lambda p:PERSONAS.get(p,p))
                    result=obj['ab_by_persona'][persona]
                    if result.get('metric')!='first_session_customer_click_rate':
                        raise ValueError('첫 세션 클릭률로 생성한 A/B 결과가 필요합니다. 이전 scale_summary.json을 다시 생성하세요.')
                    st.json(result)
                    st.bar_chart(pd.DataFrame({"그룹":["A","B"],"첫 세션 고객 클릭률":[result['A']['rate'],result['B']['rate']]}).set_index('그룹'))
                    groups={r['variant']:r for r in obj.get('groups',[]) if r.get('persona')==persona}
                    if all(v in groups for v in ('A','B')):
                        st.dataframe(pd.DataFrame([{'그룹':v,'전체 세션 클릭':groups[v]['clicks'],
                                                    '전체 세션 노출':groups[v]['views'],
                                                    '전체 세션 CTR':groups[v]['clicks']/groups[v]['views'] if groups[v]['views'] else None}
                                                   for v in ('A','B')]),hide_index=True)
                    st.caption("CTR = 전체 세션 클릭 / 전체 세션 노출. 페르소나별 검정은 다중 비교 미보정입니다. 자동 승자는 선택하지 않습니다.")
                except (ValueError, KeyError, TypeError, OSError) as error:st.error(str(error))
            else: st.info("아직 A/B 시뮬레이션 결과가 없습니다. python -m simulator.scale 실행 후 연결하세요.")
    with search_tab:
        upload=st.file_uploader("검색 품질 평가 JSON",type=['json'],key='search_quality')
        st.caption("형식: source, split, evaluation_id, metrics 객체. 값은 0~1. 평가 코드는 별도이며 이 화면은 저장값을 표시합니다.")
        if upload:
            try:
                from dashboard.data import load_json, number
                obj=load_json(upload.getvalue())
                if not all(obj.get(k) for k in ('source','split','evaluation_id')):raise ValueError('평가 조건·출처 필요')
                metrics=obj['metrics']
                allowed={'hitrate@50','ndcg@50','mrr@10'}
                if not isinstance(metrics,dict) or not metrics or not set(metrics)<=allowed:raise ValueError('지원 지표: HitRate@50, NDCG@50, MRR@10')
                st.caption(f"{obj['source']} / {obj['split']} / {obj['evaluation_id']}")
                values={k:number(v,k,1) for k,v in metrics.items()}
                st.bar_chart(pd.DataFrame({'지표':list(values),'값':list(values.values())}).set_index('지표'))
            except (ValueError,KeyError,TypeError) as error:st.error(str(error))
        else:st.info("검색 평가 파일 미연결 · 수치를 임의로 채우지 않습니다.")


def monitoring_page():
    st.title("모니터링 페이지")
    st.caption("닫힌 집계 구간의 CTR·HitRate, 임계값 알림, 재학습 요청과 후보 버전을 확인합니다.")
    st.info("현재 파일 기반 모니터입니다. 서버 집계 생산자·실제 학습 작업 실행·모델 활성화 연동은 후속 작업입니다.")
    files=sorted(RUNS.glob('*/monitor_state.json')) if RUNS.is_dir() else []
    chosen=st.selectbox('모니터링 기록',files,format_func=lambda p:p.parent.name) if files else None
    upload=st.file_uploader('monitor_state.json',type=['json'],key='monitor_upload')
    if upload or chosen:
        try:
            from dashboard.data import load_json, number
            obj=load_json(upload.getvalue() if upload else read_small(chosen))
            latest=obj['latest']; values=latest['values']
            a,b=st.columns(2)
            for col,key in [(a,'ctr'),(b,'hitrate')]:
                value=values[key]
                col.metric(key.upper(), '계산 불가' if value is None else f"{number(value,key,1):.2%}")
            st.caption(f"집계 종료: {obj['last_window_end']} · 모델/source/고객집합/정의: {obj['identity']}")
            st.subheader('임계값 알림');st.dataframe(pd.DataFrame(obj.get('alerts',[])))
            st.subheader('재학습 요청·후보 버전');st.dataframe(pd.DataFrame(obj.get('jobs',[])))
            st.caption('requested는 재학습 요청 기록입니다. 학습 완료나 운영 교체를 뜻하지 않습니다.')
        except (ValueError,KeyError,TypeError,OSError) as error:st.error(str(error))
    else:st.info('아직 모니터링 집계 파일이 없습니다.')


def main():
    st.set_page_config(page_title="MSRS · 관리자", page_icon="📊", layout="wide")
    st.markdown("""<style>
      .block-container{padding-top:4.5rem;max-width:1440px}
      [data-testid="stSidebar"]{background:#f5f6f8}
      h1{letter-spacing:-.04em} [data-testid="stMetricValue"]{font-weight:650}
      footer{visibility:hidden}
    </style>""", unsafe_allow_html=True)
    page = sidebar()
    st.caption("MSRS / 관리자 · 로컬 평가 워크스페이스")
    if page == "A/B 테스트": ab_page()
    elif page == "시뮬레이터": simulator()
    else: monitoring_page()


if __name__ == "__main__":
    main()
