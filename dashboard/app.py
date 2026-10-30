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
PAGES = ["개요", "모델 실험실", "A/B 테스트", "알림·재학습", "시뮬레이터", "데이터 현황"]
LABELS = {"map@12": "MAP@12", "ndcg@12": "NDCG@12", "recall@12": "Recall@12", "recall@300": "Recall@300",
          "coverage@12": "Coverage@12", "brand_alignment@12": "브랜드 일치도@12", "intra_list_diversity@12": "목록 내 다양성@12",
          "hitrate@50": "HitRate@50", "ndcg@50": "NDCG@50", "mrr@10": "MRR@10"}
PERSONAS = {"brand_loyal": "브랜드 충성형 · 스트레스 테스트", "price_sensitive": "가격 민감형",
            "explorer": "탐색형", "repeat_purchaser": "재구매형"}


def read_small(path):
    with path.open("rb") as f:
        return f.read(8 * 1024 * 1024 + 1)


def choose_offline():
    default = HERE / "data/reported_offline.json"
    files = sorted(RESULTS.glob("*.json")) if RESULTS.is_dir() else []
    labels = {"문서 보고값 · PR #23": default}
    labels.update({f"로컬 결과 · {p.name}": p for p in files})
    label = st.selectbox("결과 파일", list(labels), key="offline_file")
    uploaded = st.file_uploader("평가 JSON 가져오기", type=["json"], key="offline_upload",
                                help="recsys.ume 또는 recsys.run_eval의 결과. 서로 다른 파일의 고객 집합을 임의로 합치지 않습니다.")
    try:
        report = offline_report(uploaded.getvalue() if uploaded else read_small(labels[label]), uploaded.name if uploaded else label)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"평가 파일 확인 필요: {error}")
        st.stop()
    sources = sorted({s["source"] for s in report["sources"]}, key=lambda s: (s != "hm", s))
    source_name = st.selectbox("데이터셋", sources, format_func=lambda s: "H&M" if s == "hm" else "Amazon", key="dataset")
    candidates = [s for s in report["sources"] if s["source"] == source_name]
    if len(candidates) != 1:
        st.error("같은 source의 평가가 중복되었습니다. 한 평가 집합만 제공하세요.")
        st.stop()
    return report, candidates[0]


def sidebar():
    with st.sidebar:
        st.markdown("### MSRS<span style='color:#e34b48'>.</span>", unsafe_allow_html=True)
        st.caption("ADMIN CONSOLE / LOCAL")
        page = st.radio("관리 메뉴", PAGES, index=1, label_visibility="collapsed", key="page")
        st.divider()
        st.caption("읽기 전용 v0.1 · 서버 변경 없음")
        st.caption("온라인 검정·모델 교체·재학습은 후속 단계입니다.")
    return page


def report_banner(report, source):
    if report["reported"]:
        st.warning("문서 보고값 · 반올림 수치 / 실시간 결과가 아닙니다.")
    else:
        st.info("평가 파일에서 읽은 기록값입니다. 이번 화면에서 모델을 재평가하지 않습니다.")
    n = source.get("eval", {}).get("users")
    st.caption(f"{source['source'].upper()} · {source['split']} · 평가 고객 {int(n):,}명" if n is not None else f"{source['source']} · {source['split']} · 평가 고객 수 미제공")


def lab():
    st.title("모델 실험실")
    st.caption("같은 평가 집합에서 모델을 비교하고, 정확도와 노출 범위의 차이를 확인합니다.")
    with st.expander("결과 파일과 데이터셋 선택", expanded=True):
        report, source = choose_offline()
    report_banner(report, source)
    names = list(source["models"])
    selected = st.multiselect("비교 모델 · 최대 4개", names, default=names[:4], max_selections=4, key="models")
    if not selected:
        st.info("비교할 모델을 하나 이상 선택하세요.")
        return
    candidate = next((n for n in selected if n != "popularity"), selected[0])
    st.subheader("주요 지표")
    st.caption(f"카드 표시: {candidate} / 차이는 같은 파일의 popularity 대비 절대 차이입니다.")
    columns = st.columns(3)
    for column, metric in zip(columns, ("map@12", "recall@300", "coverage@12")):
        value = source["models"][candidate].get(metric)
        change = delta(source, candidate, metric)
        with column:
            st.metric(LABELS[metric], "미제공" if value is None else f"{value:.4f}",
                      delta=f"{change['difference']:+.4f}" if change and candidate != "popularity" else None,
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


def overview():
    st.title("운영 개요")
    st.caption("현재 연결한 정보와 아직 준비되지 않은 기능을 구분합니다.")
    st.info("오프라인 평가 파일·로컬 시뮬레이터 요약은 표시 가능 / 실시간 운영 지표는 아직 연결하지 않았습니다.")
    a, b, c = st.columns(3)
    a.metric("오프라인 비교", "파일 기반", border=True)
    b.metric("온라인 A/B 판정", "미연결", border=True)
    c.metric("자동 재학습", "미구현", border=True)
    st.subheader("API 연결 확인 · 읽기 전용")
    if st.button("서버 상태와 모델 목록 조회"):
        st.session_state.pop("api_snapshot", None)
        try:
            health, models = get_api(API, "/health"), get_api(API, "/models")
            if not isinstance(health, dict) or not isinstance(models, list) or any(not isinstance(m, dict) for m in models):
                raise ValueError("API 응답 형식을 확인하세요.")
            st.session_state.api_snapshot = {"at": datetime.now(timezone.utc).isoformat(), "health": health, "models": models}
        except ValueError as error:
            st.error(str(error))
    snap = st.session_state.get("api_snapshot")
    if snap:
        st.caption(f"조회 시각(UTC): {snap['at']} · 이후 변경은 다시 조회해야 합니다.")
        st.json(snap["health"])
        if snap["models"]:
            columns = [k for k in ("id", "name", "version", "status") if k in snap["models"][0]]
            st.dataframe(pd.DataFrame(snap["models"])[columns], hide_index=True)
        else:
            st.info("등록된 모델이 없습니다.")


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


def pending(page):
    st.title(page)
    if page == "A/B 테스트":
        st.info("실험 목록·그룹별 건수 API는 존재하지만 이 화면의 온라인 집계 연결과 통계 검정은 아직 구현하지 않았습니다.")
        st.dataframe(pd.DataFrame({"항목": ["CTR / 구매 전환율", "p-value", "95% 신뢰구간", "최소 기간·표본"],
                                  "상태": ["온라인 API 연결 예정", "검정 미실행", "온라인 구간 미계산", "팀 정책 확정 필요"]}), hide_index=True)
        st.caption("로컬 시뮬레이터의 클릭 건수를 실제 서버 A/B 검정 표본으로 간주하지 않습니다. 고객 단위 반복 행동과 실험 배정을 고려해야 합니다.")
    else:
        st.info("지표 하락 모니터링·재학습 작업·롤백은 요구사항 단계입니다. 실행 중인 것처럼 표시하지 않습니다.")
        st.button("재학습 시작 · 후속 단계", disabled=True)
        st.button("모델 롤백 · 후속 단계", disabled=True)


def inventory():
    st.title("데이터 현황")
    st.caption("로컬 개발용 파일의 메타데이터입니다. 전체 서비스 카탈로그나 DB 행 수와 다를 수 있습니다.")
    try:
        rows = parquet_inventory(DATA)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("개발용 parquet가 연결되지 않았습니다. DEV_DATA_DIR을 확인하세요.")
    except (OSError, ValueError) as error:
        st.error(f"메타데이터를 읽을 수 없습니다: {error}")
    st.caption("행 수는 파일 메타데이터에서 읽으며, 원본 고객·상품·행동 행을 화면에 펼치지 않습니다.")


def main():
    st.set_page_config(page_title="MSRS · 모델 실험실", page_icon="📊", layout="wide")
    st.markdown("""<style>
      .block-container{padding-top:4.5rem;max-width:1440px}
      [data-testid="stSidebar"]{background:#f5f6f8}
      h1{letter-spacing:-.04em} [data-testid="stMetricValue"]{font-weight:650}
      footer{visibility:hidden}
    </style>""", unsafe_allow_html=True)
    page = sidebar()
    st.caption("MSRS / 관리자 · 로컬 평가 워크스페이스")
    if page == "모델 실험실": lab()
    elif page == "개요": overview()
    elif page == "시뮬레이터": simulator()
    elif page == "데이터 현황": inventory()
    else: pending(page)


if __name__ == "__main__":
    main()
