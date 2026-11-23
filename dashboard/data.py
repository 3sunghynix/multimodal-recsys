"""Adapters. Display recorded metrics; do not recalculate model quality.
The only write is activate_model (operator-confirmed model swap)."""
import json
import math
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


LIMIT = 8 * 1024 * 1024
KNOWN_METRICS = ("map@12", "ndcg@12", "recall@12", "recall@300", "coverage@12",
                 "brand_alignment@12", "intra_list_diversity@12", "hitrate@50", "ndcg@50", "mrr@10")


def load_json(raw: bytes) -> dict:
    if len(raw) > LIMIT:
        raise ValueError("JSON 파일은 8MB 이하여야 합니다.")
    try:
        value = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("NaN/Infinity 금지")))
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"JSON을 읽을 수 없습니다: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("JSON 최상위는 객체여야 합니다.")
    return value


def number(value, label, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{label}: 유효한 비음수 숫자가 필요합니다.")
    return float(value)


def offline_report(raw: bytes, filename: str) -> dict:
    """Native recsys.ume output, run_eval output, or a labelled document snapshot."""
    obj = load_json(raw)
    reported = obj.get("kind") == "document_reported_snapshot"
    if reported:
        if not obj.get("provenance"):
            raise ValueError("문서 보고값에는 출처가 필요합니다.")
        sources = obj["sources"]
    else:
        if obj.get("source") not in {"amazon", "hm"}:
            raise ValueError("source는 amazon 또는 hm이어야 합니다.")
        if "models" not in obj and "metrics" not in obj:
            raise ValueError("recsys.ume 또는 recsys.run_eval의 결과 JSON이 필요합니다.")
        models = obj.get("models")
        if models is None:
            models = {obj.get("model", "unknown"): obj["metrics"]}
        sources = [{"source": obj["source"], "eval": obj.get("eval", {}), "models": models,
                    "vs_popularity": obj.get("vs_popularity", {}), "params": obj.get("params", {}),
                    "split": obj.get("split", obj.get("eval", {}).get("split")), "input": obj.get("input")}]
    if not isinstance(sources, list) or any(not isinstance(s, dict) for s in sources):
        raise ValueError("평가 source 목록 형식이 잘못됐습니다.")
    validated = []
    for source in sources:
        if source.get("source") not in {"amazon", "hm"}:
            raise ValueError("지원하지 않는 source입니다.")
        models = source.get("models")
        if not isinstance(models, dict) or not models:
            raise ValueError("모델 지표가 없습니다.")
        evaluation = source.get("eval", {})
        if not isinstance(evaluation, dict):
            raise ValueError("평가 집합 형식이 잘못됐습니다.")
        users = evaluation.get("users")
        if users is not None:
            if not number(users, "평가 고객 수").is_integer():
                raise ValueError("평가 고객 수는 정수여야 합니다.")
        split = source.get("split", evaluation.get("split"))
        if split not in {"test", "valid"}:
            raise ValueError("평가 split(test/valid)이 명시되어야 합니다.")
        clean = {}
        for model, metrics in models.items():
            if not isinstance(model, str) or not model or not isinstance(metrics, dict):
                raise ValueError("잘못된 모델 지표 형식입니다.")
            clean[model] = {}
            for name, value in metrics.items():
                if name in KNOWN_METRICS and value is not None:
                    clean[model][name] = number(value, name, 2 if name.startswith("intra_list_diversity") else 1)
            if not clean[model]:
                raise ValueError(f"표시 가능한 지표 없음: {model}")
        validated.append({**source, "models": clean, "split": split})
    if not validated:
        raise ValueError("비어 있는 평가 결과입니다.")
    return {"filename": filename, "reported": reported, "provenance": obj.get("provenance"),
            "sources": validated, "notes": obj.get("notes", [])}


def delta(source, model, metric):
    """Only compare models sharing this one report's eval cohort; never merge files."""
    base = source["models"].get("popularity", {}).get(metric)
    value = source["models"].get(model, {}).get(metric)
    if base is None or value is None:
        return None
    return {"difference": value - base, "relative": (value - base) / base if base else None}


def confidence_interval(source, model, metric):
    entry = source.get("vs_popularity", {}).get(model, {}).get(metric)
    if entry is None:
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("ci95"), (list, tuple)) or len(entry["ci95"]) != 2:
        raise ValueError("신뢰구간 형식이 잘못됐습니다.")
    lo, hi = entry["ci95"]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (lo, hi)) or lo > hi:
        raise ValueError("신뢰구간 값이 잘못됐습니다.")
    return lo, hi


def simulator_summary(raw):
    obj = load_json(raw)
    if obj.get("kind") != "local_synthetic_exploratory_not_server_ab" or obj.get("calibrated") is not False:
        raise ValueError("지원하는 로컬 시뮬레이터 summary.json이 아닙니다.")
    if not number(obj.get("generated_customers"), "생성 고객 수").is_integer():
        raise ValueError("생성 고객 수는 정수여야 합니다.")
    if not isinstance(obj.get("skipped_customers"), list) or not isinstance(obj.get("groups"), list):
        raise ValueError("시뮬레이터 요약 형식이 잘못됐습니다.")
    for group in obj["groups"]:
        if not isinstance(group, dict):
            raise ValueError("모델 그룹은 객체여야 합니다.")
        for k in ("customers", "views", "clicks", "carts", "purchases"):
            value = number(group.get(k), k)
            if not value.is_integer():
                raise ValueError(f"{k}: 정수 건수가 필요합니다.")
        if not group["purchases"] <= group["carts"] <= group["clicks"] <= group["views"]:
            raise ValueError("행동 건수의 순서가 일치하지 않습니다.")
        expected = group["clicks"] / group["views"] if group["views"] else None
        if group.get("ctr") is not None:
            number(group["ctr"], "CTR", 1)
        if (expected is None) != (group.get("ctr") is None) or (expected is not None and not math.isclose(expected, group["ctr"], abs_tol=1e-10)):
            raise ValueError("CTR과 click/view 건수가 일치하지 않습니다.")
        if not isinstance(group.get("persona"), str) or not isinstance(group.get("model"), str):
            raise ValueError("페르소나·모델 이름이 필요합니다.")
    return obj


SNAPSHOTS = "/metrics/snapshots?limit=1000"
# 재학습 trainer(recsys/retrain.py accept)와 같은 채택 규칙: 주 지표 MAP@12, 보조 Recall@k
MAP_TOLERANCE, RECALL_TOLERANCE = 0.02, 0.10


def get_api(base_url, endpoint, timeout=3):
    """GET only; URL supplied by deployment environment, not an action form."""
    if endpoint not in {"/health", "/models", "/experiments", SNAPSHOTS}:
        raise ValueError("지원하지 않는 읽기 전용 endpoint")
    request = Request(base_url.rstrip("/") + endpoint, method="GET", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError("API 응답이 너무 큽니다.")
        return json.loads(raw)
    except (URLError, OSError, ValueError) as error:
        raise ValueError("API에 연결할 수 없습니다. 서버 실행 상태와 API_BASE_URL을 확인하세요.") from error


def _get_json(url: str, timeout: int = 3) -> dict | list:
    """GET only — 공통 HTTP → JSON. 응답 크기 제한 · ValueError 로 통합."""
    request = Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError("API 응답이 너무 큽니다.")
        return json.loads(raw)
    except (URLError, OSError, ValueError) as error:
        raise ValueError("API에 연결할 수 없습니다. 서버 실행 상태와 API_BASE_URL을 확인하세요.") from error


def get_experiment_stats(base_url: str, exp_id: int, timeout: int = 3) -> dict:
    """A/B 통계 검정 (서버 /experiments/{id}/stats) — 그룹별 집계 + 그룹 간 Wilson CI · Z-test (A5).

    is_setup=True 이벤트는 집계 제외 (PR #30 규칙). 반환: {experiment_id, experiment_name, variants[], comparisons[]}."""
    if isinstance(exp_id, bool) or not isinstance(exp_id, int) or exp_id <= 0:
        raise ValueError("experiment id는 양의 정수여야 합니다.")
    return _get_json(f"{base_url.rstrip('/')}/experiments/{exp_id}/stats", timeout=timeout)


def get_metrics_window(base_url: str, start_iso: str, end_iso: str, *,
                       model_version: str = "live", source: str = "all",
                       model_version_id: int | None = None, timeout: int = 5) -> dict:
    """지정 구간 CTR · views · clicks · evaluated_users · hit_users 집계 (서버 /metrics/window).

    start_iso · end_iso: ISO 8601 with timezone (서버가 Z 또는 +00:00 요구). model_version_id 가 오면
    해당 모델의 노출만 집계 — A/B 중 모델별 CTR 추이 그래프에 쓴다."""
    from urllib.parse import quote
    params = f"start={quote(start_iso, safe='')}&end={quote(end_iso, safe='')}" \
             f"&model_version={quote(model_version, safe='')}&source={quote(source, safe='')}"
    if model_version_id is not None:
        if isinstance(model_version_id, bool) or not isinstance(model_version_id, int) or model_version_id <= 0:
            raise ValueError("model_version_id는 양의 정수여야 합니다.")
        params += f"&model_version_id={model_version_id}"
    return _get_json(f"{base_url.rstrip('/')}/metrics/window?{params}", timeout=timeout)


def get_metrics_timeseries(base_url: str, *, hours: int = 24, step_hours: int = 1,
                           model_version_id: int | None = None, now_utc=None, timeout: int = 5) -> list[dict]:
    """지난 `hours` 시간을 `step_hours` 단위로 쪼개 /metrics/window 를 여러 번 호출 → CTR 추이용 시계열.
    각 bucket 의 {start, end, views, clicks, ctr} 를 돌려준다. 실패한 bucket 은 건너뜀 (None 대신 skip)."""
    from datetime import datetime, timedelta, timezone
    if hours < 1 or step_hours < 1 or step_hours > hours:
        raise ValueError("hours · step_hours 는 1 이상, step_hours ≤ hours 이어야 합니다.")
    end = (now_utc or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    buckets: list[dict] = []
    for i in range(hours // step_hours):
        b_end = end - timedelta(hours=i * step_hours)
        b_start = b_end - timedelta(hours=step_hours)
        try:
            w = get_metrics_window(base_url, b_start.isoformat().replace("+00:00", "Z"),
                                    b_end.isoformat().replace("+00:00", "Z"),
                                    model_version_id=model_version_id, timeout=timeout)
        except ValueError:
            continue                                            # 한 bucket 실패는 UI 전체를 막지 않는다
        views, clicks = int(w.get("views", 0)), int(w.get("clicks", 0))
        buckets.append({"start": b_start, "end": b_end, "views": views, "clicks": clicks,
                         "ctr": (clicks / views) if views else None})
    return list(reversed(buckets))                             # 과거 → 최근 순으로


def activate_model(base_url, model_id, timeout=10):
    """대시보드에서 유일한 쓰기 동작: 운영 모델 교체 (POST /models/{id}/activate).
    화면에서 확인 체크 후에만 부른다. 기존 active는 서버가 candidate로 내린다."""
    if isinstance(model_id, bool) or not isinstance(model_id, int) or model_id <= 0:
        raise ValueError("모델 id는 양의 정수여야 합니다.")
    request = Request(f"{base_url.rstrip('/')}/models/{model_id}/activate", method="POST",
                      headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read(LIMIT + 1))
    except (URLError, OSError, ValueError) as error:
        raise ValueError(f"모델 교체 실패: {error}") from error


def candidate_comparison(models, snapshots):
    """재학습 후보(params.warm_start_from_version_id가 있는 버전)마다, 같은 trainer 실행에서 기록된
    기준 · 후보의 holdout 지표(`<데이터셋>/holdout/<지표>`)를 짝지어 비교한다.

    같은 실행의 기준 스냅샷은 후보 스냅샷과 지표 이름 · 평가 고객 수(n)가 같고 생성 시각이 가장 가까운 것이다
    (trainer가 한 트랜잭션에서 두 줄을 함께 쓴다). 반환: 행 목록 (데이터셋 판정은 trainer와 같은 규칙)."""
    by_id = {m["id"]: m for m in models}
    snaps = [s for s in snapshots if isinstance(s.get("metric"), str) and "/holdout/" in s["metric"]]
    rows = []
    for cand in models:
        base_id = (cand.get("params") or {}).get("warm_start_from_version_id")
        if base_id is None:
            continue
        mine = {}
        for s in snaps:
            if s.get("model_version_id") == cand["id"]:
                mine.setdefault(s["metric"], s)                    # 최신 먼저 (id 내림차순 응답)
        per_source = {}
        for metric, s in mine.items():
            peers = [b for b in snaps if b.get("model_version_id") == base_id and b["metric"] == metric and b.get("n") == s.get("n")]
            if not peers:
                continue
            base = min(peers, key=lambda b: abs(_ts(b["created_at"]) - _ts(s["created_at"])))
            source, _, name = metric.split("/", 2)
            per_source.setdefault(source, {})[name] = (base["value"], s["value"], s.get("n"))
        for source, values in sorted(per_source.items()):
            verdict = _verdict(values)
            for name, (b, c, n) in sorted(values.items()):
                rows.append({"후보": f"{cand['id']} · {cand.get('version')}", "후보 id": cand["id"],
                             "기준": f"{base_id} · {by_id.get(base_id, {}).get('version', '?')}", "데이터셋": source,
                             "지표": name, "기준값": b, "후보값": c, "차이": c - b,
                             "상대 변화": (c - b) / b if b else None, "평가 고객": n, "채택 규칙": verdict})
    return rows


def _ts(value):
    from datetime import datetime
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def _verdict(values):
    """values: {지표: (기준, 후보, n)}. MAP@12 ≥ 기준 × 0.98 그리고 Recall ≥ 기준 × 0.90."""
    m = values.get("map@12")
    if m is None:
        return "판정 불가 (MAP@12 없음)"
    ok = m[1] >= m[0] * (1 - MAP_TOLERANCE)
    for name, (b, c, _) in values.items():
        if name.startswith("recall@"):
            ok = ok and c >= b * (1 - RECALL_TOLERANCE)
    return "통과" if ok else "미통과"


def parquet_inventory(directory):
    import pyarrow.parquet as pq
    rows = []
    for name in ("items", "users", "events", "brands"):
        path = Path(directory) / f"{name}.parquet"
        if path.is_file():
            meta = pq.read_metadata(path)
            rows.append({"파일": path.name, "행 수": meta.num_rows, "열 수": meta.num_columns})
    return rows
