"""Read-only adapters. Display recorded metrics; do not recalculate model quality."""
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


def get_api(base_url, endpoint, timeout=3):
    """GET only; URL supplied by deployment environment, not an action form."""
    if endpoint not in {"/health", "/models", "/experiments"}:
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


def parquet_inventory(directory):
    import pyarrow.parquet as pq
    rows = []
    for name in ("items", "users", "events", "brands"):
        path = Path(directory) / f"{name}.parquet"
        if path.is_file():
            meta = pq.read_metadata(path)
            rows.append({"파일": path.name, "행 수": meta.num_rows, "열 수": meta.num_columns})
    return rows
