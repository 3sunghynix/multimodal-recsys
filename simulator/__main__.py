"""python -m simulator: local-only dry-run or replay, never POST /events."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from . import __version__
from .config import load_config
from .data import file_hash, load_data
from .engine import local_popularity_responses, make_customers, simulate, summarize


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def run(data_dir, out, config_path=None, count=None, history_size=10, recommendations=None):
    config_path = Path(config_path or Path(__file__).with_name("personas.yaml"))
    out, data_dir = Path(out), Path(data_dir)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}; choose a new run folder (no overwrite)")
    config = load_config(config_path)
    data = load_data(data_dir)
    customers, skipped = make_customers(data, config, count=count, history_size=history_size)
    if not customers:
        raise ValueError(f"All customers blocked by missing setup features: {skipped[:3]}")
    responses = (read_jsonl(recommendations) if recommendations else
                 local_popularity_responses(customers, data, config["pilot"]["recommendations_per_session"]))
    if not responses:
        raise ValueError("No recommendation responses; provide at least one (items may be empty)")
    hashes = {name: file_hash(data_dir / name) for name in ("items.parquet", "events.parquet")}
    code_hashes = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    identity = {"config_sha256": file_hash(config_path), "data_sha256": hashes, "code_sha256": code_hashes,
                "count_per_persona": count or config["pilot"]["customers_per_persona"],
                "history_size": history_size, "recommendations_sha256": file_hash(recommendations) if recommendations else "local_popularity",
                "simulator_version": __version__}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    decisions, events = simulate(customers, responses, data, config, run_id)
    summary = summarize(customers, skipped, responses, events, config)
    responding = {r["customer_id"] for r in responses}
    summary["customers_without_response"] = sorted(c["customer_id"] for c in customers if c["customer_id"] not in responding)
    try:
        if not (Path(__file__).parent.parent / ".git").exists():
            raise FileNotFoundError("Source snapshot has no own Git repository")
        code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent,
                                              stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        code_commit = None
    metadata = {**identity, "run_id": run_id, "code_commit": code_commit, "data": data.stats,
                "mode": "recorded_list_replay" if recommendations else "local_popularity_dry_run",
                "history_size_proposal": history_size, "config_snapshot": config,
                "behavior_rng": "sha256(seed,customer,session,rank,action); model/variant excluded",
                "local_ids_are_not_server_idempotency": True, "server_writes": False,
                "input_key_namespace": "source:processed_item_id, NOT database items.id",
                "local_baseline_is_not_production_popularity": not bool(recommendations)}
    # Validate everything before creating output. Atomic directory creation refuses concurrent reuse.
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", metadata)
    write_json(out / "customers.json", customers)
    write_jsonl(out / "recommendations.jsonl", responses)
    write_jsonl(out / "decisions.jsonl", decisions)
    write_jsonl(out / "events.jsonl", events)
    write_json(out / "summary.json", summary)
    text = ["# 로컬 합성 시뮬레이터 실행 결과", "", "서버 A/B 결과·실제 고객 효과가 아닌 dry-run/replay입니다.", "",
            "| 페르소나 | 모델 | 고객 | 노출 | 클릭 | 구매 | CTR |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["groups"]:
        ctr = "계산 불가" if row["ctr"] is None else f"{row['ctr']:.4%}"
        text.append(f"| {row['persona']} | {row['model']} | {row['customers']} | {row['views']} | {row['clicks']} | {row['purchases']} | {ctr} |")
    text += ["", f"생성 고객 {len(customers)}명 / 생성 제외 {len(skipped)}명 / 응답 없는 고객 {len(summary['customers_without_response'])}명",
             "", "브랜드 충성형은 스트레스 테스트입니다. 페르소나 간 CTR 순위를 모델 성능 순위로 해석하지 않습니다.",
             "초기 이력은 별도 customers.json에 보존하며 이벤트 수에 포함하지 않습니다."]
    (out / "SUMMARY.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    outputs = {p.name: file_hash(p) for p in sorted(out.iterdir())}
    write_json(out / "checksums.json", outputs)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="items.parquet and events.parquet with explicit split/feedback")
    ap.add_argument("--out", type=Path, required=True, help="new local output directory; no overwrite")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("personas.yaml"))
    ap.add_argument("--customers-per-persona", type=int, default=None)
    ap.add_argument("--history-size", type=int, default=10, help="proposed synthetic initial-history size, 3..50")
    ap.add_argument("--recommendations", type=Path, help="optional JSONL of recorded source-qualified lists")
    args = ap.parse_args()
    try:
        result = run(args.data, args.out, args.config, args.customers_per_persona, args.history_size, args.recommendations)
    except (ValueError, KeyError, OSError) as error:
        ap.exit(2, f"Simulator stopped: {error}\n")
    print(f"Generated customers: {result['generated_customers']}; skipped: {len(result['skipped_customers'])}")
    for g in result["groups"]:
        print(f"{g['persona']} | {g['model']} | views={g['views']} clicks={g['clicks']} purchases={g['purchases']} CTR={g['ctr']}")
    print(f"Saved locally: {args.out}\nNo server events were sent.")


if __name__ == "__main__":
    main()
