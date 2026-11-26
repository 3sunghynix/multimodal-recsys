"""서버 연결 2단계 (다음작업 A1) — 가상 고객 생성 → 서버 추천 → 이벤트 일괄 전송 한 바퀴.

목표: 서비스 한 바퀴 (A1) 를 **실제 서버 API 로** 돌려 재학습·대시보드가 기댈 로그를 만든다.
현재 `simulator/catalog_pilot.py` 는 서비스 함수를 직접 부르고 rollback 해서 서버 로그가 남지 않는다.

이 모듈의 역할:
    · adapter (server_adapter.py) 를 통해 **실제 HTTP** 로 서버 API 호출 → 서버 DB 에 로그가 영구 저장
    · 결정론적 (같은 run_id 로 재실행하면 같은 고객 · 같은 이벤트 멱등 보존)
    · 행동 모델 두 가지: 기본 = 고정 확률 (상품과 무관, 연결 점검용) / --catalog = 페르소나 v0.3 (persona_behavior.py)

소규모 end-to-end 확인:
    python -m simulator.server_run --base-url http://localhost:8000 --customers 10 --sessions 2 --k 5

대량 실행 (고객 1만 · 로그 100만):
    python -m simulator.server_run --base-url http://... --customers 10000 --sessions 10 --k 10 \\
                                   --click-rate 0.10 --purchase-rate 0.02 --run-id prod-$(date +%s)

중단 복구: 지원 X. 추천 호출마다 서버가 **새 request_id** 를 발급해서, 같은 --run-id 로 이어서
돌리면 같은 event_id + 다른 request_id → 409. 중단되면 **새 --run-id** 로 처음부터 다시 돌린다.

행동 모델 1 — 고정 확률 (기본, 연결 점검용. A/B · 재학습 근거로 쓰지 않는다):
    · 각 노출 → click_rate 확률로 click
    · click 한 상품 → purchase_rate 확률로 purchase

행동 모델 2 — 페르소나 (--catalog · --item-snapshot, A1 후속):
    · 고객 = engine.make_customers (페르소나 7종 · 선호 소분류 · 브랜드 · 예산 · 초기 이력), POST /users 에 persona 저장
    · 초기 이력은 is_setup=True 구매로 한 번 보냄 → 서버 추천의 개인화 출발점 (집계 · 인기 · 재학습 카운터에서는 빠짐)
    · 노출마다 선호 점수 → 클릭 확률 (engine.click_probability), 클릭 → 장바구니 → 구매 조건부 (personas.yaml)
    · 추첨은 (seed, 고객, 세션, 순위, 행동)만으로 → 같은 상품을 보여 주면 어느 그룹이든 같은 결과
    API 컨테이너 안에서 실행 (pandas · 카탈로그 필요):
      docker compose exec api python -m scripts.export_simulator_items --out /data/db_items.json
      docker compose run --rm --no-deps -v "$PWD/simulator:/app/simulator" -v "$PWD/recsys:/app/recsys:ro" api \
        python -m simulator.server_run --base-url http://api:8000 --catalog /data/full_catalog \
        --item-snapshot /data/db_items.json --customers 100 --sessions 3 --run-id p100
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .server_adapter import RecommendResponse, ServerClient


# 서버 session_id max_length=64 (schemas.EventBase · routers/recommend.py).
# 이 값을 넘기면 /events 전송이 전부 422 로 실패 → 실행 시작 전에 확인한다 (PR #71 리뷰).
_SESSION_ID_MAX = 64


def _session_id(run_id: str, customer_idx: int, session_idx: int) -> str:
    """세션 ID — 서버의 세션 로직 (services/session.py) 이 이 값을 그대로 쓴다."""
    return f"{run_id}:c{customer_idx}:s{session_idx}"


def _check_run_id_fits(run_id: str, customers: int, sessions: int) -> None:
    """가장 긴 session_id ({run_id}:c{customers-1}:s{sessions-1}) 가 64자 이하인지 확인.
    넘으면 서버가 이벤트를 전부 422 로 거절하므로 실행 시작 전에 막는다 (PR #71 리뷰)."""
    longest = _session_id(run_id, customers - 1, sessions - 1)
    if len(longest) > _SESSION_ID_MAX:
        over = len(longest) - _SESSION_ID_MAX
        raise ValueError(
            f"--run-id 가 너무 깁니다 ({len(run_id)}자). 가장 긴 session_id={longest!r} "
            f"가 {len(longest)}자로 서버 제한 {_SESSION_ID_MAX}자를 {over}자 초과합니다. "
            f"--run-id 를 {len(run_id) - over}자 이하로 줄이세요."
        )


def _build_impression_events(client: ServerClient, rec: RecommendResponse, *,
                              run_id: str, customer_idx: int, session_idx: int,
                              user_id: int, session_id: str) -> list[dict]:
    """노출 view × k 건 — 이벤트연결규약 톡방 결정 3 의 "노출 view = request_id 있는 것만" 집계 대상."""
    events = []
    for position, item_id in rec.impressions:
        events.append({
            "user_id": user_id, "type": "view", "item_id": item_id,
            "request_id": rec.request_id, "position": position,
            "session_id": session_id,
            "event_id": client.event_id(run_id, customer_idx, session_idx, f"p{position}:view"),
        })
    return events


def _simulate_behavior(client: ServerClient, rec: RecommendResponse, rng: random.Random, *,
                       run_id: str, customer_idx: int, session_idx: int,
                       user_id: int, session_id: str,
                       click_rate: float, purchase_rate: float) -> list[dict]:
    """노출 상품마다 click/purchase 결정 → 이벤트 리스트 (view 는 포함 안 함; _build_impression_events 에서 생성)."""
    events = []
    for position, item_id in rec.impressions:
        if rng.random() >= click_rate:
            continue
        events.append({
            "user_id": user_id, "type": "click", "item_id": item_id,
            "request_id": rec.request_id, "position": position,
            "session_id": session_id,
            "event_id": client.event_id(run_id, customer_idx, session_idx, f"p{position}:click"),
        })
        # 구매 — 클릭한 상품 중 purchase_rate 확률로. 추천에서 비롯된 구매라 request_id 유지.
        if rng.random() < purchase_rate:
            events.append({
                "user_id": user_id, "type": "purchase", "item_id": item_id,
                "request_id": rec.request_id, "position": position,
                "session_id": session_id,
                "event_id": client.event_id(run_id, customer_idx, session_idx, f"p{position}:purchase"),
            })
    return events


def _send_setup(client: ServerClient, behavior, run_id: str, ci: int, user_id: int) -> list[dict]:
    """초기 이력 (is_setup=True 구매). 같은 run_id 로 다시 돌리면 같은 event_id → duplicate."""
    return [{"user_id": user_id, "type": "purchase", "item_id": item_id, "is_setup": True,
             "event_id": client.event_id(run_id, ci, -1, f"setup{n}:purchase")}
            for n, item_id in enumerate(behavior.setup_purchases(ci))]


def _persona_events(client: ServerClient, rec: RecommendResponse, behavior, *, run_id: str, customer_idx: int,
                    session_idx: int, user_id: int, session_id: str) -> list[dict]:
    """페르소나 행동 → click · add_to_cart · purchase 이벤트 (view 는 _build_impression_events)."""
    by_pos = dict(rec.impressions)
    acted = behavior.actions(customer_idx, session_idx, rec.products or [])
    events = []
    for position in sorted(acted):
        for action in acted[position]:
            events.append({
                "user_id": user_id, "type": action, "item_id": by_pos[position],
                "request_id": rec.request_id, "position": position, "session_id": session_id,
                "event_id": client.event_id(run_id, customer_idx, session_idx, f"p{position}:{action}"),
            })
    return events


def run(base_url: str, customers: int, sessions: int, k: int, run_id: str, *,
        click_rate: float = 0.10, purchase_rate: float = 0.02, seed: int = 42,
        batch_size: int = 1000, timeout: float = 15.0, behavior=None) -> dict:
    """소규모 → 대량 모두 같은 코드. 재실행 시 멱등 — created · duplicate 집계로 확인 가능.
    behavior: None 이면 고정 확률, PersonaBehavior 면 페르소나 행동 (고객 수는 behavior 가 만든 고객 수와 같아야 함).

    반환: {customers, sessions, recommend_calls, events_created, events_duplicate, events_error, elapsed_sec[, behavior]}
    """
    if customers < 1 or sessions < 1:
        raise ValueError("--customers 와 --sessions 는 1 이상")
    _check_run_id_fits(run_id, customers, sessions)
    if behavior is not None and len(behavior.customers) != customers:
        raise ValueError(f"페르소나 고객 {len(behavior.customers)}명 ≠ --customers {customers}")

    client = ServerClient(base_url, timeout=timeout)
    rng = random.Random(seed)

    created = duplicate = error = recommend_calls = 0
    pending: list[dict] = []
    t_start = time.perf_counter()

    def flush(force=False):
        nonlocal pending, created, duplicate, error
        while len(pending) >= batch_size or (force and pending):
            chunk, pending = pending[:batch_size], pending[batch_size:]
            result = client.send_events_batch(chunk)
            created += result.get("created", 0)
            duplicate += result.get("duplicate", 0)
            error += result.get("error", 0)

    for ci in range(customers):
        if behavior is None:
            user_id = client.ensure_customer(run_id, ci)
        else:
            user_id = client.ensure_customer(run_id, ci, persona=behavior.persona(ci))
            pending.extend(_send_setup(client, behavior, run_id, ci, user_id))
            flush(force=True)                                   # 첫 추천 전에 초기 이력이 서버에 있어야 개인화된다
        for si in range(sessions):
            sid = _session_id(run_id, ci, si)
            rec = client.recommend(user_id, sid, k=k)
            recommend_calls += 1

            # 노출 (view × k) + 행동 (click · cart · purchase) 를 한 세션 안에서 모아 batch 로 전송
            events = _build_impression_events(client, rec,
                                               run_id=run_id, customer_idx=ci, session_idx=si,
                                               user_id=user_id, session_id=sid)
            if behavior is None:
                events += _simulate_behavior(client, rec, rng,
                                              run_id=run_id, customer_idx=ci, session_idx=si,
                                              user_id=user_id, session_id=sid,
                                              click_rate=click_rate, purchase_rate=purchase_rate)
            else:
                events += _persona_events(client, rec, behavior, run_id=run_id, customer_idx=ci, session_idx=si,
                                          user_id=user_id, session_id=sid)
            pending.extend(events)
            if behavior is not None:
                flush(force=True)                               # 다음 세션 추천이 이 세션 클릭을 보도록 (세션 반영)
            else:
                flush()

    flush(force=True)

    out = {
        "customers": customers, "sessions": sessions, "k": k,
        "recommend_calls": recommend_calls,
        "events_created": created, "events_duplicate": duplicate, "events_error": error,
        "elapsed_sec": round(time.perf_counter() - t_start, 1),
    }
    if behavior is not None:
        out["behavior"] = {"kind": "persona_v0.3", **behavior.summary()}
    else:
        out["behavior"] = {"kind": "fixed_rate", "click_rate": click_rate, "purchase_rate": purchase_rate}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000", help="서버 API base URL")
    ap.add_argument("--customers", type=int, default=100, help="가상 고객 수 (소규모 100, 대량 10000)")
    ap.add_argument("--sessions", type=int, default=3, help="고객당 세션 수")
    ap.add_argument("--k", type=int, default=10, help="세션당 추천 개수")
    ap.add_argument("--click-rate", type=float, default=0.10, help="노출당 클릭 확률 [0, 1]")
    ap.add_argument("--purchase-rate", type=float, default=0.02, help="클릭당 구매 확률 [0, 1]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", default="dev", help="실행 식별자 (고객 · 이벤트 멱등 키의 네임스페이스)")
    ap.add_argument("--out", type=Path, default=Path("simulator/runs/server_run"), help="결과 요약 저장 경로")
    ap.add_argument("--catalog", type=Path, default=None,
                    help="페르소나 행동: 적재에 쓴 카탈로그 폴더 (items.parquet · events.parquet). 없으면 고정 확률")
    ap.add_argument("--item-snapshot", type=Path, default=None,
                    help="페르소나 행동: 서버 상품 id 표 (scripts.export_simulator_items 결과)")
    args = ap.parse_args()
    behavior = None
    if args.catalog or args.item_snapshot:
        if not (args.catalog and args.item_snapshot):
            raise SystemExit("--catalog 와 --item-snapshot 은 함께 줘야 합니다")
        from .persona_behavior import PersonaBehavior               # pandas 필요 — 고정 확률 모드는 표준 라이브러리만
        behavior = PersonaBehavior(args.catalog, args.item_snapshot, args.customers, seed=args.seed)

    summary = run(
        args.base_url, args.customers, args.sessions, args.k,
        run_id=args.run_id,
        click_rate=args.click_rate, purchase_rate=args.purchase_rate,
        seed=args.seed, behavior=behavior,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
