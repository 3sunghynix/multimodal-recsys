"""Feature Store 관찰용 — 명세 4.4 "피처 조회 시간 ≤ 10ms" 검증에 쓴다.

GET /features/latency
   {"redis": true, "ping_ms": 0.48, "within_spec": true, "limit_ms": 10}
   또는 Redis 없으면 {"redis": false}
"""
from fastapi import APIRouter

from ..services import feature_store

router = APIRouter(prefix="/features", tags=["features"])

SPEC_LIMIT_MS = 10.0


@router.get("/latency")
def latency():
    """Redis 왕복 시간 측정. 명세 4.4 "피처 조회 시간 ≤ 10ms" 체크용.
    within_spec=False 이면 운영 환경에서 Redis 가 느려졌거나 네트워크 문제 — 조사 필요."""
    ping = feature_store.ping_latency_ms()
    if ping is None:
        return {"redis": False, "reason": "Redis 미설정 또는 연결 실패 — feature store 비활성"}
    return {
        "redis": True,
        "ping_ms": ping,
        "within_spec": ping <= SPEC_LIMIT_MS,
        "limit_ms": SPEC_LIMIT_MS,
    }
