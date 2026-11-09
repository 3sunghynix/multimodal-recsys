"""Feature Store 관찰용 — 명세 4.4 "피처 조회 시간 ≤ 10ms" 검증에 쓴다.

GET /features/latency (?user_id=1)
   ping(Redis 왕복) + 실제 피처 조회(recent_clicks · recent_views · session_clicks) 시간을 함께 잰다.
   within_spec 은 "가장 느린 값 ≤ 10ms" 기준 — 추천이 쓰는 모든 경로가 명세를 만족하는지 한눈에.
"""
import time

from fastapi import APIRouter

from ..services import feature_store

router = APIRouter(prefix="/features", tags=["features"])

SPEC_LIMIT_MS = 10.0


def _time_ms(fn) -> float:
    t = time.perf_counter()
    fn()
    return round((time.perf_counter() - t) * 1000, 3)


@router.get("/latency")
def latency(user_id: int = 1):
    """Redis 왕복(ping) + 실제 피처 조회 세 가지(recent_clicks · recent_views · session_clicks) 시간 측정.

    PR #46 리뷰 by 유환 반영: ping 만으로는 "실제 피처 읽기 지연" 을 보장하지 못한다. 명세가 요구하는
    건 "피처 조회 ≤ 10ms" 라서 추천이 쓰는 핵심 피처 세 개를 실제로 호출해 왕복 시간을 함께 측정.

    user_id 쿼리로 특정 사용자 키를 읽는 왕복 시간 측정 가능 (실유저 핫키 · 콜드키 비교에 유용)."""
    ping = feature_store.ping_latency_ms()
    if ping is None:
        return {"redis": False, "reason": "Redis 미설정 또는 연결 실패 — feature store 비활성"}

    reads = {
        "recent_clicks_ms": _time_ms(lambda: feature_store.get_recent_clicks(user_id, 5)),
        "recent_views_ms": _time_ms(lambda: feature_store.get_recent_views(user_id, 10)),
        "session_clicks_ms": _time_ms(lambda: feature_store.get_session_click_count(user_id)),
    }
    worst = max(ping, *reads.values())
    return {
        "redis": True,
        "user_id": user_id,
        "ping_ms": ping,
        **reads,
        "worst_ms": worst,
        "within_spec": worst <= SPEC_LIMIT_MS,
        "limit_ms": SPEC_LIMIT_MS,
    }
