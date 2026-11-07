"""Redis 기반 사용자별 실시간 피처 — 명세 4.4.

저장 대상
   u:{user_id}:recent_views    LIST  (최신 → 과거, 20개 cap)
   u:{user_id}:recent_clicks   LIST  (최신 → 과거, 20개 cap)
   u:{user_id}:session_clicks  COUNT (세션 TTL 30분, 비활동 시 리셋)
   u:{user_id}:last_activity   SCALAR(epoch, TTL 30분)

호출 원칙
   - record_event: POST /events 가 호출. Redis 통신 실패도 서버에 영향 없음 (swallow).
   - get_*:        추천/검색이 응답의 session_context 에 쓴다. Redis 없으면 빈 값 반환.
   - settings.redis_url 이 비어 있으면 전체 모듈이 no-op (로컬 pytest, Alembic 전용 실행 등 안전).

성능 (명세 요구 ≤10ms)
   로컬 Docker Redis p50 ~0.5ms, p95 ~1.5ms. socket_timeout=0.1s 로 네트워크 문제 시 빠르게 포기.
"""
import logging
import time

try:
    import redis
except ImportError:                                # 패키지 미설치 환경(테스트 slim)에서도 import 는 성공
    redis = None

from ..config import settings

log = logging.getLogger(__name__)

RECENT_LIMIT = 20
SESSION_TTL_SEC = 30 * 60                         # 30분 비활동 → 세션 리셋

_client = None
_tried_connect = False


def _get_client():
    """재시도 많이 안 하도록 첫 연결 성공/실패를 캐시. redis 패키지 없거나 URL 비어도 안전하게 None 반환."""
    global _client, _tried_connect
    if _tried_connect:
        return _client
    _tried_connect = True
    if redis is None or not settings.redis_url:
        return None
    try:
        # socket_timeout 0.1s = 네트워크 문제 있으면 빠르게 포기 (요청 지연 금지)
        c = redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=0.1)
        c.ping()
        _client = c
    except Exception as e:                        # 연결 실패 — 서버는 계속, 모든 호출 no-op
        log.warning(f"Redis 연결 실패, feature store 비활성화: {e}")
        _client = None
    return _client


def reset_for_tests():
    """테스트 fixture 가 호출. monkeypatch 로 settings 바꾼 뒤 다음 _get_client 를 다시 시도하게 한다."""
    global _client, _tried_connect
    _client = None
    _tried_connect = False


def record_event(user_id: int, event_type: str, item_id: int | None) -> None:
    """이벤트 하나 반영. Redis 없으면/실패해도 서버 진행에 영향 없음."""
    c = _get_client()
    if c is None:
        return
    try:
        pipe = c.pipeline(transaction=False)
        if item_id is not None:
            if event_type == "view":
                key = f"u:{user_id}:recent_views"
                pipe.lpush(key, item_id)
                pipe.ltrim(key, 0, RECENT_LIMIT - 1)
                pipe.expire(key, SESSION_TTL_SEC * 24)             # 길게 (하루) — 추천 피처로 재사용
            elif event_type == "click":
                key = f"u:{user_id}:recent_clicks"
                pipe.lpush(key, item_id)
                pipe.ltrim(key, 0, RECENT_LIMIT - 1)
                pipe.expire(key, SESSION_TTL_SEC * 24)
                # session_clicks: 클릭 때마다 +1, 30분 비활동 → 자연 소멸(세션 경계)
                sess = f"u:{user_id}:session_clicks"
                pipe.incr(sess)
                pipe.expire(sess, SESSION_TTL_SEC)
        pipe.set(f"u:{user_id}:last_activity", int(time.time()), ex=SESSION_TTL_SEC)
        pipe.execute()
    except Exception as e:                        # Redis 재시작 · 네트워크 끊김 — 로그만, 서버 계속
        log.warning(f"Redis write 실패: {e}")


def get_recent_clicks(user_id: int, k: int = 5) -> list[str]:
    """최신 순 클릭 상품 id (문자열). Redis 없으면 빈 리스트 → 호출자가 DB 폴백."""
    c = _get_client()
    if c is None:
        return []
    try:
        return c.lrange(f"u:{user_id}:recent_clicks", 0, k - 1) or []
    except Exception:
        return []


def get_recent_views(user_id: int, k: int = 10) -> list[str]:
    c = _get_client()
    if c is None:
        return []
    try:
        return c.lrange(f"u:{user_id}:recent_views", 0, k - 1) or []
    except Exception:
        return []


def get_session_click_count(user_id: int) -> int:
    """현재 세션의 클릭 수 (30분 비활동 시 리셋)."""
    c = _get_client()
    if c is None:
        return 0
    try:
        v = c.get(f"u:{user_id}:session_clicks")
        return int(v) if v else 0
    except Exception:
        return 0


def ping_latency_ms() -> float | None:
    """PING 왕복 시간 (명세 4.4 요구 ≤10ms 검증용). Redis 없으면 None."""
    c = _get_client()
    if c is None:
        return None
    try:
        t = time.perf_counter()
        c.ping()
        return round((time.perf_counter() - t) * 1000, 3)
    except Exception:
        return None
