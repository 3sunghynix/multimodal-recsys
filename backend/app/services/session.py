"""현재 세션 — 추천이 "지금 보고 있는 것"을 반영하는 데 쓴다.

세션 = 마지막 활동 후 30분이 지나지 않은 동안의 클릭 (Feature Store와 같은 경계).
- Redis 있음: u:{id}:session_clicks(세션 클릭 수, 30분 비활동 시 소멸)만큼 recent_clicks(최신 순)의 앞부분
- Redis 없음: DB에서 최근 30분 클릭 (초기 이력 is_setup 제외)
"""
from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Event, Item
from . import feature_store

SESSION_MINUTES = feature_store.SESSION_TTL_SEC // 60


def session_click_ids(db: Session, user_id: int, limit: int = 10) -> list[int]:
    """현재 세션의 클릭 상품 id, 최신 순 (최대 limit개). 세션이 없으면 빈 리스트."""
    if feature_store.enabled():
        n = feature_store.get_session_click_count(user_id)
        if n <= 0:
            return []
        return [int(i) for i in feature_store.get_recent_clicks(user_id, min(n, limit)) if str(i).isdigit()]
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=SESSION_MINUTES)   # DB 시각은 UTC (server_default now)
    return list(db.scalars(
        select(Event.item_id).where(Event.user_id == user_id, Event.type == "click", Event.item_id.is_not(None),
                                    Event.is_setup == False, Event.created_at >= since)       # noqa: E712
        .order_by(Event.created_at.desc(), Event.id.desc()).limit(limit)))


def session_interest(db: Session, item_ids: list[int]) -> str | None:
    """세션 클릭 상품들의 가장 많은 소분류 (동률이면 더 최근에 클릭한 쪽). 클릭이 없으면 None."""
    if not item_ids:
        return None
    cats = dict(db.execute(select(Item.id, Item.category).where(Item.id.in_(item_ids))).all())
    leaves = [(cats.get(i) or "").split(">")[-1].strip() for i in item_ids]
    leaves = [c for c in leaves if c]
    if not leaves:
        return None
    count = Counter(leaves)
    best = max(count.values())
    return next(c for c in leaves if count[c] == best)                # leaves는 최신 순 → 동률이면 최근 것
