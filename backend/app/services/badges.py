"""상품 키워드 배지 (D4): 상품 사진 · 설명으로는 알 수 없고 사 본 사람 · 구매 데이터만 아는 정보를 카드의 짧은 키워드로.

- 리뷰 배지 (Amazon): item_badges 테이블 (recsys/review_badges.py가 리뷰 429만 건에서 계산 → scripts/load_badges.py).
  예: 작게 나와요 · 착용감 좋음 · 비침 주의 · 체형 보정 · 출근룩. 근거 = 리뷰 N건 중 M건, 같은 카테고리 평균의 몇 배.
- 구매 로그 배지 (두 데이터셋): 요청 때 계산 · 캐시. 로그가 쌓이면 바뀐다 (시뮬레이터 고객 제외 — 인기 통계와 같은 범위).
  최근 인기 = 최근 창(그 데이터셋 서비스 구매의 마지막 시각 기준 H&M 7일 · Amazon 28일) 구매 3건 이상이고 같은 카테고리 상위 10%.
  급상승   = 최근 창 구매 5건 이상이고 직전 같은 길이 창의 2배 이상.
카드에는 최대 MAX_CARD개: 구매 로그 배지 → 리뷰 배지(주의 먼저, 카테고리 대비 배수 순 — 적재 때 정한 rank).
"""
from datetime import timedelta

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Event, Item, ItemBadge
from .catalog_index import catalog_signature
from .event_scope import service_users

MAX_CARD = 3
WINDOW_DAYS = {"hm": 7, "amazon": 28}
HOT_TOP, HOT_MIN, RISING_MIN, RISING_X = 0.10, 3, 5, 2.0
_log_cache: dict = {}


def _evidence_review(b: ItemBadge) -> str:
    lift = f", 같은 카테고리의 {b.lift:.1f}배" if b.lift else ""
    return f"리뷰 {b.n_reviews:,}건 중 {b.n_mention:,}건{lift}"


def review_badges(db: Session, ids: list[int]) -> dict[int, list[dict]]:
    if not ids:
        return {}
    rows = db.scalars(select(ItemBadge).where(ItemBadge.item_id.in_(ids)).order_by(ItemBadge.item_id, ItemBadge.rank)).all()
    out: dict[int, list[dict]] = {}
    for b in rows:
        out.setdefault(b.item_id, []).append({"label": b.label, "kind": b.kind, "origin": b.origin, "evidence": _evidence_review(b)})
    return out


def _log_badges_for_source(db: Session, source: str) -> dict[int, list[dict]]:
    days = WINDOW_DAYS.get(source, 28)
    base = service_users(select(Event.item_id, func.count()).join(Item, Item.id == Event.item_id)
                         .where(Event.type == "purchase", Item.source == source, Event.is_setup == False))   # noqa: E712
    latest = db.scalar(service_users(select(func.max(Event.created_at)).join(Item, Item.id == Event.item_id)
                                     .where(Event.type == "purchase", Item.source == source, Event.is_setup == False)))  # noqa: E712
    if latest is None:
        return {}
    w0, w1 = latest - timedelta(days=days), latest - timedelta(days=2 * days)
    recent = dict(db.execute(base.where(Event.created_at > w0).group_by(Event.item_id)).all())
    if not recent:
        return {}
    prev = dict(db.execute(base.where(Event.created_at > w1, Event.created_at <= w0).group_by(Event.item_id)).all())
    cats = dict(db.execute(select(Item.id, Item.category).where(Item.source == source)).all())
    by_cat: dict[str, list[int]] = {}
    for i, c in cats.items():
        by_cat.setdefault(c or "", []).append(i)
    span = f"최근 {days}일"
    out: dict[int, list[dict]] = {}
    for cat, members in by_cat.items():
        counts = np.array([recent.get(i, 0) for i in members])
        if counts.max() < HOT_MIN:
            continue
        cut = max(HOT_MIN, float(np.quantile(counts, 1 - HOT_TOP)))
        for i, c in zip(members, counts.tolist()):
            if c >= cut:
                out.setdefault(i, []).append({"label": "최근 인기", "kind": "trend", "origin": "log",
                                              "evidence": f"{span} 구매 {c}건, 같은 카테고리 상위 {int(HOT_TOP * 100)}%"})
    for i, c in recent.items():
        p = prev.get(i, 0)
        if c >= RISING_MIN and c >= RISING_X * max(p, 1):
            out.setdefault(i, []).append({"label": "급상승", "kind": "trend", "origin": "log",
                                          "evidence": f"{span} 구매 {c}건 (그 전 {days}일 {p}건)"})
    return out


def log_badges(db: Session, ids: list[int]) -> dict[int, list[dict]]:
    """구매 로그 배지. 데이터셋별로 계산해 두고, 서비스 범위 마지막 구매 행이 바뀌면 다시 계산한다."""
    if not ids:
        return {}
    sources = [s for (s,) in db.execute(select(Item.source).where(Item.id.in_(ids)).distinct()).all()]
    out: dict[int, list[dict]] = {}
    for source in sources:
        sig = db.execute(service_users(select(Event.id, Event.item_id, Event.created_at).join(Item, Item.id == Event.item_id)
                                       .where(Event.type == "purchase", Item.source == source, Event.is_setup == False))  # noqa: E712
                         .order_by(Event.id.desc()).limit(1)).first()
        key = (str(db.bind.url), catalog_signature(db), tuple(sig) if sig is not None else None)
        cached = _log_cache.get(source)
        if cached is None or cached["key"] != key:
            cached = _log_cache[source] = {"key": key, "badges": _log_badges_for_source(db, source)}
        for i in ids:
            if i in cached["badges"]:
                out[i] = cached["badges"][i]
    return out


def badges_for(db: Session, ids: list[int], limit: int | None = MAX_CARD) -> dict[int, list[dict]]:
    """상품별 배지 목록 (구매 로그 → 리뷰 순). limit=None이면 전부 (상세 화면)."""
    logs, reviews = log_badges(db, ids), review_badges(db, ids)
    out = {}
    for i in ids:
        b = logs.get(i, []) + reviews.get(i, [])
        if b:
            out[i] = b if limit is None else b[:limit]
    return out
