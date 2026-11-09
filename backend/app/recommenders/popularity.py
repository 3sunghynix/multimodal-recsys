"""기준 모델: 전체 인기 상품, 고객이 선호하는 브랜드 안의 인기 상품."""
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from ..models import Event, Item
from .base import Recommender, register

WEIGHTS = {"purchase": 3.0, "add_to_cart": 2.0, "click": 1.0, "view": 0.2}   # 이벤트 종류별 인기도 가중치


def _popular_scored(db: Session, k: int, brand_ids=None, exclude=()) -> list[tuple[int, float]]:
    """공통 인기도 (id, score) 쌍. score 는 이벤트 가중치 합계(view 0.2 ~ purchase 3.0).
    is_setup=True 는 초기 이력 적재 전용이므로 집계에서 제외 (개인화 경로는 포함)."""
    w = case(*[(Event.type == t, v) for t, v in WEIGHTS.items()], else_=0.0)
    q = (select(Event.item_id, func.sum(w).label("s"))
         .where(Event.item_id.is_not(None), Event.is_setup == False)                    # noqa: E712
         .group_by(Event.item_id).order_by(func.sum(w).desc(), Event.item_id))
    if brand_ids:
        q = q.join(Item, Item.id == Event.item_id).where(Item.brand_id.in_(brand_ids))
    excluded = set(exclude)
    scored: list[tuple[int, float]] = [
        (int(r.item_id), float(r.s)) for r in db.execute(q.limit(k + len(excluded)))
    ]
    scored = [(i, s) for i, s in scored if i not in excluded]
    if len(scored) < k:   # 로그가 적으면 상품 id 순으로 채운다 (score=0 백필, 시작 단계용)
        seen = {i for i, _ in scored}
        fill = select(Item.id).order_by(Item.id)
        if brand_ids:
            fill = fill.where(Item.brand_id.in_(brand_ids))
        for (i,) in db.execute(fill.limit(k * 3)):
            if i not in seen and i not in excluded:
                scored.append((i, 0.0))
                seen.add(i)
            if len(scored) >= k:
                break
    return scored[:k]


def _popular(db: Session, k: int, brand_ids=None, exclude=()) -> list[int]:
    """하위호환: id 만 돌려준다. UME 등 다른 모델에서 폴백으로 호출됨."""
    return [i for i, _ in _popular_scored(db, k, brand_ids, exclude)]


@register
class Popularity(Recommender):
    name = "popularity"
    scored_native = True              # 이벤트 가중치 합계가 실제 모델 점수
    score_basis = "event_weight_sum"

    def recommend(self, db, user_id, k):
        return _popular(db, k)

    def recommend_scored(self, db, user_id, k):
        return _popular_scored(db, k)


@register
class BrandPopularity(Recommender):
    """고객이 가장 많이 상호작용한 브랜드(params.top_brands개) 안의 인기 상품. 브랜드 충성도 기준 모델."""
    name = "brand_popularity"
    scored_native = True              # 브랜드 안의 인기도 가중치 합계가 실제 모델 점수
    score_basis = "event_weight_sum"

    def recommend(self, db, user_id, k):
        return [i for i, _ in self.recommend_scored(db, user_id, k)]

    def recommend_scored(self, db, user_id, k):
        n = int(self.params.get("top_brands", 2))
        q = (select(Item.brand_id, func.count().label("c"))
             .join(Event, Event.item_id == Item.id)
             .where(Event.user_id == user_id, Item.brand_id.is_not(None))
             .group_by(Item.brand_id).order_by(func.count().desc()).limit(n))
        brands = [r.brand_id for r in db.execute(q)]
        if not brands:
            return _popular_scored(db, k)
        scored = _popular_scored(db, k, brand_ids=brands)
        if len(scored) >= k:
            return scored
        excluded = {i for i, _ in scored}
        return scored + _popular_scored(db, k - len(scored), exclude=excluded)
