"""기준 모델: 전체 인기 상품, 고객이 선호하는 브랜드 안의 인기 상품."""
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from ..models import Event, Item
from .base import Recommender, register

WEIGHTS = {"purchase": 3.0, "add_to_cart": 2.0, "click": 1.0, "view": 0.2}   # 이벤트 종류별 인기도 가중치


def _popular(db: Session, k: int, brand_ids=None, exclude=()):
    w = case(*[(Event.type == t, v) for t, v in WEIGHTS.items()], else_=0.0)
    q = (select(Event.item_id, func.sum(w).label("s"))
         .where(Event.item_id.is_not(None))
         .group_by(Event.item_id).order_by(func.sum(w).desc(), Event.item_id))
    if brand_ids:
        q = q.join(Item, Item.id == Event.item_id).where(Item.brand_id.in_(brand_ids))
    ids = [r.item_id for r in db.execute(q.limit(k + len(exclude)))]
    ids = [i for i in ids if i not in set(exclude)]
    if len(ids) < k:   # 로그가 적으면 상품 id 순으로 채운다 (시작 단계용)
        fill = select(Item.id).order_by(Item.id)
        if brand_ids:
            fill = fill.where(Item.brand_id.in_(brand_ids))
        for (i,) in db.execute(fill.limit(k * 3)):
            if i not in ids and i not in set(exclude):
                ids.append(i)
            if len(ids) >= k:
                break
    return ids[:k]


@register
class Popularity(Recommender):
    name = "popularity"

    def recommend(self, db, user_id, k):
        return _popular(db, k)


@register
class BrandPopularity(Recommender):
    """고객이 가장 많이 상호작용한 브랜드(params.top_brands개) 안의 인기 상품. 브랜드 충성도 기준 모델."""
    name = "brand_popularity"

    def recommend(self, db, user_id, k):
        n = int(self.params.get("top_brands", 2))
        q = (select(Item.brand_id, func.count().label("c"))
             .join(Event, Event.item_id == Item.id)
             .where(Event.user_id == user_id, Item.brand_id.is_not(None))
             .group_by(Item.brand_id).order_by(func.count().desc()).limit(n))
        brands = [r.brand_id for r in db.execute(q)]
        if not brands:
            return _popular(db, k)
        ids = _popular(db, k, brand_ids=brands)
        return ids if len(ids) >= k else ids + _popular(db, k - len(ids), exclude=ids)
