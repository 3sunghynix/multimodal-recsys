"""데모용 작은 데이터 (브랜드 3 · 상품 30 · 고객 10 · 이벤트 약 150, 기준 모델 2개). 실제 데이터 적재 스크립트가 생기기 전까지 사용.

실행 (backend/ 에서): python -m scripts.seed_sample
"""
import random

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import Brand, Event, Item, ModelVersion, User

BRANDS = ["Alpha", "Beta", "Gamma"]
CATEGORIES = ["T-shirt", "Dress", "Trousers", "Sneakers", "Jacket"]
COLORS = ["black", "white", "blue", "red", "beige", "green"]


def seed(db, n_items=30, n_users=10, n_events=150, seed=42):
    if db.scalars(select(Item)).first() is not None:
        return False
    rng = random.Random(seed)
    brands = [Brand(name=b) for b in BRANDS]
    db.add_all(brands)
    db.flush()
    items = []
    for i in range(n_items):
        b, c, col = brands[i % len(brands)], CATEGORIES[i % len(CATEGORIES)], rng.choice(COLORS)
        items.append(Item(source="sample", source_item_id=f"s{i}", brand_id=b.id, title=f"{b.name} {col} {c}",
                          description=f"{col} {c.lower()} by {b.name}", category=c, price=round(rng.uniform(10, 120), 2)))
    users = [User(source="sample", source_user_id=f"u{u}") for u in range(n_users)]
    db.add_all(items + users)
    db.flush()
    for _ in range(n_events):
        u = rng.choice(users)
        fav = brands[u.id % len(brands)]                 # 고객마다 선호 브랜드 (브랜드 충성도 예시)
        pool = [it for it in items if it.brand_id == fav.id] if rng.random() < 0.7 else items
        db.add(Event(user_id=u.id, item_id=rng.choice(pool).id,
                     type=rng.choices(["view", "click", "add_to_cart", "purchase"], [6, 3, 1, 1])[0]))
    db.add_all([ModelVersion(name="popularity", status="active"),
                ModelVersion(name="brand_popularity", params={"top_brands": 1})])
    db.commit()
    return True


if __name__ == "__main__":
    init_db()
    with SessionLocal() as db:
        print("샘플 데이터 적재" if seed(db) else "이미 데이터가 있어 건너뜀")
