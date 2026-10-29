"""데모용 작은 데이터 (브랜드 3 · 상품 30 · 고객 10 · 이벤트 약 150, 기준 모델 2개). 테스트 · 빠른 실행용.

실제 멀티모달 검색 데모는 PR #15의 load_dev_subset으로 돌린다 (브랜드 21 · 상품 ~2,000 · Amazon/H&M
이미지 + Marqo PCA 256 임베딩이 드라이브 msrs_project/embeddings/v2에 있다):
    docker compose exec api python -m scripts.load_dev_subset --data /data/dev_subset --embeddings /data/embeddings/v2

이 스크립트는 image_embedding을 채우지 않는다. text 임베딩을 image 열에 넣으면 열의 의미가 꼬이고,
실제 이미지와 다른 공간이라 /search/image 결과가 왜곡된다.

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
