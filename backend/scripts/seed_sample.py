"""데모용 작은 데이터 (브랜드 3 · 상품 30 · 고객 10 · 이벤트 약 150, 기준 모델 2개). 실제 데이터 적재 스크립트가 생기기 전까지 사용.

실행 (backend/ 에서): python -m scripts.seed_sample

PostgreSQL + CLIP이 사용 가능하면 상품 title을 CLIP text encoder로 임베딩해 image_embedding
자리에 저장한다. CLIP은 image/text 공유 공간이라 /search/image의 이미지 쿼리와 바로 비교된다.
SQLite 또는 encode 실패 시에는 임베딩 없이 적재한다(기능 테스트는 가능).
"""
import random

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal, init_db
from app.models import Brand, Event, Item, ModelVersion, User

BRANDS = ["Alpha", "Beta", "Gamma"]
CATEGORIES = ["T-shirt", "Dress", "Trousers", "Sneakers", "Jacket"]
COLORS = ["black", "white", "blue", "red", "beige", "green"]


def seed(db, n_items=30, n_users=10, n_events=150, seed=42, with_embeddings=True):
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
    # CLIP text encoder로 상품 title을 임베딩해 image_embedding 자리에 넣는다 (CLIP의 공유 공간)
    if with_embeddings and db.bind.dialect.name == "postgresql":
        _attach_text_embeddings(items)
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


def _attach_text_embeddings(items):
    """상품 title을 CLIP text encoder로 묶어 인코딩해 image_embedding 컬럼에 저장한다."""
    try:
        from app.encoders.clip_image import load_encoder
        enc = load_encoder(settings.image_encoder_model)
        vecs = enc.encode_texts([it.title for it in items])   # (N, dim) L2 정규화 완료
        for it, v in zip(items, vecs):
            it.image_embedding = v.tolist()
        print(f"CLIP text embedding으로 {len(items)}개 상품의 image_embedding을 채움")
    except Exception as e:
        print(f"[주의] CLIP 로드 실패 — 임베딩 없이 적재 ({e})")


if __name__ == "__main__":
    init_db()
    with SessionLocal() as db:
        print("샘플 데이터 적재" if seed(db) else "이미 데이터가 있어 건너뜀")
