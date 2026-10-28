"""드라이브 msrs_project/dev_subset (또는 processed/amazon)을 서비스 DB에 적재한다.

입력 폴더: items · brands · users · events.parquet (+ 선택: images.zip)
선택: --embeddings 폴더(msrs_project/embeddings/amazon)를 주면 items의 콘텐츠 임베딩 열(text_embedding · image_embedding,
      PR #14)에 PCA 256 벡터를 넣는다. 열이 없는 스키마면 건너뛴다.

실행 (backend/ 에서):
    python -m scripts.load_dev_subset --data ../data/dev_subset [--embeddings ../data/embeddings/amazon] [--reset]
Docker:
    docker compose exec api python -m scripts.load_dev_subset --data /data/dev_subset   (data 폴더 마운트 필요)
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.db import Base, SessionLocal, engine, init_db
from app.models import Brand, Event, Item, ModelVersion, User

CHUNK = 5_000


def _none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def load(db, data: Path, embeddings: Path | None = None, extract_images: bool = True):
    items = pd.read_parquet(data / "items.parquet")
    brands = pd.read_parquet(data / "brands.parquet")
    users = pd.read_parquet(data / "users.parquet")
    events = pd.read_parquet(data / "events.parquet")
    if db.scalars(select(Item).where(Item.source == "amazon")).first() is not None:
        raise SystemExit("이미 amazon 데이터가 있습니다. 다시 넣으려면 --reset (모든 테이블 초기화)")

    # 브랜드: 이미 같은 이름이 있으면 재사용
    existing = {b.name: b.id for b in db.scalars(select(Brand))}
    new = [Brand(name=n) for n in brands.brand if n not in existing]
    db.add_all(new)
    db.flush()
    existing.update({b.name: b.id for b in new})
    brand_db = {int(r.brand_id): existing[r.brand] for r in brands.itertuples()}

    img_dir = None
    if extract_images and (data / "images.zip").exists():
        img_dir = data / "images"
        if not img_dir.exists():
            zipfile.ZipFile(data / "images.zip").extractall(img_dir)

    vec = {}
    if embeddings is not None and hasattr(Item, "text_embedding"):
        ids = np.load(embeddings / "item_ids.npy")
        pos = {int(i): k for k, i in enumerate(ids)}
        text, image = np.load(embeddings / "text_pca256.npy"), np.load(embeddings / "image_pca256.npy")
        has = np.load(embeddings / "has_image.npy")
        for i in items.item_id:
            k = pos[int(i)]
            vec[int(i)] = (text[k].tolist(), image[k].tolist() if has[k] else None)
    elif embeddings is not None:
        print("주의: items에 임베딩 열이 없어 임베딩은 건너뜀 (PR #14 병합 후 다시 적재)")

    item_db = {}
    for start in range(0, len(items), CHUNK):
        objs = []
        for r in items.iloc[start:start + CHUNK].itertuples():
            desc = " ".join(x for x in (_none(r.features), _none(r.description)) if x)
            kw = dict(source="amazon", source_item_id=str(r.source_item_id), brand_id=brand_db.get(int(r.brand_id)),
                      title=str(r.title)[:500], description=desc, category=str(_none(r.categories) or "")[:200],
                      price=_none(r.price), image_path=f"dev_subset/images/{int(r.item_id)}.jpg" if img_dir else None)
            if vec:
                kw["text_embedding"], kw["image_embedding"] = vec[int(r.item_id)]
            objs.append((int(r.item_id), Item(**kw)))
        db.add_all([o for _, o in objs])
        db.flush()
        item_db.update({i: o.id for i, o in objs})

    user_db = {}
    for start in range(0, len(users), CHUNK):
        objs = [(int(r.user_id), User(source="amazon", source_user_id=str(r.source_user_id)))
                for r in users.iloc[start:start + CHUNK].itertuples()]
        db.add_all([o for _, o in objs])
        db.flush()
        user_db.update({u: o.id for u, o in objs})

    ev = events[events.user_id.isin(user_db.keys()) & events.item_id.isin(item_db.keys())]
    for start in range(0, len(ev), CHUNK):
        db.add_all([Event(user_id=user_db[int(r.user_id)], item_id=item_db[int(r.item_id)], type=str(r.type),
                          created_at=pd.Timestamp(r.ts).to_pydatetime())
                    for r in ev.iloc[start:start + CHUNK].itertuples()])
        db.flush()

    if db.scalars(select(ModelVersion)).first() is None:     # 처음 적재면 기준 모델 등록
        db.add_all([ModelVersion(name="popularity", status="active"),
                    ModelVersion(name="brand_popularity", params={"top_brands": 2})])
    db.commit()
    return {"brands": len(new), "items": len(item_db), "users": len(user_db), "events": len(ev),
            "embeddings": bool(vec), "images": img_dir is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("../data/dev_subset"))
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--reset", action="store_true", help="모든 테이블을 지우고 다시 만든다 (로컬 개발용)")
    ap.add_argument("--no-images", action="store_true", help="images.zip 압축 해제를 건너뛴다")
    a = ap.parse_args()
    if a.reset:
        Base.metadata.drop_all(bind=engine)
    init_db()
    with SessionLocal() as db:
        print("적재 완료:", load(db, a.data, a.embeddings, extract_images=not a.no_images))


if __name__ == "__main__":
    main()
