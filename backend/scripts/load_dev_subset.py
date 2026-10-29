"""드라이브 msrs_project/dev_subset (v2: Amazon + H&M)을 서비스 DB에 적재한다.

입력 폴더: items · brands · users · events.parquet (+ 선택: images.zip)
  - 모든 표에 source(amazon · hm) 열. item_id · user_id는 source 안에서만 유일하므로 (source, id)로 연결한다
  - 브랜드는 brand_norm(표기 통일)이 같으면 같은 브랜드, 표시 이름은 brand
  - images.zip 안 경로: {source}/{item_id}.jpg (v1 형식 {item_id}.jpg는 amazon으로 간주)
선택: --embeddings 폴더(msrs_project/embeddings/v2)를 주면 items의 콘텐츠 임베딩 열(text_embedding · image_embedding)에
      PCA 256 벡터를 넣는다. catalog.parquet(row, source, item_id)로 행을 찾는다. 열이 없는 스키마면 건너뛴다.

실행 (backend/ 에서):
    python -m scripts.load_dev_subset --data ../data/dev_subset [--embeddings ../data/embeddings/v2] [--reset]
Docker:
    docker compose exec api python -m scripts.load_dev_subset --data /data/dev_subset [--embeddings /data/embeddings/v2]
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


def _with_source(df, default="amazon"):
    return df if "source" in df.columns else df.assign(source=default)


def _embeddings(path: Path, keys):
    """(source, item_id) → (text 벡터, image 벡터 또는 None)."""
    if (path / "catalog.parquet").exists():                    # v2: Amazon + H&M 통합
        cat = pd.read_parquet(path / "catalog.parquet")
        row = {(s, int(i)): int(r) for r, s, i in zip(cat.row, cat.source, cat.item_id)}
    else:                                                      # v1: Amazon 단독 (행 = item_ids.npy 순서)
        row = {("amazon", int(i)): k for k, i in enumerate(np.load(path / "item_ids.npy"))}
    text, image = np.load(path / "text_pca256.npy"), np.load(path / "image_pca256.npy")
    has = np.load(path / "has_image.npy")
    return {key: (text[row[key]].tolist(), image[row[key]].tolist() if has[row[key]] else None)
            for key in keys if key in row}


def load(db, data: Path, embeddings: Path | None = None, extract_images: bool = True):
    items = _with_source(pd.read_parquet(data / "items.parquet"))
    users = _with_source(pd.read_parquet(data / "users.parquet"))
    events = _with_source(pd.read_parquet(data / "events.parquet"))
    sources = sorted(items.source.unique())
    if db.scalars(select(Item).where(Item.source.in_(sources))).first() is not None:
        raise SystemExit(f"이미 {sources} 데이터가 있습니다. 다시 넣으려면 --reset (모든 테이블 초기화)")

    # 브랜드: brand_norm 기준으로 하나, 이름이 이미 DB에 있으면 재사용
    if "brand_norm" not in items.columns:
        items = items.assign(brand_norm=items.brand.str.lower())
    names = items.groupby("brand_norm").brand.agg(lambda s: s.value_counts().index[0])
    existing = {b.name: b.id for b in db.scalars(select(Brand))}
    new = [Brand(name=n) for n in sorted(set(names.values) - set(existing))]
    db.add_all(new)
    db.flush()
    existing.update({b.name: b.id for b in new})
    brand_db = {norm: existing[name] for norm, name in names.items()}

    img_dir = None
    if extract_images and (data / "images.zip").exists():
        img_dir = data / "images"
        if not img_dir.exists():
            with zipfile.ZipFile(data / "images.zip") as z:
                for n in z.namelist():                         # v1 형식(item_id.jpg)은 amazon/ 아래로
                    target = img_dir / (n if "/" in n else f"amazon/{n}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(n))

    keys = list(zip(items.source, items.item_id.astype(int)))
    vec = {}
    if embeddings is not None and hasattr(Item, "text_embedding"):
        vec = _embeddings(embeddings, keys)
        missing = len(keys) - len(vec)
        if missing:
            print(f"주의: 임베딩이 없는 상품 {missing}개 (빈 값으로 적재)")
    elif embeddings is not None:
        print("주의: items에 임베딩 열이 없어 임베딩은 건너뜀 (PR #14 병합 후 다시 적재)")
    has_price_source = hasattr(Item, "price_source")

    item_db = {}
    for start in range(0, len(items), CHUNK):
        objs = []
        for r in items.iloc[start:start + CHUNK].itertuples():
            key = (r.source, int(r.item_id))
            desc = " ".join(x for x in (_none(getattr(r, "features", None)), _none(getattr(r, "description", None))) if x)
            cat = " > ".join(x for x in (_none(getattr(r, "cat_target", None)), _none(getattr(r, "cat_type", None))) if x)
            kw = dict(source=r.source, source_item_id=str(r.source_item_id), brand_id=brand_db.get(r.brand_norm),
                      title=str(r.title)[:500], description=desc,
                      category=(cat or str(_none(getattr(r, "categories", None)) or ""))[:200],
                      price=_none(getattr(r, "price", None)),
                      image_path=f"dev_subset/images/{r.source}/{int(r.item_id)}.jpg" if img_dir else None)
            if has_price_source:
                kw["price_source"] = _none(getattr(r, "price_source", None))
            if key in vec:
                kw["text_embedding"], kw["image_embedding"] = vec[key]
            objs.append((key, Item(**kw)))
        db.add_all([o for _, o in objs])
        db.flush()
        item_db.update({k: o.id for k, o in objs})

    user_db = {}
    for start in range(0, len(users), CHUNK):
        objs = [((r.source, int(r.user_id)), User(source=r.source, source_user_id=str(r.source_user_id)))
                for r in users.iloc[start:start + CHUNK].itertuples()]
        db.add_all([o for _, o in objs])
        db.flush()
        user_db.update({k: o.id for k, o in objs})

    n_ev = 0
    for start in range(0, len(events), CHUNK):
        objs = []
        for r in events.iloc[start:start + CHUNK].itertuples():
            u, i = user_db.get((r.source, int(r.user_id))), item_db.get((r.source, int(r.item_id)))
            if u is not None and i is not None:
                objs.append(Event(user_id=u, item_id=i, type=str(r.type), created_at=pd.Timestamp(r.ts).to_pydatetime()))
        db.add_all(objs)
        db.flush()
        n_ev += len(objs)

    if db.scalars(select(ModelVersion)).first() is None:     # 처음 적재면 기준 모델 등록
        db.add_all([ModelVersion(name="popularity", status="active"),
                    ModelVersion(name="brand_popularity", params={"top_brands": 2})])
    db.commit()
    return {"sources": sources, "brands": len(new), "items": len(item_db), "users": len(user_db), "events": n_ev,
            "embeddings": len(vec), "images": img_dir is not None}


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
