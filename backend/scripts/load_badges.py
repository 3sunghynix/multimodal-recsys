"""리뷰 키워드 배지 적재 (D4): recsys/review_badges.py 결과(parquet)를 item_badges 테이블에 넣는다.

docker compose exec api python -m scripts.load_badges --file /data/badges/amazon_review_badges.parquet
같은 origin(review)의 기존 배지는 지우고 다시 넣는다 (다시 실행해도 안전). 원본 id(parent_asin)가 DB에 없는 상품은 건너뛴다.
"""
import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, select

from app.db import SessionLocal
from app.models import Item, ItemBadge


def load(db, df: pd.DataFrame, source: str = "amazon", origin: str = "review") -> dict:
    ids = dict(db.execute(select(Item.source_item_id, Item.id).where(Item.source == source)).all())
    db.execute(delete(ItemBadge).where(ItemBadge.origin == origin,
                                       ItemBadge.item_id.in_(select(Item.id).where(Item.source == source))))
    rows, missing = [], 0
    for r in df.itertuples(index=False):
        item_id = ids.get(str(r.parent_asin))
        if item_id is None:
            missing += 1
            continue
        rows.append({"item_id": item_id, "label": r.badge, "kind": r.kind, "origin": origin, "rank": int(r.rank),
                     "n_mention": int(r.n_mention), "n_reviews": int(r.n_reviews),
                     "lift": float(r.lift) if pd.notna(r.lift) else None})
    if rows:
        db.execute(ItemBadge.__table__.insert(), rows)
    db.commit()
    return {"inserted": len(rows), "items": len({r["item_id"] for r in rows}), "missing_items": missing}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--source", default="amazon")
    a = ap.parse_args()
    with SessionLocal() as db:
        print(load(db, pd.read_parquet(a.file), a.source))


if __name__ == "__main__":
    main()
