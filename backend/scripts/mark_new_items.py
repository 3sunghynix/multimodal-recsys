"""데모 신규 상품 지정 — items.registered_at을 최근 시각으로 채운다.

공개 데이터셋에는 등록일이 없어서 적재 상품은 registered_at이 비어 있다 (= 신규 아님).
신규 상품 노출 보장(명세: 등록 7일 이내)을 확인하려면 일부 상품을 신규로 지정한다.

    docker compose exec api python -m scripts.mark_new_items --count 200            # 무작위 200개, 최근 7일 안
    docker compose exec api python -m scripts.mark_new_items --ids 101 102 103 --days 0
    docker compose exec api python -m scripts.mark_new_items --clear                 # 전부 신규 아님으로
"""
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import select, update

from app.db import SessionLocal, init_db
from app.models import Item


def mark(db, ids: list[int] | None = None, count: int = 0, days: int = 7, seed: int = 42,
         now: datetime | None = None) -> list[int]:
    """ids(또는 무작위 count개)의 registered_at = now − [0, days)일 무작위. 반환: 지정한 id (정렬)."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)     # DB 시각은 naive UTC
    rng = np.random.default_rng(seed)
    if not ids:
        all_ids = np.array(db.scalars(select(Item.id).order_by(Item.id)).all(), np.int64)
        ids = rng.choice(all_ids, size=min(count, len(all_ids)), replace=False).tolist() if len(all_ids) else []
    ids = sorted(int(i) for i in ids)
    for i in ids:
        ago = timedelta(seconds=float(rng.uniform(0, max(days, 0) * 86400))) if days > 0 else timedelta(0)
        db.execute(update(Item).where(Item.id == i).values(registered_at=now - ago))
    db.commit()
    return ids


def clear(db) -> None:
    db.execute(update(Item).values(registered_at=None))
    db.commit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=0, help="무작위로 고를 상품 수")
    ap.add_argument("--ids", type=int, nargs="*", help="지정할 상품 id")
    ap.add_argument("--days", type=int, default=7, help="최근 며칠 안에 등록된 것으로 할지 (0 = 지금)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clear", action="store_true", help="모든 상품을 신규 아님으로 되돌림")
    a = ap.parse_args()
    if not a.clear and not a.ids and a.count <= 0:
        ap.error("--count, --ids, --clear 중 하나가 필요합니다")
    init_db()
    with SessionLocal() as db:
        if a.clear:
            clear(db)
            print("모든 상품 registered_at = NULL")
        else:
            ids = mark(db, a.ids, a.count, a.days, a.seed)
            print(f"신규 상품 {len(ids)}개 지정: {ids[:20]}{' …' if len(ids) > 20 else ''}")


if __name__ == "__main__":
    main()
