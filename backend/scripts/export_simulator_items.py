"""Read-only product-ID snapshot. No init_db, inserts, updates, or commits.

python -m scripts.export_simulator_items --out /runs/db_items_001.json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Item


def export_items(db, out):
    out = Path(out)
    records = [dict(id=r.id, source=r.source, source_item_id=r.source_item_id)
               for r in db.execute(select(Item.id, Item.source, Item.source_item_id).order_by(Item.id))]
    payload = {"kind": "simulator_db_item_snapshot_v1", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
               "database_writes": False, "items": records}
    # Exclusive creation also prevents accidental replacement by simultaneous exports.
    with out.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return len(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with SessionLocal() as db:
        count = export_items(db, args.out)
    print(f"Exported {count} product IDs to {args.out}. Database was not changed.")


if __name__ == "__main__":
    main()
