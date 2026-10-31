import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.db import Base
from app.models import Item
from scripts.export_simulator_items import export_items


def test_export_is_read_only_and_exclusive(tmp_path):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(Item(source="hm", source_item_id="000123", title="x", price=10))
        db.commit()
        out = tmp_path / "items.json"
        assert export_items(db, out) == 1
        row = json.loads(out.read_text(encoding="utf-8"))
        assert row["items"][0]["source_item_id"] == "000123"
        assert row["database_writes"] is False and not db.new and not db.dirty
        with pytest.raises(FileExistsError):
            export_items(db, out)
