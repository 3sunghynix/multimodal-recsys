"""EmbeddingVector 타입이 list·numpy 둘 다 안전하게 저장하고 통일된 타입으로 돌려주는지 검증.

운영은 PostgreSQL(pgvector)이고 테스트는 SQLite(JSON)지만, bind/result 처리는 두 경로 모두
EmbeddingVector의 process_bind_param · process_result_value 가 담당하므로 SQLite로도 동작을 검증한다.
"""
import numpy as np
import pytest
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.models import Brand, Item


@pytest.fixture()
def session(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'types.db'}")
    app_db.init_db(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Session() as s:
        yield s


def test_bind_accepts_list(session):
    b = Brand(name="X")
    session.add(b)
    session.flush()
    session.add(Item(source="t", source_item_id="list", brand_id=b.id, title="L",
                     image_embedding=[0.1, 0.2, 0.3]))
    session.commit()
    row = session.query(Item).filter_by(source_item_id="list").one()
    assert isinstance(row.image_embedding, list)
    assert row.image_embedding == [0.1, 0.2, 0.3]


def test_bind_accepts_numpy_array(session):
    b = Brand(name="Y")
    session.add(b)
    session.flush()
    session.add(Item(source="t", source_item_id="np", brand_id=b.id, title="N",
                     image_embedding=np.array([0.4, 0.5, 0.6], dtype=np.float32)))
    session.commit()
    row = session.query(Item).filter_by(source_item_id="np").one()
    # 결과는 방언 무관하게 list로 돌아와야 한다
    assert isinstance(row.image_embedding, list)
    assert row.image_embedding == pytest.approx([0.4, 0.5, 0.6])


def test_null_roundtrip(session):
    b = Brand(name="Z")
    session.add(b)
    session.flush()
    session.add(Item(source="t", source_item_id="null", brand_id=b.id, title="X",
                     image_embedding=None))
    session.commit()
    row = session.query(Item).filter_by(source_item_id="null").one()
    assert row.image_embedding is None
