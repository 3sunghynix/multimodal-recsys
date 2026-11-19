"""Alembic 마이그레이션이 한 줄로 이어지는지 (head 1개). 두 PR이 같은 부모로 마이그레이션을 추가하면
테스트(create_all)는 통과해도 api 시작 시 `alembic upgrade head`가 multiple heads로 실패한다 (#56 · #57 · #58)."""
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

BACKEND = Path(__file__).resolve().parents[1]


def test_single_alembic_head():
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    heads = ScriptDirectory.from_config(cfg).get_heads()
    assert len(heads) == 1, f"Alembic head가 여러 개입니다: {heads} — 새 마이그레이션의 down_revision을 현재 head로 바꾸세요"


def test_reset_db_drops_alembic_version_so_tables_are_recreated(tmp_path):
    """load_dev_subset --reset: drop_all만 하면 alembic_version이 남아 Postgres에서 init_db가 upgrade를
    건너뛰고 테이블 없는 DB가 됐다 (11-18 재현 확인). reset_db는 표시까지 지우고 다시 만든다."""
    from sqlalchemy import inspect, text
    from sqlalchemy.orm import sessionmaker

    from app import db as app_db
    from app.models import User

    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'reset.db'}")
    app_db.init_db(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version VALUES ('0005_item_registered_at')"))
    with sessionmaker(bind=engine)() as s:
        s.add(User(source="sim"))
        s.commit()

    app_db.reset_db(engine)

    tables = set(inspect(engine).get_table_names())
    assert "alembic_version" not in tables
    assert {"items", "users", "events", "recommend_requests", "metric_snapshots"} <= tables
    with sessionmaker(bind=engine)() as s:
        assert s.query(User).count() == 0
