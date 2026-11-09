from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


def make_engine(url):
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    return create_engine(url, **kwargs)


engine = make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_alembic(bind):
    """Alembic 으로 스키마를 최신화. 세 가지 상태를 자동 분기:
    1) alembic_version 테이블 있음 → upgrade head (대기 중 마이그레이션 적용, 없으면 no-op)
    2) alembic_version 없고 items 테이블 이미 있음 (기존 create_all 로 만들어진 팀 DB)
         → stamp head: 테이블 재생성 시도하지 않고 "현재 head 라고 표시" 만. 데이터 보존.
    3) alembic_version 도 items 도 없음 → upgrade head (baseline 마이그레이션이 테이블 생성)
    """
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext

    cfg = Config(str(Path(__file__).parent.parent / "alembic.ini"))

    with bind.connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()
    if current is None and inspect(bind).has_table("items"):
        command.stamp(cfg, "head")                                # (2) 팀 기존 DB 보존
    else:
        command.upgrade(cfg, "head")                              # (1), (3) 공통 경로


def init_db(bind=None):
    """Postgres: pgvector 확장 + Alembic 으로 스키마 적용 (베이스라인 생성 또는 upgrade).
    SQLite(테스트): Base.metadata.create_all — Alembic 무거운 setup 을 피하고 빠르게 돈다.

    하위호환: 기존 테스트는 init_db(engine) 을 호출해 SQLite 에 테이블을 만들어 왔다 — 그 경로 동일 유지.
    """
    from . import models  # noqa: F401  테이블 등록
    bind = bind or engine
    if bind.dialect.name == "postgresql":
        with bind.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        _run_alembic(bind)
    else:
        Base.metadata.create_all(bind=bind)
