from sqlalchemy import create_engine, text
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


def init_db(bind=None):
    from . import models  # noqa: F401  테이블 등록
    bind = bind or engine
    # PostgreSQL이면 pgvector 확장을 먼저 켠다. items 테이블의 임베딩 컬럼이 이 확장에 의존한다.
    # SQLite(테스트)에서는 EmbeddingVector 타입이 JSON으로 fallback 되므로 아무 것도 하지 않는다.
    if bind.dialect.name == "postgresql":
        with bind.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=bind)
