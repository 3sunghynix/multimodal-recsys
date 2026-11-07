"""Alembic 환경. app.config.settings 의 database_url 로 붙고, app.models 의 metadata 를 대상으로 삼는다.

- offline: SQL 파일로 떨어뜨리는 모드 (DB 접속 없음). CI 에서 스키마 미리보기에 쓸 수 있다.
- online: 실제 DB 에 연결해 마이그레이션을 돌린다. docker compose 상에서 사용.
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# app 모듈이 PYTHONPATH 에 있어야 한다. docker 안에서는 /app 이 작업 디렉토리라 자동으로 해결됨.
from app.config import settings
from app.db import Base
from app import models  # noqa: F401 — Base.metadata 에 테이블 등록

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
