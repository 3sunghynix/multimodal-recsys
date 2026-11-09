"""baseline schema (현재 main 의 ORM 테이블 전체)

Revision ID: 0001_baseline
Revises:
Create Date: 2026-11-07

이 마이그레이션 = `Base.metadata.create_all()` 이 지금까지 만들어 온 결과를 그대로 명시화한 것.
PR #14(pgvector) · #30(event_id, is_setup) · #31(brand_name은 schema 변화 없음) 등 지금까지의 모든
스키마 변화가 한 파일에 합쳐져 있다. 앞으로의 변경은 이 위에 쌓는다.

기존 Docker 환경에서 이미 테이블이 있는 경우: db.auto_migrate() 가 "테이블 있는데 version 없음" 을
감지해 upgrade 대신 `alembic stamp head` 를 자동으로 돌려준다 (수동 조치 불필요).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from app.config import settings
from app.types import EmbeddingVector

revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # PostgreSQL 이면 pgvector 확장 (SQLite 에서는 EmbeddingVector 가 JSON 으로 폴백 — no-op)
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "brands",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False, unique=True),
    )

    op.create_table(
        "items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("source_item_id", sa.String(64), nullable=False, index=True),
        sa.Column("brand_id", sa.Integer(), sa.ForeignKey("brands.id"), index=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("category", sa.String(200), nullable=False, server_default=""),
        sa.Column("price", sa.Float()),
        sa.Column("image_path", sa.String(500)),
        sa.Column("text_embedding", EmbeddingVector(settings.embedding_dim), nullable=True),
        sa.Column("image_embedding", EmbeddingVector(settings.embedding_dim), nullable=True),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.String(32), nullable=False, server_default="sim"),
        sa.Column("source_user_id", sa.String(64), index=True),
        sa.Column("persona", sa.String(64)),
        sa.UniqueConstraint("source", "source_user_id", name="uq_users_source_source_user_id"),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("version", sa.String(32), nullable=False, server_default="v1"),
        sa.Column("status", sa.String(16), nullable=False, server_default="candidate"),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("variants", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True),
        # event_id: 멱등 재전송 키 (PR #30). SQLAlchemy 2.0 Uuid 는 PG=native UUID, SQLite=CHAR(32).
        sa.Column("event_id", sa.Uuid(), nullable=False, unique=True, index=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), index=True),
        sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id"), index=True),
        sa.Column("type", sa.String(16), nullable=False),
        sa.Column("query", sa.String(500)),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_versions.id")),
        sa.Column("experiment_id", sa.Integer(), sa.ForeignKey("experiments.id")),
        sa.Column("variant", sa.String(32)),
        # is_setup: 초기 이력 적재 플래그 (PR #30). A/B 집계 · 공통 인기도에서 제외.
        sa.Column("is_setup", sa.Boolean(), nullable=False, server_default="0", index=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now(), index=True),
    )

    op.create_table(
        "metric_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_versions.id")),
        sa.Column("experiment_id", sa.Integer(), sa.ForeignKey("experiments.id")),
        sa.Column("variant", sa.String(32)),
        sa.Column("metric", sa.String(64), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    # FK 순서 뒤집어 drop — metric_snapshots, events 먼저, brands/items 마지막.
    op.drop_table("metric_snapshots")
    op.drop_table("events")
    op.drop_table("experiments")
    op.drop_table("model_versions")
    op.drop_table("users")
    op.drop_table("items")
    op.drop_table("brands")
    # pgvector 확장은 다른 DB 가 쓸 수 있어서 downgrade 에서 DROP 안 함 (명시적으로 안전)
