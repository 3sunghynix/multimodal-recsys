"""items.registered_at — 신규 상품(등록 7일 이내) 노출 보장용 등록 시각

Revision ID: 0002_item_registered_at
Revises: 0001_baseline
Create Date: 2026-11-12

nullable 열 추가라 기존 데이터는 그대로 (전부 NULL = 신규 아님). 재적재 · down -v 불필요.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_item_registered_at"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("items", sa.Column("registered_at", sa.DateTime(), nullable=True))
    op.create_index("ix_items_registered_at", "items", ["registered_at"])


def downgrade() -> None:
    op.drop_index("ix_items_registered_at", table_name="items")
    op.drop_column("items", "registered_at")
