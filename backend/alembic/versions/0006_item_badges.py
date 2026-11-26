"""item_badges — 상품 키워드 배지 (D4, 리뷰에서 뽑은 배지)

Revision ID: 0006_item_badges
Revises: 0005_item_registered_at
Create Date: 2026-11-26

새 테이블이라 기존 데이터는 그대로. 비어 있으면 리뷰 배지 없이 동작한다 (적재: scripts/load_badges.py).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006_item_badges"
down_revision: Union[str, None] = "0005_item_registered_at"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "item_badges",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.String(40), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("origin", sa.String(16), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("n_mention", sa.Integer(), nullable=False),
        sa.Column("n_reviews", sa.Integer(), nullable=False),
        sa.Column("lift", sa.Float(), nullable=True),
    )
    op.create_index("ix_item_badges_item_id", "item_badges", ["item_id"])


def downgrade() -> None:
    op.drop_index("ix_item_badges_item_id", table_name="item_badges")
    op.drop_table("item_badges")
