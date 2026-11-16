"""events.request_id · events.position — 추천 요청 역조회용 (이벤트연결규약 §2.2, PR #57 ③)

Revision ID: 0004_events_request_id_position
Revises: 0003_recommend_requests
Create Date: 2026-11-16 03:48:20.170011

이벤트에 request_id 가 있으면 서버가 recommend_requests 역조회로 model_version_id ·
experiment_id · variant 를 자동 채운다 (톡방 결정 5). position 은 추천 목록의 몇 번째
자리였는지 (1부터). /metrics/window 의 노출 view 는 request_id 가 있는 이벤트만 센다
(톡방 결정 3, user_click_through_v1).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_events_request_id_position"
down_revision: Union[str, None] = "0003_recommend_requests"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("events", sa.Column("request_id", sa.Uuid(), nullable=True))
    op.add_column("events", sa.Column("position", sa.Integer(), nullable=True))
    op.create_index(op.f("ix_events_request_id"), "events", ["request_id"], unique=False)
    op.create_foreign_key(
        "fk_events_request_id_recommend_requests",
        "events", "recommend_requests",
        ["request_id"], ["request_id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_events_request_id_recommend_requests", "events", type_="foreignkey")
    op.drop_index(op.f("ix_events_request_id"), table_name="events")
    op.drop_column("events", "position")
    op.drop_column("events", "request_id")
