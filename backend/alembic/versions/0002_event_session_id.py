"""events.session_id — 클라이언트 세션 id (이벤트연결규약 2.4)

Revision ID: 0002_event_session_id
Revises: 0001_baseline
Create Date: 2026-11-12

시뮬레이터처럼 한 고객의 세션을 몇 초 간격으로 보내면 서버 시각 30분 규칙으로는 세션이 합쳐진다.
이벤트에 session_id(선택)를 저장하고, 있으면 현재 세션을 이 값으로 구분한다 (services/session.py).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_event_session_id"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("events", sa.Column("session_id", sa.String(64), nullable=True))
    op.create_index("ix_events_session_id", "events", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_events_session_id", table_name="events")
    op.drop_column("events", "session_id")
