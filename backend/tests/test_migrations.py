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
