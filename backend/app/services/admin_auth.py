"""관리자 쓰기 엔드포인트 (B4) 간단 토큰 인증.

보호 대상: `admin.py` 의 상태 변경 엔드포인트 — POST /models, /models/{id}/activate, /experiments, /experiments/{id}/stop.
읽기(GET)는 그대로 공개 — 대시보드 조회 · 프론트에서 자유롭게 부른다.

규칙:
    · `settings.admin_api_key` 가 비어 있으면 통과 (로컬 개발 · CI · 기존 테스트 호환, 서버 시작 시 WARN 1회).
    · 설정되어 있으면 `X-Admin-Key: <값>` 헤더가 정확히 일치해야 함 (상수 시간 비교). 없음 · 틀림 → 401.

범위 밖 (후속): `POST /metrics/snapshots` — trainer/monitoring 자동화가 호출하는 경로라 scope 확장 필요."""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from ..config import settings


def require_admin(x_admin_key: str | None = Header(None, alias="X-Admin-Key")) -> None:
    key = settings.admin_api_key
    if not key:                                                        # auth disabled (dev · tests)
        return
    if not x_admin_key or not hmac.compare_digest(x_admin_key, key):
        raise HTTPException(status_code=401, detail="관리자 키 필요 (X-Admin-Key 헤더)")
