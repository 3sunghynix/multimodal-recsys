"""고객 생성. 시뮬레이터·외부 데이터셋 적재에서 쓰는 멱등 upsert.

POST /users: (source, source_user_id)가 이미 있으면 저장된 user를 반환(200).
  내용(persona)이 들어 있고 저장된 값과 다르면 409로 거절한다. source_user_id가 None이면 익명 신규 생성.
  동시 요청이 pre-check 뒤 INSERT 로 가는 사이 다른 요청이 먼저 저장하면 UNIQUE 충돌로 500 이 나는 레이스를
  IntegrityError 로 잡아 롤백 후 재조회해 같은 멱등 로직(200 · 409) 으로 응답한다.
"""
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User
from ..schemas import UserIn, UserOut

router = APIRouter(prefix="/users", tags=["users"])


def _idempotent_or_409(existing: User, u: UserIn) -> User:
    # 요청에서 persona를 지정했는데 저장값과 다르면 조용히 덮지 않는다 — 다른 고객을 가리키려는 것일 수 있다.
    if u.persona is not None and existing.persona is not None and u.persona != existing.persona:
        raise HTTPException(409, "같은 (source, source_user_id)에 다른 persona가 이미 저장돼 있습니다")
    return existing


@router.post("", response_model=UserOut, status_code=201)
def create_user(u: UserIn, response: Response, db: Session = Depends(get_db)):
    if u.source_user_id is not None:
        existing = db.scalar(select(User).where(
            User.source == u.source, User.source_user_id == u.source_user_id))
        if existing is not None:
            response.status_code = 200
            return _idempotent_or_409(existing, u)
    user = User(source=u.source, source_user_id=u.source_user_id, persona=u.persona)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # 동시 요청이 pre-check 뒤 사이에 같은 키로 먼저 저장했다. 롤백 후 저장된 행을 돌려준다.
        db.rollback()
        if u.source_user_id is None:
            raise                                                # 익명은 UNIQUE 키가 없음 — 다른 원인
        existing = db.scalar(select(User).where(
            User.source == u.source, User.source_user_id == u.source_user_id))
        if existing is None:
            raise                                                # UNIQUE 외의 원인이면 원래대로 전파
        response.status_code = 200
        return _idempotent_or_409(existing, u)
    db.refresh(user)
    return user
