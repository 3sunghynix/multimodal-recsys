"""행동 로그 수집. 사용자 화면 · 시뮬레이터가 모두 여기로 보낸다. 재학습 자동화는 이 로그 양을 기준으로 한다.

event_id가 들어오면 재전송 멱등: 같은 내용은 200으로 저장분을 돌려주고, 다른 내용은 409로 거절한다.
is_setup=True 이벤트는 초기 이력 적재 전용(시뮬레이터 신규 고객 등) — A/B · CTR 집계에서 제외된다.
"""
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Event, Item, User
from ..schemas import EventIn, EventOut

router = APIRouter(prefix="/events", tags=["events"])

# 재전송 멱등 비교에 쓰는 필드. event_id · 서버 생성 필드(id, created_at)는 제외.
_IDEMPOTENT_FIELDS = ("user_id", "item_id", "type", "query",
                      "model_version_id", "experiment_id", "variant", "is_setup")


@router.post("", response_model=EventOut, status_code=201)
def log_event(e: EventIn, response: Response, db: Session = Depends(get_db)):
    # 멱등 처리를 먼저: 저장된 이벤트가 있으면 요청 유효성 검사 없이 그 결과를 돌려준다.
    if e.event_id is not None:
        existing = db.scalar(select(Event).where(Event.event_id == e.event_id))
        if existing is not None:
            incoming = e.model_dump()
            if any(getattr(existing, f) != incoming[f] for f in _IDEMPOTENT_FIELDS):
                raise HTTPException(409, "같은 event_id로 다른 내용의 이벤트가 이미 저장돼 있습니다")
            response.status_code = 200
            return existing
    if db.get(User, e.user_id) is None:
        raise HTTPException(404, f"고객 없음: {e.user_id}")
    if e.type != "search" and (e.item_id is None or db.get(Item, e.item_id) is None):
        raise HTTPException(422, "search 외 이벤트는 존재하는 item_id가 필요합니다")
    # event_id가 None이면 모델의 default=uuid4가 채운다.
    payload = e.model_dump(exclude_none=False)
    if payload["event_id"] is None:
        payload.pop("event_id")
    ev = Event(**payload)
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev


@router.get("/count")
def count_events(db: Session = Depends(get_db)):
    rows = db.execute(select(Event.type, func.count()).group_by(Event.type)).all()
    return {t: n for t, n in rows}
