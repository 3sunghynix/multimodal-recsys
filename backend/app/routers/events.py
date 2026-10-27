"""행동 로그 수집. 사용자 화면 · 시뮬레이터가 모두 여기로 보낸다. 재학습 자동화는 이 로그 양을 기준으로 한다."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Event, Item, User
from ..schemas import EventIn, EventOut

router = APIRouter(prefix="/events", tags=["events"])


@router.post("", response_model=EventOut, status_code=201)
def log_event(e: EventIn, db: Session = Depends(get_db)):
    if db.get(User, e.user_id) is None:
        raise HTTPException(404, f"고객 없음: {e.user_id}")
    if e.type != "search" and (e.item_id is None or db.get(Item, e.item_id) is None):
        raise HTTPException(422, "search 외 이벤트는 존재하는 item_id가 필요합니다")
    ev = Event(**e.model_dump())
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev


@router.get("/count")
def count_events(db: Session = Depends(get_db)):
    rows = db.execute(select(Event.type, func.count()).group_by(Event.type)).all()
    return {t: n for t, n in rows}
