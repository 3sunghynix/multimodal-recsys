"""행동 로그 수집. 사용자 화면 · 시뮬레이터가 모두 여기로 보낸다. 재학습 자동화는 이 로그 양을 기준으로 한다.

event_id가 들어오면 재전송 멱등: 같은 내용은 200으로 저장분을 돌려주고, 다른 내용은 409로 거절한다.
동시 요청이 pre-check 뒤 INSERT 로 가는 사이 다른 요청이 같은 event_id 로 먼저 저장하면 UNIQUE 충돌로
500 이 나는 레이스를 IntegrityError 로 잡아 롤백 후 재조회해 같은 멱등 로직(200 · 409) 으로 응답한다.
is_setup=True 이벤트는 초기 이력 적재 전용(시뮬레이터 신규 고객 등) — A/B · CTR 집계 뿐 아니라
공통 인기도(popularity · UME 의 popularity_z) 에서도 제외된다. 개인화 경로(해당 고객의 최근 이력 ·
브랜드 선호) 에는 그대로 포함된다.
"""
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Event, Item, User
from ..schemas import EventIn, EventOut
from ..services import feature_store

router = APIRouter(prefix="/events", tags=["events"])

# 재전송 멱등 비교에 쓰는 필드. event_id · 서버 생성 필드(id, created_at)는 제외.
_IDEMPOTENT_FIELDS = ("user_id", "item_id", "type", "query",
                      "model_version_id", "experiment_id", "variant", "is_setup")


def _idempotent_or_409(existing: Event, incoming: dict) -> Event:
    if any(getattr(existing, f) != incoming[f] for f in _IDEMPOTENT_FIELDS):
        raise HTTPException(409, "같은 event_id로 다른 내용의 이벤트가 이미 저장돼 있습니다")
    return existing


@router.post("", response_model=EventOut, status_code=201)
def log_event(e: EventIn, response: Response, db: Session = Depends(get_db)):
    incoming = e.model_dump()
    # 멱등 처리를 먼저: 저장된 이벤트가 있으면 요청 유효성 검사 없이 그 결과를 돌려준다.
    if e.event_id is not None:
        existing = db.scalar(select(Event).where(Event.event_id == e.event_id))
        if existing is not None:
            response.status_code = 200
            return _idempotent_or_409(existing, incoming)
    if db.get(User, e.user_id) is None:
        raise HTTPException(404, f"고객 없음: {e.user_id}")
    if e.type != "search" and (e.item_id is None or db.get(Item, e.item_id) is None):
        raise HTTPException(422, "search 외 이벤트는 존재하는 item_id가 필요합니다")
    # event_id가 None이면 모델의 default=uuid4가 채운다.
    payload = incoming.copy()
    if payload["event_id"] is None:
        payload.pop("event_id")
    ev = Event(**payload)
    db.add(ev)
    try:
        db.commit()
    except IntegrityError:
        # 동시 요청이 같은 event_id 로 먼저 저장했다. 롤백 후 저장된 행을 돌려준다.
        db.rollback()
        if e.event_id is None:
            raise                                                # 서버 생성 uuid4 는 충돌 가능성 사실상 0 — 다른 원인
        existing = db.scalar(select(Event).where(Event.event_id == e.event_id))
        if existing is None:
            raise                                                # UNIQUE 외의 원인이면 원래대로 전파
        response.status_code = 200
        return _idempotent_or_409(existing, incoming)
    db.refresh(ev)
    # Feature Store 반영 (명세 4.4). is_setup=True 는 집계 제외 규칙에 따라 Redis 에도 쓰지 않는다
    # — 초기 이력 적재가 "최근 활동" 으로 오해되지 않게. record_event 는 Redis 통신 실패 시 swallow.
    if not e.is_setup:
        feature_store.record_event(user_id=e.user_id, event_type=e.type, item_id=e.item_id)
    return ev


@router.get("/count")
def count_events(db: Session = Depends(get_db)):
    rows = db.execute(select(Event.type, func.count()).group_by(Event.type)).all()
    return {t: n for t, n in rows}
