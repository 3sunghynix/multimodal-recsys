"""행동 로그 수집. 사용자 화면 · 시뮬레이터가 모두 여기로 보낸다. 재학습 자동화는 이 로그 양을 기준으로 한다.

event_id가 들어오면 재전송 멱등: 같은 내용은 200으로 저장분을 돌려주고, 다른 내용은 409로 거절한다.
동시 요청이 pre-check 뒤 INSERT 로 가는 사이 다른 요청이 같은 event_id 로 먼저 저장하면 UNIQUE 충돌로
500 이 나는 레이스를 IntegrityError 로 잡아 롤백 후 재조회해 같은 멱등 로직(200 · 409) 으로 응답한다.
is_setup=True 이벤트는 초기 이력 적재 전용(시뮬레이터 신규 고객 등) — A/B · CTR 집계 뿐 아니라
공통 인기도(popularity · UME 의 popularity_z) 에서도 제외된다. 개인화 경로(해당 고객의 최근 이력 ·
브랜드 선호) 에는 그대로 포함된다.

이벤트연결규약 (PR #57 ③④):
    · request_id 있으면 recommend_requests 역조회로 model_version_id · experiment_id · variant
      자동 채움 (생략 시) 또는 일치 검증 (명시 시, 틀리면 422)
    · POST /events/batch 로 최대 1,000건 일괄 전송 (건별 결과 + 묶음 commit)
"""
import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Event, Item, RecommendRequest, User
from ..schemas import (EventBatchResult, EventIn, EventOut, EventsBatchIn,
                        EventsBatchOut)
from ..services import feature_store

router = APIRouter(prefix="/events", tags=["events"])

# 재전송 멱등 비교에 쓰는 필드. event_id · 서버 생성 필드(id, created_at)는 제외.
_IDEMPOTENT_FIELDS = ("user_id", "item_id", "type", "query",
                      "model_version_id", "experiment_id", "variant", "is_setup",
                      "session_id", "request_id", "position")


def _idempotent_or_409(existing: Event, incoming: dict) -> Event:
    if any(getattr(existing, f) != incoming[f] for f in _IDEMPOTENT_FIELDS):
        raise HTTPException(409, "같은 event_id로 다른 내용의 이벤트가 이미 저장돼 있습니다")
    return existing


def _apply_request_lookup(incoming: dict, rr: RecommendRequest) -> None:
    """request_id 로 조회한 RecommendRequest 를 기준으로 incoming 을 검증·자동 채움.
    실패 시 HTTPException(422, ...) 전파.

    규칙 (이벤트연결규약 §2.2 + 톡방 결정 5):
      - user_id 불일치 → 422 (보안 이슈 — 다른 유저의 request_id 가로채기 방지)
      - position 명시됐으면 item_ids[position-1] 이 item_id 와 같아야 함
        position 생략 + item_id 있으면 item_ids 에서 찾아 자동 채움 (없으면 422)
      - model_version_id · experiment_id · variant:
          · 생략됨 → 서버 값으로 자동 채움
          · 명시됨 + 서버 기록과 일치 → 그대로
          · 명시됨 + 불일치 → 422 (조용히 덮어쓰지 않음)
    """
    if incoming["user_id"] != rr.user_id:
        raise HTTPException(422, f"request_id 의 user_id ({rr.user_id}) 와 다릅니다: {incoming['user_id']}")

    item_ids: list[int] = list(rr.item_ids or [])
    if incoming.get("position") is not None:
        pos = incoming["position"]
        if pos < 1 or pos > len(item_ids):
            raise HTTPException(422, f"position={pos} 가 request 의 추천 길이 ({len(item_ids)}) 범위 밖")
        if incoming.get("item_id") is not None and item_ids[pos - 1] != incoming["item_id"]:
            raise HTTPException(422, f"position={pos} 의 상품 ({item_ids[pos - 1]}) 과 item_id ({incoming['item_id']}) 불일치")
    elif incoming.get("item_id") is not None:
        # position 생략 → item_id 로 찾아 자동 채움. 요청 목록에 없으면 422.
        if incoming["item_id"] not in item_ids:
            raise HTTPException(422, f"item_id={incoming['item_id']} 가 request 의 추천 목록에 없음")
        incoming["position"] = item_ids.index(incoming["item_id"]) + 1

    for field in ("model_version_id", "experiment_id", "variant"):
        server_value = getattr(rr, field)
        client_value = incoming.get(field)
        if client_value is None:
            incoming[field] = server_value           # 생략 → 자동 채움
        elif server_value != client_value:
            raise HTTPException(
                422,
                f"{field} 가 request 서버 기록 ({server_value}) 과 다릅니다: {client_value}",
            )


def _lookup_requests(db: Session, request_ids: set[_uuid.UUID]) -> dict[_uuid.UUID, RecommendRequest]:
    """배치 전용: distinct request_id 들을 한 번에 fetch 해 N+1 방지."""
    if not request_ids:
        return {}
    rows = db.scalars(select(RecommendRequest).where(RecommendRequest.request_id.in_(request_ids))).all()
    return {r.request_id: r for r in rows}


def _validate_user_item(db: Session, incoming: dict) -> None:
    if db.get(User, incoming["user_id"]) is None:
        raise HTTPException(404, f"고객 없음: {incoming['user_id']}")
    if incoming["type"] != "search" and (incoming.get("item_id") is None or db.get(Item, incoming["item_id"]) is None):
        raise HTTPException(422, "search 외 이벤트는 존재하는 item_id가 필요합니다")


def _build_event(incoming: dict) -> Event:
    payload = incoming.copy()
    if payload.get("event_id") is None:
        payload.pop("event_id", None)
    return Event(**payload)


@router.post("", response_model=EventOut, status_code=201)
def log_event(e: EventIn, response: Response, db: Session = Depends(get_db)):
    incoming = e.model_dump()

    # request_id 있으면 역조회 + 자동 채움을 먼저 — 재전송 시 incoming 도 같은 방식으로 채워지므로
    # 저장된 (자동 채움된) 값과 멱등 비교가 통과한다 (PR #59 리뷰 by 유환 — 자동채움 후 재전송 409 버그).
    if e.request_id is not None:
        rr = db.get(RecommendRequest, e.request_id)
        if rr is None:
            raise HTTPException(422, f"request_id={e.request_id} 가 recommend_requests 에 없음")
        _apply_request_lookup(incoming, rr)

    # 그 다음 멱등 처리: 저장된 이벤트가 있으면 요청 유효성 검사 없이 그 결과를 돌려준다.
    if e.event_id is not None:
        existing = db.scalar(select(Event).where(Event.event_id == e.event_id))
        if existing is not None:
            response.status_code = 200
            return _idempotent_or_409(existing, incoming)

    _validate_user_item(db, incoming)
    ev = _build_event(incoming)
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
        feature_store.record_event(user_id=e.user_id, event_type=e.type, item_id=e.item_id, session_id=e.session_id)
    return ev


@router.post("/batch", response_model=EventsBatchOut)
def log_events_batch(body: EventsBatchIn, db: Session = Depends(get_db)):
    """최대 1,000건 일괄 전송. 건별 **savepoint** 로 저장해 한 건이 실패해도 나머지는 그대로
    저장됨 (톡방 결정 2 · PR #59 리뷰 by 유환). 같은 event_id 로 저장된 이벤트가 있으면
    재전송 멱등 (내용 같으면 duplicate, 다르면 error 409). Feature Store 는 is_setup 제외해 묶음 반영.

    N+1 방지: 배치 내 distinct request_id 는 한 번에 fetch, 저장돼 있는 event_id 들도 한 번에 fetch.
    배치 안에서 같은 event_id 가 중복으로 들어오면 두 번째부터는 duplicate / error(409) 로 처리
    (전체 롤백 없음; PR #59 리뷰).

    refresh() 는 생략 — db.flush() 가 ev.id 를 세팅하므로 응답 시간 절감 (1,000건 round-trip 1,000회 제거).
    """
    results: list[EventBatchResult] = []
    created_items: list[tuple[Event, EventIn]] = []    # 성공 저장 — Feature Store 반영용

    # 배치 내 distinct request_id · event_id 를 한 번에 fetch (I/O 최소화)
    req_ids = {e.request_id for e in body.events if e.request_id is not None}
    ev_ids = {e.event_id for e in body.events if e.event_id is not None}
    req_map = _lookup_requests(db, req_ids)
    existing_map: dict[_uuid.UUID, Event] = {}
    if ev_ids:
        existing_rows = db.scalars(select(Event).where(Event.event_id.in_(ev_ids))).all()
        existing_map = {r.event_id: r for r in existing_rows}

    for idx, e in enumerate(body.events):
        incoming = e.model_dump()

        # 1) request_id 있으면 역조회 + 자동 채움을 먼저 (PR #59 리뷰 — 자동채움 후 재전송 멱등 보전)
        if e.request_id is not None:
            rr = req_map.get(e.request_id)
            if rr is None:
                results.append(EventBatchResult(index=idx, status="error", code=422,
                                                detail=f"request_id={e.request_id} 가 recommend_requests 에 없음"))
                continue
            try:
                _apply_request_lookup(incoming, rr)
            except HTTPException as exc:
                results.append(EventBatchResult(index=idx, status="error",
                                                code=exc.status_code, detail=str(exc.detail)))
                continue

        # 2) 저장돼 있는 event_id 멱등 체크 (배치 사전 fetch + 이 루프에서 성공 저장한 것 포함)
        if e.event_id is not None and e.event_id in existing_map:
            existing = existing_map[e.event_id]
            if any(getattr(existing, f) != incoming[f] for f in _IDEMPOTENT_FIELDS):
                results.append(EventBatchResult(index=idx, status="error", code=409,
                                                detail="같은 event_id 로 다른 내용의 이벤트가 이미 저장돼 있음"))
            else:
                results.append(EventBatchResult(index=idx, status="duplicate",
                                                id=existing.id, event_id=existing.event_id))
            continue

        # 3) User / Item 검증
        try:
            _validate_user_item(db, incoming)
        except HTTPException as exc:
            results.append(EventBatchResult(index=idx, status="error",
                                            code=exc.status_code, detail=str(exc.detail)))
            continue

        # 4) savepoint 안에서 저장 — IntegrityError (동시 요청이 먼저 저장한 경우) 는 이 savepoint 만 롤백.
        ev = _build_event(incoming)
        try:
            with db.begin_nested():
                db.add(ev)
                db.flush()                                # IntegrityError 를 이 시점에서 발생시킴 (id 도 세팅)
        except IntegrityError:
            # event_id UNIQUE 충돌 — 동시 요청이 먼저 저장. 서버 생성 uuid4 는 사실상 충돌 불가.
            if e.event_id is None:
                results.append(EventBatchResult(index=idx, status="error", code=500,
                                                detail="저장 실패 (UNIQUE 외의 IntegrityError)"))
                continue
            existing = db.scalar(select(Event).where(Event.event_id == e.event_id))
            if existing is None:
                results.append(EventBatchResult(index=idx, status="error", code=500,
                                                detail="저장 실패 — 재조회도 실패"))
            elif any(getattr(existing, f) != incoming[f] for f in _IDEMPOTENT_FIELDS):
                results.append(EventBatchResult(index=idx, status="error", code=409,
                                                detail="같은 event_id 로 다른 내용의 이벤트가 이미 저장돼 있음"))
            else:
                results.append(EventBatchResult(index=idx, status="duplicate",
                                                id=existing.id, event_id=existing.event_id))
            continue

        # 5) 성공 — refresh 불필요 (flush 로 id · event_id 세팅됨)
        results.append(EventBatchResult(index=idx, status="created",
                                        id=ev.id, event_id=ev.event_id))
        # 배치 내 후속 행이 같은 event_id 로 들어오면 멱등 검출 가능하게 캐시 갱신
        existing_map[ev.event_id] = ev
        created_items.append((ev, e))

    # 최종 commit — 모든 savepoint 가 이미 flush 됐으므로 트랜잭션만 마무리
    db.commit()

    # Feature Store 묶음 반영 (is_setup 제외 규칙 유지)
    for ev, orig in created_items:
        if not orig.is_setup:
            feature_store.record_event(
                user_id=orig.user_id, event_type=orig.type,
                item_id=orig.item_id, session_id=orig.session_id,
            )

    return EventsBatchOut(
        results=results,
        created=sum(1 for r in results if r.status == "created"),
        duplicate=sum(1 for r in results if r.status == "duplicate"),
        error=sum(1 for r in results if r.status == "error"),
    )


@router.get("/count")
def count_events(db: Session = Depends(get_db)):
    rows = db.execute(select(Event.type, func.count()).group_by(Event.type)).all()
    return {t: n for t, n in rows}
