"""지표 집계 · 평가 결과 저장 — 명세 4.6 Continuous Training 지원.

MetricSnapshot CRUD
    POST /metrics/snapshots         오프라인 평가 결과 저장 (재학습 pipeline · 대시보드가 씀)
    GET  /metrics/snapshots         저장된 결과 조회 (model/metric 필터)

CT runner 연결점 (광석 monitoring/runner.py 가 cron 으로 호출)
    GET  /metrics/window            지정 구간 집계 → observe() 가 요구하는 window JSON
    GET  /metrics/log-delta         since_event_id 이후 쌓인 non-setup 이벤트 수 → training 트리거 판단
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Event, MetricSnapshot
from ..schemas import LogDeltaOut, MetricSnapshotIn, MetricSnapshotOut, MetricWindowOut

router = APIRouter(prefix="/metrics", tags=["metrics"])


# ───── MetricSnapshot 저장/조회 ─────

@router.post("/snapshots", response_model=MetricSnapshotOut, status_code=201)
def create_snapshot(s: MetricSnapshotIn, db: Session = Depends(get_db)):
    """오프라인 평가 · 모델 교체 전후 지표 저장. 광석 monitoring 이 finish_training 후 호출.
    자유 label(metric="recall@50" 등) 로 모델/실험/변량 축에 기록된다."""
    row = MetricSnapshot(**s.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/snapshots", response_model=list[MetricSnapshotOut])
def list_snapshots(
    model_version_id: int | None = None,
    experiment_id: int | None = None,
    variant: str | None = None,
    metric: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """저장된 지표 조회. 필터 없이 부르면 최신 100개. 대시보드가 모델별 recall@50 추이를 그리는 데 쓴다."""
    stmt = select(MetricSnapshot).order_by(MetricSnapshot.id.desc()).limit(limit)
    if model_version_id is not None:
        stmt = stmt.where(MetricSnapshot.model_version_id == model_version_id)
    if experiment_id is not None:
        stmt = stmt.where(MetricSnapshot.experiment_id == experiment_id)
    if variant is not None:
        stmt = stmt.where(MetricSnapshot.variant == variant)
    if metric is not None:
        stmt = stmt.where(MetricSnapshot.metric == metric)
    return list(db.scalars(stmt))


# ───── CT runner 연결점 ─────

def _parse_tz_iso(s: str, field: str) -> datetime:
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise HTTPException(422, f"{field}: ISO 8601 형식이어야 함 ({e})")
    if dt.tzinfo is None:
        raise HTTPException(422, f"{field}: 타임존이 명시돼야 함 (runner 가 거절함)")
    return dt


@router.get("/window", response_model=MetricWindowOut)
def metric_window(
    start: str = Query(..., description="ISO 8601 with timezone (예: 2026-11-01T00:00:00+00:00)"),
    end: str = Query(..., description="ISO 8601 with timezone, strictly > start"),
    model_version: str = Query(..., description="runner identity용 라벨 (예: popularity-v1)"),
    source: str = Query(..., description="데이터셋 (amazon · hm · all)"),
    cohort: str = Query("all"),
    metric_definition_version: str = Query("v1"),
    model_version_id: int | None = Query(None, description="A/B 중 모델별 집계용 — Event.model_version_id 필터 (PR #48 리뷰 반영)"),
    db: Session = Depends(get_db),
):
    """지정한 (start, end] 구간에서 views · clicks · evaluated_users · hit_users 를 집계하고
    cumulative_new_logs (window_end 시점까지 전체 non-setup 이벤트 수)와 함께 돌려준다.

    is_setup=True 는 집계 전체에서 제외 (PR #30 규칙 일관). 결과 shape 는 광석 monitoring observe() 가
    기대하는 것과 같아, cron 으로 받아 그대로 넣으면 임계값/트리거 로직이 돌아간다.

    model_version_id 를 주면 그 모델 이벤트만 집계 (A/B 중 모델별 CTR/HitRate 판단).
    """
    t_start_aware = _parse_tz_iso(start, "start")
    t_end_aware = _parse_tz_iso(end, "end")
    if not (t_start_aware < t_end_aware):
        raise HTTPException(422, "start < end 이어야 함")

    # DB 비교용 naive UTC — SQLite 의 tz-aware 비교가 세션 시간대에 따라 달라지지 않도록
    # 모든 DB 에서 같은 결과를 보장한다 (PR #48 리뷰 by 유환).
    t_start = t_start_aware.astimezone(timezone.utc).replace(tzinfo=None)
    t_end = t_end_aware.astimezone(timezone.utc).replace(tzinfo=None)

    non_setup = Event.is_setup == False                                                   # noqa: E712
    model_filter = (Event.model_version_id == model_version_id) if model_version_id is not None else True
    in_window = and_(Event.created_at > t_start, Event.created_at <= t_end, non_setup, model_filter)

    # 데이터셋 필터는 Event 에 source 가 없어서 join 이 필요한데, 지금은 "source=all" 만 정확 지원.
    # 데이터셋별 분리가 필요하면 별도 파라미터로 Event.item_id → Item.source 조인 추가 (후속 PR).
    if source != "all":
        raise HTTPException(422, "현재는 source='all' 만 지원 — 데이터셋별 분리는 후속 PR")

    views = int(db.scalar(select(func.count(Event.id))
                          .where(and_(in_window, Event.type == "view"))) or 0)
    clicks = int(db.scalar(select(func.count(Event.id))
                           .where(and_(in_window, Event.type == "click"))) or 0)

    # evaluated_users = 노출 받은 distinct user, hit_users = 노출 받고 **또** 클릭한 user (교집합).
    # 노출 없이 클릭만 한 유저 (예: 검색결과 클릭) 는 hit_users 에 포함 안 됨 (PR #48 리뷰 by 유환).
    viewers = {r for (r,) in db.execute(
        select(Event.user_id).distinct().where(and_(in_window, Event.type == "view")))}
    clickers = {r for (r,) in db.execute(
        select(Event.user_id).distinct().where(and_(in_window, Event.type == "click")))}
    evaluated_users = len(viewers)
    hit_users = len(viewers & clickers)

    # cumulative_new_logs: window_end 시점까지 비 setup 이벤트 전체 수 (runner 는 단조 증가만 요구).
    # model_version_id 필터는 cumulative 에도 적용 — 그 모델 스트림의 누적 로그로 재학습 임계값 판단.
    cumulative_cond = and_(Event.created_at <= t_end, non_setup, model_filter)
    cumulative = int(db.scalar(select(func.count(Event.id)).where(cumulative_cond)) or 0)

    window_id = f"{t_start_aware.isoformat()}__{t_end_aware.isoformat()}__{model_version}__{source}__{cohort}"
    return MetricWindowOut(
        window_id=window_id,
        # 응답의 window_start/end 는 호출자가 준 timezone-aware ISO 그대로 — runner 가 tz 요구함
        window_start=t_start_aware.isoformat(), window_end=t_end_aware.isoformat(),
        model_version=model_version, source=source, cohort=cohort,
        metric_definition_version=metric_definition_version,
        setup_excluded=True,
        views=views, clicks=clicks,
        evaluated_users=evaluated_users, hit_users=hit_users,
        cumulative_new_logs=cumulative,
    )


@router.get("/log-delta", response_model=LogDeltaOut)
def log_delta(since_event_id: int = Query(0, ge=0), db: Session = Depends(get_db)):
    """since_event_id 초과 ~ max(Event.id) 구간의 non-setup 이벤트 수를 돌려준다.
    runner 가 정책(예: 10000건 이상 쌓이면 재학습) 판단에 쓴다 — 임계값/정책은 runner 책임.

    reset_detected: max_event_id < since_event_id 이면 --reset 재적재 가능성 — runner 가 기준점을
    다시 잡아야 한다 (PR #48 리뷰 반영)."""
    non_setup = Event.is_setup == False                                                   # noqa: E712
    max_id = int(db.scalar(select(func.max(Event.id))) or 0)
    delta = int(db.scalar(select(func.count(Event.id))
                          .where(and_(Event.id > since_event_id, non_setup))) or 0)
    return LogDeltaOut(
        since_event_id=since_event_id,
        max_event_id=max_id,
        delta=delta,
        reset_detected=max_id < since_event_id,
    )
