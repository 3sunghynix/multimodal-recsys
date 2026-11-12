"""추천. 진행 중인 A/B 실험이 있으면 고객을 그룹에 할당해 그 모델로, 없으면 active 모델로 추천한다.

명세(2-5) 요구 응답 필드:
- recommendations[]: product_id · score · reason · is_exploration
- pipeline_latency: candidate_ms · ranking_ms · reranking_ms · total_ms
- session_context: recent_clicks · session_interest
"""
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import recommenders
from ..config import settings
from ..db import get_db
from ..models import Event, Experiment, ModelVersion, User
from ..schemas import PipelineLatency, RecommendationOut, RecommendOut, SessionContext
from ..services import feature_store
from ..services.ab import assign_variant
from ..services.session import session_click_ids, session_interest

router = APIRouter(prefix="/recommend", tags=["recommend"])

# 모델 이름 → 명세 reason 라벨. Multi-Stage Ranking 이 들어오면 아이템별로 더 세밀하게 분기.
_REASON = {"popularity": "popular", "brand_popularity": "brand_match", "ume": "personalized"}
_RECENT_CLICK_LIMIT = 5


@router.get("/{user_id}", response_model=RecommendOut)
def recommend(user_id: int, k: int | None = None,
              session_id: str | None = Query(None, max_length=64, description="클라이언트 세션 id (선택). 주면 그 세션의 클릭으로 세션 반영"),
              db: Session = Depends(get_db)):
    t_start = time.perf_counter()
    if db.get(User, user_id) is None:
        raise HTTPException(404, f"고객 없음: {user_id}")
    k = k or settings.default_k
    exp = db.scalars(select(Experiment).where(Experiment.status == "running").order_by(Experiment.id.desc())).first()
    variant = None
    if exp:
        variant = assign_variant(user_id, exp.id, exp.variants)
        mv = db.get(ModelVersion, variant["model_version_id"])
    else:
        mv = db.scalars(select(ModelVersion).where(ModelVersion.status == "active")).first()
    if mv is None:
        raise HTTPException(409, "active 모델이 없습니다. /models 에서 등록 · 활성화하세요")
    recommender = recommenders.get(mv.name, mv.params)
    recommender.session_id = session_id
    scored = recommender.recommend_scored(db, user_id, k)       # (id, score) 쌍 — 모델이 실제 점수를 돌려주거나 base 가 폴백
    items = [i for i, _ in scored]
    model_ms = round((time.perf_counter() - t_start) * 1000, 3)   # 고객 확인 · 모델 선택 · 추천까지 (세션 컨텍스트 전)

    # recommendations[]: product_id · score · score_kind · reason · is_exploration.
    #   score_kind="model" → recommender.scored_native=True (Popularity · BrandPopularity 는 이벤트 가중치 합계가 실제 점수)
    #   score_kind="rank"  → 모델이 점수를 돌려주지 않아 base 가 순위 기반 placeholder (1.0 → 0.0)
    # 자동 채점·리뷰에서 "모델 점수"로 오해되지 않도록 두 종류를 분리해 노출한다.
    score_kind = "model" if recommender.scored_native else "rank"
    reason = recommender.fallback_reason or _REASON.get(mv.name, mv.name)   # 대체 경로면 그 이유 (예: UME 신규 고객 → popular)
    recommendations = [RecommendationOut(
        product_id=str(item_id),
        score=round(float(score), 4),
        score_kind=score_kind,
        score_basis=recommender.score_basis,                   # 점수 단위 (UME 표준화 · 데이터셋 백분위 · 인기 가중치 합 · 순위)
        reason=reason,
        is_exploration=False,                                  # MAB 미구현 — 전부 exploit
    ) for item_id, score in scored]

    # 세션 컨텍스트: 해당 고객의 최근 클릭 상품 5개 (Feature Store 우선, 비면 DB 폴백).
    # 서비스가 처음 뜨거나 Redis 가 없거나 Redis 가 cold(앞 선 이벤트 없음) 일 때 DB 가 보완.
    recent_clicks = feature_store.get_recent_clicks(user_id, _RECENT_CLICK_LIMIT)
    if not recent_clicks:
        recent = db.scalars(select(Event.item_id).where(
            Event.user_id == user_id, Event.type == "click", Event.item_id.is_not(None)
        ).order_by(Event.created_at.desc(), Event.id.desc()).limit(_RECENT_CLICK_LIMIT)).all()
        recent_clicks = [str(i) for i in recent]
    # 세션 관심: 현재 세션(30분 안) 클릭 상품의 가장 많은 소분류 (세션이 없으면 None)
    interest = session_interest(db, session_click_ids(db, user_id, session_id=session_id))
    # total_ms 는 세션 컨텍스트(최근 클릭 · 세션 관심의 DB · Redis 조회)까지 만든 뒤 — 응답 직전까지의 요청 처리 시간
    total_ms = round((time.perf_counter() - t_start) * 1000, 3)

    # 단계별 시간: 나눠 재는 모델(UME)은 그 값, 아니면 모델 추천 시간 전체를 candidate_ms 에.
    t = recommender.timings
    latency = (PipelineLatency(candidate_ms=round(t["candidate_ms"], 3), ranking_ms=round(t["ranking_ms"], 3),
                               reranking_ms=round(t["reranking_ms"], 3), total_ms=total_ms) if t else
               PipelineLatency(candidate_ms=model_ms, ranking_ms=0.0, reranking_ms=0.0, total_ms=total_ms))

    return RecommendOut(
        user_id=user_id, model=mv.name, model_version_id=mv.id,
        experiment_id=exp.id if exp else None, variant=variant["name"] if variant else None,
        items=items,
        recommendations=recommendations,
        pipeline_latency=latency,
        session_context=SessionContext(
            recent_clicks=recent_clicks, session_interest=interest,
        ),
    )
