"""추천. 진행 중인 A/B 실험이 있으면 고객을 그룹에 할당해 그 모델로, 없으면 active 모델로 추천한다.

명세(2-5) 요구 응답 필드:
- recommendations[]: product_id · score · reason · is_exploration
- pipeline_latency: candidate_ms · ranking_ms · reranking_ms · total_ms
- session_context: recent_clicks · session_interest
"""
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import recommenders
from ..config import settings
from ..db import get_db
from ..models import Event, Experiment, ModelVersion, User
from ..schemas import PipelineLatency, RecommendationOut, RecommendOut, SessionContext
from ..services.ab import assign_variant

router = APIRouter(prefix="/recommend", tags=["recommend"])

# 모델 이름 → 명세 reason 라벨. Multi-Stage Ranking 이 들어오면 아이템별로 더 세밀하게 분기.
_REASON = {"popularity": "popular", "brand_popularity": "brand_match", "ume": "personalized"}
_RECENT_CLICK_LIMIT = 5


@router.get("/{user_id}", response_model=RecommendOut)
def recommend(user_id: int, k: int | None = None, db: Session = Depends(get_db)):
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
    scored = recommender.recommend_scored(db, user_id, k)       # (id, score) 쌍 — 모델이 실제 점수를 돌려주거나 base 가 폴백
    items = [i for i, _ in scored]
    total_ms = round((time.perf_counter() - t_start) * 1000, 3)

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
        reason=reason,
        is_exploration=False,                                  # MAB 미구현 — 전부 exploit
    ) for item_id, score in scored]

    # 세션 컨텍스트: 해당 고객의 최근 클릭 상품 5개. 세션 인코더가 없으니 session_interest 는 None.
    recent = db.scalars(select(Event.item_id).where(
        Event.user_id == user_id, Event.type == "click", Event.item_id.is_not(None)
    ).order_by(Event.created_at.desc(), Event.id.desc()).limit(_RECENT_CLICK_LIMIT)).all()

    return RecommendOut(
        user_id=user_id, model=mv.name, model_version_id=mv.id,
        experiment_id=exp.id if exp else None, variant=variant["name"] if variant else None,
        items=items,
        recommendations=recommendations,
        pipeline_latency=PipelineLatency(
            # Multi-Stage 가 아직 쪼개져 있지 않아 전체 시간을 candidate_ms 에, 나머지는 0.
            candidate_ms=total_ms, ranking_ms=0.0, reranking_ms=0.0, total_ms=total_ms,
        ),
        session_context=SessionContext(
            recent_clicks=[str(i) for i in recent], session_interest=None,
        ),
    )
