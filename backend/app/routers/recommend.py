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
    items = recommenders.get(mv.name, mv.params).recommend(db, user_id, k)
    total_ms = round((time.perf_counter() - t_start) * 1000, 3)

    # 명세 recommendations[]: product_id/name/score/reason/is_exploration.
    # 지금은 랭킹 점수를 모델이 외부로 돌려주지 않아서 순위 기반 placeholder (1.0 → 0.0 감소) 로 노출.
    # Ranking 모델(DeepFM) 들어오면 실제 CTR/CVR 예측값으로 대체.
    n = len(items)
    reason = _REASON.get(mv.name, mv.name)
    recommendations = [RecommendationOut(
        product_id=str(item_id),
        score=round(1 - (idx / max(n, 1)), 4),
        reason=reason,
        is_exploration=False,                                  # MAB 미구현 — 전부 exploit
    ) for idx, item_id in enumerate(items)]

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
