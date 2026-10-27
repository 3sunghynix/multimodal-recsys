"""추천. 진행 중인 A/B 실험이 있으면 고객을 그룹에 할당해 그 모델로, 없으면 active 모델로 추천한다."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import recommenders
from ..config import settings
from ..db import get_db
from ..models import Experiment, ModelVersion, User
from ..schemas import RecommendOut
from ..services.ab import assign_variant

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.get("/{user_id}", response_model=RecommendOut)
def recommend(user_id: int, k: int | None = None, db: Session = Depends(get_db)):
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
    return RecommendOut(user_id=user_id, model=mv.name, model_version_id=mv.id,
                        experiment_id=exp.id if exp else None, variant=variant["name"] if variant else None,
                        items=items)
