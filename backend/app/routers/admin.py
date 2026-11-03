"""관리자 기능: 모델 레지스트리(등록 · 교체 버튼), A/B 실험(생성 · 중지 · 그룹별 지표)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import recommenders
from ..db import get_db
from ..models import Event, Experiment, ModelVersion
from ..schemas import ExperimentIn, ExperimentOut, ModelVersionIn, ModelVersionOut, VariantMetrics

router = APIRouter(tags=["admin"])


@router.get("/models", response_model=list[ModelVersionOut])
def list_models(db: Session = Depends(get_db)):
    return db.scalars(select(ModelVersion).order_by(ModelVersion.id)).all()


@router.get("/models/available")
def available_models():
    return sorted(recommenders.REGISTRY)


@router.post("/models", response_model=ModelVersionOut, status_code=201)
def create_model(m: ModelVersionIn, db: Session = Depends(get_db)):
    if m.name not in recommenders.REGISTRY:
        raise HTTPException(422, f"등록되지 않은 모델: {m.name} (가능: {sorted(recommenders.REGISTRY)})")
    mv = ModelVersion(**m.model_dump())
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return mv


@router.post("/models/{model_id}/activate", response_model=ModelVersionOut)
def activate_model(model_id: int, db: Session = Depends(get_db)):
    """대시보드 '이 모델로 교체' 버튼. 기존 active는 candidate로 내린다."""
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise HTTPException(404, f"모델 없음: {model_id}")
    for other in db.scalars(select(ModelVersion).where(ModelVersion.status == "active")):
        other.status = "candidate"
    mv.status = "active"
    db.commit()
    db.refresh(mv)
    return mv


@router.post("/experiments", response_model=ExperimentOut, status_code=201)
def create_experiment(e: ExperimentIn, db: Session = Depends(get_db)):
    for v in e.variants:
        if db.get(ModelVersion, v.model_version_id) is None:
            raise HTTPException(422, f"모델 없음: {v.model_version_id}")
    for running in db.scalars(select(Experiment).where(Experiment.status == "running")):
        running.status = "stopped"           # 동시에 하나만 진행
    exp = Experiment(name=e.name, variants=[v.model_dump() for v in e.variants])
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


@router.get("/experiments", response_model=list[ExperimentOut])
def list_experiments(db: Session = Depends(get_db)):
    return db.scalars(select(Experiment).order_by(Experiment.id.desc())).all()


@router.post("/experiments/{exp_id}/stop", response_model=ExperimentOut)
def stop_experiment(exp_id: int, db: Session = Depends(get_db)):
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404, f"실험 없음: {exp_id}")
    exp.status = "stopped"
    db.commit()
    db.refresh(exp)
    return exp


@router.get("/experiments/{exp_id}/metrics", response_model=list[VariantMetrics])
def experiment_metrics(exp_id: int, db: Session = Depends(get_db)):
    """그룹별 노출(view) · 클릭 · 구매 · CTR. 통계 검정 · 지표 하락 알림은 평가 모듈(김광석)에서 추가.

    is_setup=True 이벤트(신규 고객 초기 이력 주입)는 집계에서 제외한다.
    """
    if db.get(Experiment, exp_id) is None:
        raise HTTPException(404, f"실험 없음: {exp_id}")
    base = (Event.experiment_id == exp_id) & (Event.is_setup == False)   # noqa: E712 (SQLAlchemy는 == False 가 올바른 비교)
    rows = db.execute(select(Event.variant, Event.type, func.count())
                      .where(base).group_by(Event.variant, Event.type)).all()
    agg = {}
    for variant, typ, n in rows:
        agg.setdefault(variant, {})[typ] = n
    users = dict(db.execute(select(Event.variant, func.count(func.distinct(Event.user_id)))
                            .where(base).group_by(Event.variant)).all())
    out = []
    for variant, c in sorted(agg.items(), key=lambda x: str(x[0])):
        imp, clk = c.get("view", 0), c.get("click", 0)
        out.append(VariantMetrics(variant=str(variant), users=users.get(variant, 0), impressions=imp, clicks=clk,
                                  purchases=c.get("purchase", 0), ctr=clk / imp if imp else 0.0))
    return out
