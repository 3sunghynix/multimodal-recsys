"""DB 테이블. 상세 설명은 docs/project/데이터스키마.md.

데이터셋(Amazon · H&M 등)은 하나의 상품 · 이벤트 테이블에 합치고 brand와 source 열로 구분한다.
"""
import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from .config import settings
from .db import Base
from .types import EmbeddingVector

# 콘텐츠 임베딩(검색용)의 차원. 운영 인코더가 OpenAI CLIP ViT-B/32라서 기본값은 settings.embedding_dim = 512.
# 다른 인코더(예: 논문 PCA 256 + L2)로 교체 시 settings에서 embedding_dim 함께 변경한다.
EMBEDDING_DIM = settings.embedding_dim

EVENT_TYPES = ("view", "click", "add_to_cart", "purchase", "search")


class Brand(Base):
    __tablename__ = "brands"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)


class Item(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32))                  # amazon | hm | ...
    source_item_id: Mapped[str] = mapped_column(String(64), index=True)
    brand_id: Mapped[int | None] = mapped_column(ForeignKey("brands.id"), index=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(200), default="")
    price: Mapped[float | None] = mapped_column(Float)
    image_path: Mapped[str | None] = mapped_column(String(500))      # 로컬 경로 또는 URL
    # 콘텐츠 임베딩(검색용). 상품마다 고정값이며 인코더가 바뀔 때만 재계산한다.
    # 인코더 산출물(MiniLM·Marqo·CLIP 등 L2 정규화 완료 벡터)을 그대로 저장한다 — 차원은 위 EMBEDDING_DIM.
    # Two-Tower 학습 결과 상품 벡터(모델마다 달라짐, A/B로 모델 두 개 동시 서빙 가능)는
    # items에 넣지 않고 item_vectors(model_version_id, item_id, vector) 같은 별도 테이블로 분리할 예정.
    # 쿼리 이미지에도 같은 전처리(인코더 + 필요시 PCA)를 적용해야 하므로 PCA 변환 행렬도 산출물로 함께 저장한다.
    # PostgreSQL에서는 pgvector로 코사인 ANN 검색, SQLite(테스트)에서는 JSON으로 보관만 한다.
    text_embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(EMBEDDING_DIM), nullable=True)
    image_embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(EMBEDDING_DIM), nullable=True)


class User(Base):
    __tablename__ = "users"
    # (source, source_user_id) 는 "같은 외부 고객 재등록 시 같은 user를 돌려준다"는 계약을 위해 UNIQUE.
    # source_user_id가 NULL이면 익명 생성 — SQLite · PostgreSQL 모두 UNIQUE에서 NULL을 서로 다른 값으로 취급해서 충돌 없음.
    __table_args__ = (UniqueConstraint("source", "source_user_id", name="uq_users_source_source_user_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32), default="sim")   # 실제 데이터셋 고객 또는 sim(시뮬레이터)
    source_user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    persona: Mapped[str | None] = mapped_column(String(64))          # 시뮬레이터 페르소나 (6종 정의 후)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    # 재전송 멱등을 위해 클라이언트가 제공하는 UUID. 없으면 서버가 즉시 생성한다 (기존 호출 하위호환).
    # UNIQUE: 같은 event_id로 같은 내용이면 200 멱등, 다른 내용이면 409.
    event_id: Mapped[uuid.UUID] = mapped_column(Uuid, default=uuid.uuid4, unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    item_id: Mapped[int | None] = mapped_column(ForeignKey("items.id"), index=True)  # search는 상품 없음
    type: Mapped[str] = mapped_column(String(16))                    # EVENT_TYPES
    query: Mapped[str | None] = mapped_column(String(500))
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))  # 어떤 모델의 추천에서 왔는가
    experiment_id: Mapped[int | None] = mapped_column(ForeignKey("experiments.id"))
    variant: Mapped[str | None] = mapped_column(String(32))
    # 초기 이력 적재(시뮬레이터 신규 고객의 train 이력 주입)는 is_setup=True로 저장하고,
    # A/B 집계(CTR 등)에서 제외한다. 실험 전환 · 재학습 트리거는 is_setup=False만 센다.
    is_setup: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class ModelVersion(Base):
    """추천 모델 레지스트리. 관리자 대시보드의 '모델 교체' 버튼이 status를 바꾼다."""
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64))                    # recommenders 레지스트리의 이름
    version: Mapped[str] = mapped_column(String(32), default="v1")
    status: Mapped[str] = mapped_column(String(16), default="candidate")   # active | candidate | retired
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Experiment(Base):
    """A/B 테스트. variants: [{"name": "A", "model_version_id": 1, "weight": 0.5}, ...]"""
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(16), default="running")    # running | stopped
    variants: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MetricSnapshot(Base):
    """평가 결과 기록 (오프라인 평가 · A/B 집계 · 재학습 전후). 대시보드 그래프의 데이터."""
    __tablename__ = "metric_snapshots"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    experiment_id: Mapped[int | None] = mapped_column(ForeignKey("experiments.id"))
    variant: Mapped[str | None] = mapped_column(String(32))
    metric: Mapped[str] = mapped_column(String(64))                  # recall@20, ctr, brand_diversity ...
    value: Mapped[float] = mapped_column(Float)
    n: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
