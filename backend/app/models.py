"""DB 테이블. 상세 설명은 docs/project/데이터스키마.md.

데이터셋(Amazon · H&M 등)은 하나의 상품 · 이벤트 테이블에 합치고 brand와 source 열로 구분한다.
"""
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base
from .types import EmbeddingVector

EMBEDDING_DIM = 256  # 논문 파이프라인(PCA 256 + L2) 기본값과 일치

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
    # 멀티모달 임베딩. 논문의 Two-Tower + PCA 256 + L2 산출물을 저장한다.
    # PostgreSQL에서는 pgvector로 코사인 ANN 검색, SQLite(테스트)에서는 JSON으로 보관만 한다.
    text_embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(EMBEDDING_DIM), nullable=True)
    image_embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(EMBEDDING_DIM), nullable=True)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32), default="sim")   # 실제 데이터셋 고객 또는 sim(시뮬레이터)
    source_user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    persona: Mapped[str | None] = mapped_column(String(64))          # 시뮬레이터 페르소나 (6종 정의 후)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    item_id: Mapped[int | None] = mapped_column(ForeignKey("items.id"), index=True)  # search는 상품 없음
    type: Mapped[str] = mapped_column(String(16))                    # EVENT_TYPES
    query: Mapped[str | None] = mapped_column(String(500))
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))  # 어떤 모델의 추천에서 왔는가
    experiment_id: Mapped[int | None] = mapped_column(ForeignKey("experiments.id"))
    variant: Mapped[str | None] = mapped_column(String(32))
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
