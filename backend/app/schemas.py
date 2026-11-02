from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal["view", "click", "add_to_cart", "purchase", "search"]


class ItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    source: str
    brand_id: int | None
    brand_name: str | None = None          # brands.name JOIN 결과 — 카드에 "BRAND #10" 대신 실제 브랜드명 노출
    title: str
    category: str
    price: float | None
    image_path: str | None


class ItemDetailOut(ItemOut):
    """상품 상세 페이지용. 카드용 ItemOut 에 설명 · 외부 상품 id 를 더한다.
    배치 조회(GET /items)는 응답 가벼운 ItemOut 유지, 단건 조회(GET /items/{id})만 이 스키마로 바꾼다."""
    description: str = ""
    source_item_id: str


class SearchHit(ItemOut):
    score: float                      # 결합 점수 (merge 방식에 따라 척도가 다르다. 순서만 의미 있음)


class SearchOut(BaseModel):
    query: str | None                 # 받은 검색어
    query_en: str | None              # 인코더에 넣은 영어 검색어 (한국어면 사전 번역 결과)
    untranslated: list[str]           # 사전에 없어 번역하지 못한 한국어 단어 (화면 안내용)
    weights: dict[str, float]         # text (MiniLM → 상품 텍스트) · text_image (Marqo 텍스트 → 상품 이미지) · image (업로드 이미지)
    merge: str
    items: list[SearchHit]


class EventIn(BaseModel):
    user_id: int
    type: EventType
    item_id: int | None = None
    query: str | None = None
    model_version_id: int | None = None
    experiment_id: int | None = None
    variant: str | None = None


class EventOut(EventIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime


class RecommendOut(BaseModel):
    user_id: int
    model: str
    model_version_id: int | None
    experiment_id: int | None = None
    variant: str | None = None
    items: list[int]


class ModelVersionIn(BaseModel):
    name: str
    version: str = "v1"
    params: dict = Field(default_factory=dict)


class ModelVersionOut(ModelVersionIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: str


class Variant(BaseModel):
    name: str
    model_version_id: int
    weight: float = Field(gt=0)


class ExperimentIn(BaseModel):
    name: str
    variants: list[Variant] = Field(min_length=2)


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    status: str
    variants: list[Variant]


class VariantMetrics(BaseModel):
    variant: str
    users: int
    impressions: int
    clicks: int
    purchases: int
    ctr: float
