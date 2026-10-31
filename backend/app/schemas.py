from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal["view", "click", "add_to_cart", "purchase", "search"]


class ItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    source: str
    brand_id: int | None
    title: str
    category: str
    price: float | None
    image_path: str | None


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
    # 클라이언트가 생성한 재전송 멱등 키. 네트워크 재시도 · 시뮬레이터 재실행에 쓴다. 생략 시 서버가 UUID4 생성.
    event_id: UUID | None = None
    # 신규 고객의 초기 이력 주입. True이면 A/B · CTR 집계에서 제외.
    is_setup: bool = False


class EventOut(EventIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    event_id: UUID
    created_at: datetime


class UserIn(BaseModel):
    # (source, source_user_id)가 같으면 같은 user를 돌려주는 멱등 생성. source_user_id가 None이면 익명 신규.
    source: str = "sim"
    source_user_id: str | None = None
    persona: str | None = None


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    source: str
    source_user_id: str | None
    persona: str | None


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
