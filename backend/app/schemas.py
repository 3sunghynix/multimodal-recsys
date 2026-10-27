from datetime import datetime
from typing import Literal

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
