"""검색용 상품 임베딩 메모리 색인.

데이터셋별 결합(per_source_z · per_source_rank)은 질의마다 그 데이터셋 전체 상품의 점수 분포가 필요해서
pgvector 상위 k 질의로는 계산할 수 없다. 상품 임베딩을 메모리(numpy)에 올려 전체 점수를 한 번에 계산한다.
규모: 상품 10만 × 256차원 × 2(텍스트 · 이미지) × 4바이트 ≈ 200MB, 질의 1건 수 ms.

상품이 추가 · 삭제되면 (DB 주소, 상품 수, 최대 id)가 바뀌므로 다음 요청에서 다시 읽는다.
"""
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Item


@dataclass
class CatalogIndex:
    ids: np.ndarray          # (N,) items.id
    sources: np.ndarray      # (N,) source 문자열
    brands: np.ndarray       # (N,) brand_id (없으면 −1)
    text: np.ndarray         # (N, d) text_embedding
    image: np.ndarray        # (N, d) image_embedding (없으면 0 벡터)

    def __len__(self):
        return len(self.ids)


_cache: dict = {"key": None, "index": None}


def get_index(db: Session) -> CatalogIndex:
    has_text = Item.text_embedding.is_not(None)
    n, max_id = db.execute(select(func.count(Item.id), func.max(Item.id)).where(has_text)).one()
    key = (str(db.bind.url), n, max_id)
    if _cache["key"] != key:
        _cache["index"] = _build(db, has_text)
        _cache["key"] = key
    return _cache["index"]


def _build(db: Session, has_text) -> CatalogIndex:
    rows = db.execute(select(Item.id, Item.source, Item.brand_id, Item.text_embedding, Item.image_embedding)
                      .where(has_text).order_by(Item.id)).all()
    if not rows:
        z = np.zeros((0, 0), np.float32)
        return CatalogIndex(np.zeros(0, np.int64), np.zeros(0, object), np.zeros(0, np.int64), z, z)
    text = np.asarray([r.text_embedding for r in rows], np.float32)
    image = np.zeros_like(text)
    for i, r in enumerate(rows):
        if r.image_embedding is not None:
            image[i] = r.image_embedding
    return CatalogIndex(ids=np.array([r.id for r in rows], np.int64),
                        sources=np.array([r.source for r in rows], object),
                        brands=np.array([r.brand_id if r.brand_id is not None else -1 for r in rows], np.int64),
                        text=text, image=image)
