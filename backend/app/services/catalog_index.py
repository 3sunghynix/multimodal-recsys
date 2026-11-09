"""검색용 상품 임베딩 메모리 색인.

데이터셋별 결합(per_source_z · per_source_rank)은 질의마다 그 데이터셋 전체 상품의 점수 분포가 필요해서
pgvector 상위 k 질의로는 계산할 수 없다. 상품 임베딩을 메모리(numpy)에 올려 전체 점수를 한 번에 계산한다.
규모: 상품 10만 × 256차원 × 2(텍스트 · 이미지) × 4바이트 ≈ 200MB, 질의 1건 수 ms.

상품이 추가되거나 재적재되면 카탈로그 지문(catalog_signature)이 바뀌므로 다음 요청에서 다시 읽는다.
"""
import hashlib
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
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


def catalog_signature(db: Session):
    """가장 큰 id 상품 한 행으로 만든 카탈로그 지문: (id, 데이터셋, 원본 상품 id, 텍스트 임베딩 해시).
    상품은 추가만 되므로 보통은 id만으로 충분하지만, --reset 재적재는 id를 1부터 다시 매겨 최대 id가 예전과 같을 수 있다.
    원본 id · 임베딩까지 보면 다른 데이터 · 다른 임베딩 버전으로 재적재한 경우도 잡힌다.
    기본 키 역순 한 행이라 전체 카탈로그(10만 개)에서도 1ms 안팎 (count는 약 35ms) — 검색 · 추천 요청마다 실행된다."""
    row = db.execute(select(Item.id, Item.source, Item.source_item_id, Item.text_embedding)
                     .order_by(Item.id.desc()).limit(1)).first()
    if row is None:
        return None
    emb = None if row.text_embedding is None else hashlib.blake2b(
        np.asarray(row.text_embedding, np.float32).tobytes(), digest_size=8).hexdigest()
    return (row.id, row.source, row.source_item_id, emb)


def get_index(db: Session) -> CatalogIndex:
    """카탈로그 지문이 바뀌면 다시 만든다. 기존 상품의 임베딩만 일부 바뀐 경우는 서버 재시작 때 반영."""
    has_text = Item.text_embedding.is_not(None)
    key = (str(db.bind.url), catalog_signature(db))
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
