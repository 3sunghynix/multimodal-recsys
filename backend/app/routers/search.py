"""검색. 텍스트 키워드(ILIKE)와 이미지 쿼리(CLIP + pgvector 코사인) 두 경로를 제공한다.

이미지 검색은 CLIP ViT-B/32의 image encoder로 쿼리를 임베딩한 뒤 items.image_embedding과
코사인 유사도(pgvector `<=>`)로 상위 K개를 반환한다. 상품 임베딩은 seed_sample이 CLIP의
text encoder로 title을 인코딩해 채운다(같은 공유 공간이므로 교차 비교 가능).
"""
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Item
from ..schemas import ItemOut

router = APIRouter(prefix="/search", tags=["search"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@router.get("", response_model=list[ItemOut])
def search_text(q: str = Query(min_length=1), brand_id: int | None = None, k: int = 20, db: Session = Depends(get_db)):
    words = [w for w in q.split() if w]
    stmt = select(Item).where(*[or_(Item.title.ilike(f"%{w}%"), Item.description.ilike(f"%{w}%")) for w in words])
    if brand_id is not None:
        stmt = stmt.where(Item.brand_id == brand_id)
    return db.scalars(stmt.order_by(Item.id).limit(k)).all()


@router.post("/image", response_model=list[ItemOut])
async def search_image(
    request: Request,
    file: UploadFile = File(...),
    k: int = 20,
    brand_id: int | None = None,
    db: Session = Depends(get_db),
):
    encoder = getattr(request.app.state, "image_encoder", None)
    if encoder is None:
        raise HTTPException(503, "이미지 인코더가 비활성 상태 (settings.load_image_encoder=False)")
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"허용되지 않는 이미지 형식: {file.content_type}")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "빈 파일")
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, f"파일이 너무 큽니다 (최대 {settings.max_image_bytes // (1024 * 1024)}MB)")

    try:
        query_vec = encoder.encode_image_bytes(data)
    except Exception as e:
        raise HTTPException(400, f"이미지 처리 실패: {e}")

    # pgvector 코사인 거리 (`<=>`). items.image_embedding이 NULL이면 비교에서 자동 제외된다
    dialect = db.bind.dialect.name
    if dialect != "postgresql":
        raise HTTPException(503, "벡터 검색은 PostgreSQL + pgvector 환경에서만 동작한다")

    stmt = (select(Item)
            .where(Item.image_embedding.is_not(None))
            .order_by(Item.image_embedding.cosine_distance(query_vec.tolist()))
            .limit(k))
    if brand_id is not None:
        stmt = stmt.where(Item.brand_id == brand_id)
    return db.scalars(stmt).all()
