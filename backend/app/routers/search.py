"""검색. 텍스트 키워드(ILIKE)와 이미지 쿼리(Marqo + PCA 256 → pgvector 코사인) 두 경로를 제공한다.

이미지 검색은 Marqo-FashionSigLIP(open_clip)로 쿼리를 임베딩하고 상품과 같은 PCA 256 공간으로
투영한 뒤 items.image_embedding과 코사인 유사도(pgvector `<=>`)로 상위 K개를 반환한다.
상품 임베딩은 load_dev_subset이 드라이브 embeddings/v2를 읽어 채운다.
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
    # 1) 클라이언트 측 오류를 먼저: 형식(415) → 크기(413) → 빈 파일(400)
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"허용되지 않는 이미지 형식: {file.content_type}")

    # 상한+1까지만 읽어 큰 파일이 메모리에 통째로 올라오지 않게 한다
    data = await file.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, f"파일이 너무 큽니다 (최대 {settings.max_image_bytes // (1024 * 1024)}MB)")
    if len(data) == 0:
        raise HTTPException(400, "빈 파일")

    # 2) 서비스 상태 오류: 인코더·PCA·DB 중 하나라도 준비 안 됐으면 503
    encoder = getattr(request.app.state, "image_encoder", None)
    if encoder is None:
        raise HTTPException(503, "이미지 인코더가 비활성 상태 (settings.load_image_encoder=False)")
    if not encoder.pca_ready:
        raise HTTPException(503, f"PCA 파일 미설치 (settings.pca_image_path={settings.pca_image_path})")
    if db.bind.dialect.name != "postgresql":
        raise HTTPException(503, "벡터 검색은 PostgreSQL + pgvector 환경에서만 동작한다")

    # 3) 쿼리 인코딩 + ANN 검색
    try:
        query_vec = encoder.encode_image_bytes(data)
    except Exception as e:
        raise HTTPException(400, f"이미지 처리 실패: {e}")

    stmt = (select(Item)
            .where(Item.image_embedding.is_not(None))
            .order_by(Item.image_embedding.cosine_distance(query_vec.tolist()))
            .limit(k))
    if brand_id is not None:
        stmt = stmt.where(Item.brand_id == brand_id)
    return db.scalars(stmt).all()
