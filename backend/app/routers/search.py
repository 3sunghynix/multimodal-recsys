"""검색. 지금은 텍스트 키워드 검색만 동작한다.

예정: 텍스트 · 이미지 임베딩 검색(pgvector 또는 FAISS), 텍스트 · 이미지 우선순위 가중치, 한국어 ↔ 영어(다국어 인코더).
"""
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Item
from ..schemas import ItemOut

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=list[ItemOut])
def search_text(q: str = Query(min_length=1), brand_id: int | None = None, k: int = 20, db: Session = Depends(get_db)):
    words = [w for w in q.split() if w]
    stmt = select(Item).where(*[or_(Item.title.ilike(f"%{w}%"), Item.description.ilike(f"%{w}%")) for w in words])
    if brand_id is not None:
        stmt = stmt.where(Item.brand_id == brand_id)
    return db.scalars(stmt.order_by(Item.id).limit(k)).all()


@router.post("/image")
async def search_image(file: UploadFile = File(...)):
    # 업로드 형식만 고정해 둔다. 이미지 인코더 · 벡터 인덱스가 붙으면 여기서 유사 상품을 반환한다
    raise HTTPException(501, f"이미지 검색은 아직 구현되지 않았습니다 (받은 파일: {file.filename})")
