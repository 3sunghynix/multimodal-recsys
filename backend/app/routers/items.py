"""상품 조회. 프론트가 검색·추천 결과 상품 상세를 한 번에 받아 카드로 그리기 위한 배치 조회를 제공한다."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Item
from ..schemas import ItemOut

router = APIRouter(prefix="/items", tags=["items"])


@router.get("", response_model=list[ItemOut])
def get_items(ids: str = Query(..., description="쉼표로 구분한 item id 목록 (예: 1,2,3)"), db: Session = Depends(get_db)):
    try:
        id_list = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "ids는 쉼표 구분 정수 목록이어야 함")
    if not id_list:
        return []
    if len(id_list) > 500:
        raise HTTPException(400, "한 번에 조회 가능한 상품 수는 500개까지")
    rows = {r.id: r for r in db.scalars(select(Item).where(Item.id.in_(id_list))).all()}
    # 입력 순서 유지 (추천/검색 결과 순서가 중요)
    return [rows[i] for i in id_list if i in rows]


@router.get("/{item_id}", response_model=ItemOut)
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = db.get(Item, item_id)
    if item is None:
        raise HTTPException(404, f"상품 없음: {item_id}")
    return item
