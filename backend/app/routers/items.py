"""상품 조회. 프론트가 검색·추천 결과 상품 상세를 한 번에 받아 카드로 그리기 위한 배치 조회를 제공한다.

GET /items?ids=1,2,3   카드 그리기용 배치 (ItemOut — brand_name 포함)
GET /items/{item_id}   상품 상세 페이지용 (ItemDetailOut — ItemOut + description + source_item_id)
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Brand, Item
from ..schemas import ItemDetailOut, ItemOut

router = APIRouter(prefix="/items", tags=["items"])


def _item_out(item: Item, brand_name: str | None) -> ItemOut:
    return ItemOut(
        id=item.id, source=item.source, brand_id=item.brand_id, brand_name=brand_name,
        title=item.title, category=item.category, price=item.price, image_path=item.image_path,
    )


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
    # Brand 를 LEFT JOIN 해서 카드가 "BRAND #10" 대신 실제 이름을 바로 쓰도록 (N+1 쿼리 방지).
    rows = db.execute(
        select(Item, Brand.name).outerjoin(Brand, Item.brand_id == Brand.id)
        .where(Item.id.in_(id_list))
    ).all()
    by_id = {item.id: (item, brand_name) for item, brand_name in rows}
    # 입력 순서 유지 (추천/검색 결과 순서가 중요)
    return [_item_out(*by_id[i]) for i in id_list if i in by_id]


@router.get("/{item_id}", response_model=ItemDetailOut)
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = db.get(Item, item_id)
    if item is None:
        raise HTTPException(404, f"상품 없음: {item_id}")
    brand = db.get(Brand, item.brand_id) if item.brand_id is not None else None
    return ItemDetailOut(
        id=item.id, source=item.source, source_item_id=item.source_item_id,
        brand_id=item.brand_id, brand_name=brand.name if brand else None,
        title=item.title, category=item.category, price=item.price,
        image_path=item.image_path, description=item.description or "",
    )
