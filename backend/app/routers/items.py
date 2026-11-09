"""상품 조회. 프론트가 검색·추천 결과 상품 상세를 한 번에 받아 카드로 그리기 위한 배치 조회를 제공한다.

GET /items?ids=1,2,3   카드 그리기용 배치 (ItemOut — brand_name 포함)
GET /items/version     상품 테이블 상태 요약 — 시뮬레이터 bridge 등 snapshot 동일성 확인용
GET /items/{item_id}   상품 상세 페이지용 (ItemDetailOut — ItemOut + description + source_item_id)
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Brand, Item
from ..schemas import ItemDetailOut, ItemOut

router = APIRouter(prefix="/items", tags=["items"])

MAX_BATCH = 500


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
    if len(id_list) > MAX_BATCH:
        raise HTTPException(400, f"한 번에 조회 가능한 상품 수는 {MAX_BATCH}개까지")
    # Brand 를 LEFT JOIN 해서 카드가 "BRAND #10" 대신 실제 이름을 바로 쓰도록 (N+1 쿼리 방지).
    rows = db.execute(
        select(Item, Brand.name).outerjoin(Brand, Item.brand_id == Brand.id)
        .where(Item.id.in_(id_list))
    ).all()
    by_id = {item.id: (item, brand_name) for item, brand_name in rows}
    # 입력 순서 유지 (추천/검색 결과 순서가 중요)
    return [_item_out(*by_id[i]) for i in id_list if i in by_id]


# 주의: /items/{item_id} 보다 먼저 선언해야 "version" 이 int 로 파싱되지 않고 이 핸들러가 잡힌다.
@router.get("/version")
def items_version(db: Session = Depends(get_db)):
    """상품 테이블 상태 요약. 광석 시뮬레이터 bridge(PR #29)가 snapshot 과 live DB 의 동일성을
    확인할 때 쓸 수 있도록 노출 — 상품은 추가만 되고 삭제 안 되는 전제라 (count, max_id) 조합이
    바뀌면 상태가 바뀐 것. 가벼운 쿼리라 왕복 1회로 호출자가 판단한다.

    강한 동일성(해시 기반) 이 필요하면 export_simulator_items 가 남긴 파일 해시와 대조하는
    외부 패턴을 쓴다 — 서버가 매 호출마다 전체 id 를 해시하는 건 10만 상품에서 비싸다."""
    count, max_id = db.execute(select(func.count(Item.id), func.max(Item.id))).one()
    return {"count": int(count), "max_id": int(max_id) if max_id is not None else 0}


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
