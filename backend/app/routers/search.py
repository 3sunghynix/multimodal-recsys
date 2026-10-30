"""검색.

POST /search        텍스트(한 · 영) · 이미지 업로드를 함께 받는 통합 검색 (화면은 이것만 쓰면 된다)
GET  /search        키워드 포함 검색 (ILIKE, 인코더 없이 동작하는 단순 경로)
POST /search/image  이미지 → pgvector 코사인 상위 k (데이터셋 결합 없음)

통합 검색 점수 (실험 근거: docs/project/검색.md)
  텍스트 질의 → 한국어면 패션 용어 사전으로 영어 → 두 신호를 반씩
      text       : MiniLM 질의 → 상품 text_embedding
      text_image : Marqo 텍스트 질의 → 상품 image_embedding
  이미지 업로드 → image : Marqo 이미지 질의 → 상품 image_embedding
  텍스트 · 이미지를 둘 다 주면 text_weight : (1 − text_weight) 로 섞는다 (화면의 우선순위 슬라이더)
  데이터셋 결합 per_source_rank: 데이터셋 안의 순위 백분위 → 상위 k가 카탈로그 비율대로 섞인다.
      단 최고 상품의 원래 코사인이 그 질의 최고값의 min_rel(기본 0.7) 미만인 데이터셋은 후순위 — 비율 칸을 받지 못하고
      다른 데이터셋 상품을 다 쓴 뒤에만 채워진다 (제외가 아님, 결과는 항상 k개)
"""
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Item
from ..schemas import ItemOut, SearchHit, SearchOut
from ..services import search_core
from ..services.catalog_index import get_index
from ..services.translate_ko import HANGUL, translate_ko

router = APIRouter(prefix="/search", tags=["search"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


async def _read_image(file: UploadFile) -> bytes:
    """클라이언트 측 오류를 먼저: 형식(415) → 크기(413) → 빈 파일(400)."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"허용되지 않는 이미지 형식: {file.content_type}")
    # 상한+1까지만 읽어 큰 파일이 메모리에 통째로 올라오지 않게 한다
    data = await file.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, f"파일이 너무 큽니다 (최대 {settings.max_image_bytes // (1024 * 1024)}MB)")
    if len(data) == 0:
        raise HTTPException(400, "빈 파일")
    return data


def _ready(request: Request, name: str, what: str):
    enc = getattr(request.app.state, name, None)
    if enc is None:
        raise HTTPException(503, f"{what} 인코더가 비활성 상태")
    if not enc.pca_ready:
        raise HTTPException(503, f"{what} PCA 파일 미설치")
    return enc


@router.post("", response_model=SearchOut)
async def search(
    request: Request,
    q: str | None = Form(None, description="검색어 (한국어 · 영어)"),
    file: UploadFile | None = File(None, description="검색 이미지 (jpeg · png · webp)"),
    text_weight: float = Form(0.5, ge=0, le=1, description="텍스트와 이미지를 함께 줄 때 텍스트 비중"),
    k: int = Form(20, ge=1, le=100),
    merge: str = Form("per_source_rank", description="per_source_rank · per_source_z · global"),
    min_rel: float = Form(0.7, ge=0, le=1, description="per_source_rank: 최고 상품이 이 비율(질의 최고 코사인 대비) 미만인 데이터셋은 후순위 (0이면 끔)"),
    source: str | None = Form(None, description="데이터셋 한정 (amazon · hm)"),
    brand_id: int | None = Form(None),
    db: Session = Depends(get_db),
):
    q = (q or "").strip() or None
    if q is None and file is None:
        raise HTTPException(422, "검색어 q 또는 이미지 file 중 하나는 필요합니다")
    if merge not in search_core.MERGES:
        raise HTTPException(422, f"merge는 {search_core.MERGES} 중 하나")
    data = await _read_image(file) if file is not None else None

    # 질의 인코딩
    query_en, untranslated, queries = q, [], {}
    w_text = text_weight if data is not None else 1.0
    w_img = 1.0 - text_weight if q is not None else 1.0
    weights = {}
    if q is not None:
        if HANGUL.search(q):
            query_en, untranslated = translate_ko(q)
        queries["text"] = _ready(request, "text_encoder", "텍스트").encode([query_en])
        weights["text"] = w_text
        img_enc = getattr(request.app.state, "image_encoder", None)
        if img_enc is not None and img_enc.pca_ready:          # 이미지 인코더가 없으면 MiniLM만
            queries["text_image"] = img_enc.encode_texts([query_en])
            weights = {"text": w_text / 2, "text_image": w_text / 2}
    if data is not None:
        enc = _ready(request, "image_encoder", "이미지")
        try:
            queries["image"] = enc.encode_image_bytes(data)[None, :]
        except Exception as e:
            raise HTTPException(400, f"이미지 처리 실패: {e}")
        weights["image"] = w_img
    weights = {m: round(w, 4) for m, w in weights.items() if w > 0}
    if not weights:
        raise HTTPException(422, "가중치가 모두 0입니다 (text_weight 확인)")

    # 점수 계산 (메모리 색인, 필터 적용 후 그 안에서 결합)
    ix = get_index(db)
    keep = np.ones(len(ix), bool)
    if source is not None:
        keep &= ix.sources == source
    if brand_id is not None:
        keep &= ix.brands == brand_id
    hits = []
    if keep.any():
        sel = np.flatnonzero(keep)
        item_mats = {"text": ix.text[sel], "text_image": ix.image[sel], "image": ix.image[sel]}
        top, scores = search_core.search({m: queries[m] for m in weights}, {m: item_mats[m] for m in weights},
                                         weights, ix.sources[sel], k, merge, min_rel or None)
        ids = ix.ids[sel][top[0]].tolist()
        by_id = {it.id: it for it in db.scalars(select(Item).where(Item.id.in_(ids)))}
        hits = [SearchHit(**ItemOut.model_validate(by_id[i]).model_dump(), score=float(s))
                for i, s in zip(ids, scores[0]) if i in by_id]
    return SearchOut(query=q, query_en=query_en, untranslated=untranslated, weights=weights, merge=merge, items=hits)


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
    data = await _read_image(file)

    # 서비스 상태 오류: 인코더·PCA·DB 중 하나라도 준비 안 됐으면 503
    encoder = getattr(request.app.state, "image_encoder", None)
    if encoder is None:
        raise HTTPException(503, "이미지 인코더가 비활성 상태 (settings.load_image_encoder=False)")
    if not encoder.pca_ready:
        raise HTTPException(503, f"PCA 파일 미설치 (settings.pca_image_path={settings.pca_image_path})")
    if db.bind.dialect.name != "postgresql":
        raise HTTPException(503, "벡터 검색은 PostgreSQL + pgvector 환경에서만 동작한다")

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
