"""멀티 브랜드 멀티모달 추천 서비스 API. 실행: uvicorn app.main:app --reload (backend/ 에서), 문서: /docs

데모 접근:
- API 문서: http://localhost:8000/docs
- 데모 프론트: http://localhost:8000/app/
- 상품 이미지: http://localhost:8000/images/{source}/{item_id}.jpg (load_dev_subset이 압축 해제한 파일)
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import init_db
from .routers import admin, events, items, recommend, search, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # 이미지 인코더 preload (Marqo-FashionSigLIP + PCA 256). 테스트에서는 settings.load_image_encoder=False 로 끈다.
    # PCA 파일이 없어도 encoder는 뜨고, /search/image 라우터가 호출 시 503으로 바꿔 응답한다.
    app.state.image_encoder = None
    if settings.load_image_encoder:
        from .encoders.clip_image import load_encoder
        app.state.image_encoder = load_encoder(settings.image_encoder_model, settings.pca_image_path)
    # 텍스트 인코더 (MiniLM + PCA 256). POST /search 텍스트 질의용
    app.state.text_encoder = None
    if settings.load_text_encoder:
        from .encoders.text import load_text_encoder
        app.state.text_encoder = load_text_encoder(settings.text_encoder_model, settings.pca_text_path)
    yield


app = FastAPI(title="Multimodal RecSys", version="0.1.0", lifespan=lifespan)
for r in (search.router, recommend.router, events.router, admin.router, items.router, users.router):
    app.include_router(r)

# 상품 이미지 서빙 (/images/{source}/{item_id}.jpg). load_dev_subset이 images.zip을 압축 해제한 경로를 노출한다.
# check_dir=False: 서버 시작 후 적재해도 동작하도록 디렉토리 존재 확인을 미룬다 (요청 시 자연스럽게 404).
app.mount("/images", StaticFiles(directory=settings.images_root, check_dir=False), name="images")

# 데모 프론트 (/app/). HTML/JS 한 페이지로 통합 검색 · 추천 · 상품조회를 보여준다.
if Path(settings.frontend_root).exists():
    app.mount("/app", StaticFiles(directory=settings.frontend_root, html=True), name="frontend")


@app.get("/health", tags=["health"])
def health():
    # TestClient처럼 lifespan을 실행하지 않는 환경에선 app.state.image_encoder가 없을 수 있다.
    enc = getattr(app.state, "image_encoder", None)
    txt = getattr(app.state, "text_encoder", None)
    return {"status": "ok", "image_encoder": settings.image_encoder_model if enc else None,
            "text_encoder": settings.text_encoder_model if txt else None}
