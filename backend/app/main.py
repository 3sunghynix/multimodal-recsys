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
from .routers import admin, events, features, items, metrics, recommend, search, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # B4: 관리자 쓰기 엔드포인트 (admin.py 의 POST) 는 settings.admin_api_key 로 보호된다.
    # 비어 있으면 로컬 개발/테스트용으로 통과시키는데, 이 상태로 운영에 올라가는 사고를 막기 위해 한 번 경고한다.
    if not settings.admin_api_key:
        print("[WARN] ADMIN_API_KEY 미설정 — /models · /experiments 쓰기 엔드포인트가 공개 상태입니다 "
              "(로컬 개발에서는 정상). 운영에서는 .env 또는 docker-compose에 반드시 설정하세요.")
    # D1: 검색 인코더 로드 실패(HF 다운로드 · SSL · 디스크 등)로 api 전체가 죽는 걸 막는다.
    # 실패하면 해당 인코더는 None 으로 두고 WARN 만 찍는다 — search 라우터의 `_ready` 가 503 으로 응답하고
    # 키워드 검색 (`GET /search`) · 추천 · 이벤트 등은 그대로 동작. 실패 원인은 `/health` 의 `*_encoder_error` 로 확인.
    # 테스트에서는 settings.load_*_encoder=False 로 아예 끈다.
    # PCA 파일이 없어도 encoder 는 뜬다 (`_ready` 가 `pca_ready` 검사로 503).
    app.state.image_encoder = None
    app.state.image_encoder_error = None
    if settings.load_image_encoder:
        try:
            from .encoders.clip_image import load_encoder
            app.state.image_encoder = load_encoder(settings.image_encoder_model, settings.pca_image_path)
        except Exception as error:                                   # noqa: BLE001 — startup에서 어떤 예외든 포획해야 함
            app.state.image_encoder_error = f"{type(error).__name__}: {error}"
            print(f"[WARN] 이미지 인코더 로드 실패 — /search/image 는 503 으로 응답합니다. ({app.state.image_encoder_error})")
    app.state.text_encoder = None
    app.state.text_encoder_error = None
    if settings.load_text_encoder:
        try:
            from .encoders.text import load_text_encoder
            app.state.text_encoder = load_text_encoder(settings.text_encoder_model, settings.pca_text_path)
        except Exception as error:                                   # noqa: BLE001
            app.state.text_encoder_error = f"{type(error).__name__}: {error}"
            print(f"[WARN] 텍스트 인코더 로드 실패 — POST /search 텍스트 질의는 503 으로 응답합니다. ({app.state.text_encoder_error})")
    # 검색 · 추천 상품 색인을 백그라운드로 미리 만든다 (전체 카탈로그 10만 개면 약 17초 — 첫 사용자가 기다리지 않게)
    import threading

    def _warm():
        try:
            from .db import SessionLocal
            from .services.catalog_index import get_index
            with SessionLocal() as db:
                get_index(db)
        except Exception as e:                                   # DB가 비었거나 아직 준비 전이어도 서버는 뜬다
            print(f"[색인 미리 만들기 건너뜀] {e}")
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="Multimodal RecSys", version="0.1.0", lifespan=lifespan)
for r in (search.router, recommend.router, events.router, admin.router, items.router, users.router, features.router, metrics.router):
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
    # D1: 로드 실패 사유 — 설정됐을 때만 돌려줘서 정상 응답은 그대로 유지한다.
    body = {"status": "ok", "image_encoder": settings.image_encoder_model if enc else None,
            "text_encoder": settings.text_encoder_model if txt else None}
    for name in ("image_encoder_error", "text_encoder_error"):
        err = getattr(app.state, name, None)
        if err:
            body[name] = err
    return body
