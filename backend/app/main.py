"""멀티 브랜드 멀티모달 추천 서비스 API. 실행: uvicorn app.main:app --reload (backend/ 에서), 문서: /docs"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .db import init_db
from .routers import admin, events, recommend, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Multimodal RecSys", version="0.1.0", lifespan=lifespan)
for r in (search.router, recommend.router, events.router, admin.router):
    app.include_router(r)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
