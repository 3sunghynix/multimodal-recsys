"""프로젝트 명세(2-5) "API 응답 필수 필드 규격" 호환 확인.

검색:
  search_type · results[].{product_id, name, score, price} · latency_ms · total_count
추천:
  recommendations[].{product_id, score, reason, is_exploration}
  pipeline_latency.{candidate_ms, ranking_ms, reranking_ms, total_ms}
  session_context.{recent_clicks, session_interest}

하위호환으로 기존 필드(items 등)도 그대로 돌려주는지 함께 확인.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.models import Item
from scripts.seed_sample import seed


@pytest.fixture(autouse=True)
def _no_encoders(monkeypatch):
    monkeypatch.setattr(settings, "load_image_encoder", False)
    monkeypatch.setattr(settings, "load_text_encoder", False)


class _FakeText:
    """POST /search 테스트용. 상품 text_embedding 과 바로 매칭되는 단순 벡터를 돌려준다."""
    pca_ready = True

    def encode(self, texts):
        v = np.zeros((1, 256), np.float32)
        v[0, 0] = 1.0
        return v


@pytest.fixture()
def client(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    with Session() as s:
        seed(s)
        # POST /search 가 메모리 색인(ix) 을 쓰려면 상품에 text_embedding/image_embedding 이 있어야 한다.
        v = np.zeros(256, np.float32); v[0] = 1.0
        for item in s.scalars(select(Item)).all():
            item.text_embedding = v.tolist()
            item.image_embedding = v.tolist()
        s.commit()

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    # POST /search 가 쓸 fake 텍스트 인코더만 세팅. 이미지 인코더는 테스트에서 안 씀.
    app.state.text_encoder = _FakeText()
    app.state.image_encoder = None
    yield TestClient(app)
    app.dependency_overrides.clear()
    app.state.text_encoder = None


# ─── /recommend 응답 규격 ────────────────────────────────────────────────────

def test_recommend_response_matches_spec_schema(client):
    body = client.get("/recommend/1?k=5").json()

    # 명세 recommendations[]
    assert isinstance(body["recommendations"], list) and len(body["recommendations"]) == 5
    for rec in body["recommendations"]:
        assert isinstance(rec["product_id"], str) and rec["product_id"].isdigit()
        assert isinstance(rec["score"], (int, float))
        assert rec["score_kind"] in ("model", "rank")      # 점수 출처 명시 (리뷰 지적 반영)
        assert isinstance(rec["reason"], str) and rec["reason"]
        assert isinstance(rec["is_exploration"], bool)

    # 명세 pipeline_latency
    pl = body["pipeline_latency"]
    assert set(pl) == {"candidate_ms", "ranking_ms", "reranking_ms", "total_ms"}
    assert all(isinstance(pl[k], (int, float)) for k in pl)
    assert pl["total_ms"] >= 0

    # 명세 session_context
    sc = body["session_context"]
    assert set(sc) == {"recent_clicks", "session_interest"}
    assert isinstance(sc["recent_clicks"], list)

    # 하위호환: 기존 items (시뮬레이터 bridge 가 읽음) 유지
    assert body["items"] and body["items"] == [int(r["product_id"]) for r in body["recommendations"]]


def test_recommend_reason_reflects_active_model(client):
    body = client.get("/recommend/1?k=3").json()
    # seed 는 popularity 를 active 로 등록 → reason 은 명세 라벨 "popular".
    assert body["model"] == "popularity"
    assert {r["reason"] for r in body["recommendations"]} == {"popular"}


def test_recommend_session_context_includes_recent_clicks(client):
    # 고객 1 이 상품 3, 7 을 순차 클릭
    client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3})
    client.post("/events", json={"user_id": 1, "type": "click", "item_id": 7})
    body = client.get("/recommend/1?k=3").json()
    # 최근순, 문자열 ID
    assert body["session_context"]["recent_clicks"][:2] == ["7", "3"]
    assert body["session_context"]["session_interest"] is None


def test_popularity_emits_real_model_scores_not_rank_placeholder(client):
    """리뷰 반영: Popularity 는 이벤트 가중치 합계를 실제 점수로 돌려줘야 하고,
    응답의 score_kind 가 "model" 로 노출돼야 한다 (랭킹 placeholder 와 구분)."""
    body = client.get("/recommend/1?k=5").json()
    recs = body["recommendations"]
    # 전부 model 점수 (Popularity.scored_native=True)
    assert {r["score_kind"] for r in recs} == {"model"}
    # 순위 기반 placeholder 는 정확히 1.0 → 0.0 등간격이지만, 실제 이벤트 합계는 정수/소수 배수가 아님.
    # 최상위가 1.0 이고 간격이 균등하면 placeholder 를 의심 — 그런 패턴이 아닌지 확인.
    scores = [r["score"] for r in recs]
    assert scores[0] > 0.0                                 # seed 이벤트가 있으니 top 은 0 보다 큼
    # placeholder 는 [1.0, 0.8, 0.6, 0.4, 0.2] 같은 형식. 실제 가중치 합이면 이 등간격 패턴을 벗어남.
    expected_placeholder = [round(1 - (i / 5), 4) for i in range(5)]
    assert scores != expected_placeholder


# ─── /search 응답 규격 ──────────────────────────────────────────────────────

def test_search_text_response_matches_spec_schema(client):
    body = client.post("/search", data={"q": "Alpha dress", "k": 5}).json()
    assert body["search_type"] == "text"
    assert isinstance(body["results"], list)
    assert body["total_count"] == len(body["results"])
    assert isinstance(body["latency_ms"], (int, float)) and body["latency_ms"] >= 0
    for item in body["results"]:
        assert isinstance(item["product_id"], str) and item["product_id"].isdigit()
        assert isinstance(item["name"], str) and item["name"]
        assert isinstance(item["score"], (int, float))
        # price 는 None 일 수 있음 (명세는 price 필드 존재만 요구)
        assert "price" in item
    # 하위호환: 기존 items 유지
    assert len(body["items"]) == len(body["results"])


def test_search_type_switches_with_inputs(client):
    # image-only: 인코더 없으니 503 이지만 search_type 판정은 요청만으로 가능 → 405 가 아닌 503 확인은 다른 테스트.
    # 여기선 text/hybrid 만 확인 (hybrid 는 이미지 파일 있으면 OK 하지만 인코더 없어 503, 그래서 체크 어려움).
    text_only = client.post("/search", data={"q": "Alpha", "k": 2}).json()
    assert text_only["search_type"] == "text"
