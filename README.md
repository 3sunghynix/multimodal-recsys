# multimodal-recsys

3성하이닉스 캡스톤 — 다중 브랜드 멀티모달(텍스트 · 이미지) 추천 · 검색 서비스.
논문 실험 코드는 `paper/`, 논문 · 실험 기록은 `docs/` (논문: [docs/논문_최종본.pdf](docs/논문_최종본.pdf), 가설 대응표: [docs/가설대응표.md](docs/가설대응표.md)).

## 구조

| 폴더 | 내용 | 담당 |
|---|---|---|
| `backend/` | FastAPI 서비스: 검색 · 추천 · 로그 · 모델 레지스트리 · A/B 테스트 | 김형우 |
| `recsys/` | 오프라인 학습 · 평가 (UME 기준 모델, 3단계 학습, 평가 지표, 재학습) | 김유환 · 김광석 |
| `simulator/` | 페르소나 기반 사용자 시뮬레이터 (A/B · 재학습용 로그 생성) | 김광석 |
| `dashboard/` | 관리자(실험실 · 알림 · 모델 교체) · 사용자(쇼핑 · 검색) 화면 | 김광석 설계 |
| `data/` | 데이터셋 안내 (원본 · 가공본은 git 제외) | 김유환 |
| `paper/` | 논문 실험 파이프라인 (사전 등록 1 ~ 10차) | — |

설계 문서: [docs/project/데이터스키마.md](docs/project/데이터스키마.md)

## 실행

**Docker (API + PostgreSQL/pgvector)**

```bash
cp .env.example .env
docker compose up --build                              # http://localhost:8000/docs
docker compose exec api python -m scripts.seed_sample  # 데모 데이터 (브랜드 3 · 상품 30 · 고객 10)
```

**로컬 (SQLite)**

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m scripts.seed_sample
uvicorn app.main:app --reload
pytest -q                                              # API 전체 흐름 테스트
```

## API (현재)

| 메서드 · 경로 | 기능 |
|---|---|
| `GET /search?q=` | 텍스트 키워드 검색 (임베딩 검색 예정) |
| `POST /search/image` | 이미지 업로드 검색 (자리만 있음, 501) |
| `GET /recommend/{user_id}` | 추천. 진행 중인 A/B 실험이 있으면 그룹별 모델, 없으면 active 모델 |
| `POST /events` | 행동 로그 (view · click · add_to_cart · purchase · search) |
| `GET/POST /models`, `POST /models/{id}/activate` | 모델 레지스트리, 모델 교체 버튼 |
| `POST /experiments`, `GET /experiments/{id}/metrics` | A/B 실험 생성, 그룹별 노출 · 클릭 · 구매 · CTR |

새 추천 모델은 `backend/app/recommenders/`에 `Recommender`를 상속해 `@register`로 등록하면 위 API에서 바로 쓸 수 있다.
