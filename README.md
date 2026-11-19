# MSRS — 멀티모달 패션 검색 · 추천 서비스

3성하이닉스 캡스톤. 상품의 **이미지와 텍스트를 함께** 써서 여러 브랜드(Amazon 의류 · 신발 · 주얼리 + H&M, 상품 106,825개)의 패션 상품을 검색하고 개인화 추천한다. 페르소나 시뮬레이터로 A/B 테스트하고, 쌓인 로그로 재학습한다.

- 논문: [docs/논문_최종본.pdf](docs/논문_최종본.pdf) · 가설 대응표: [docs/가설대응표.md](docs/가설대응표.md)
- 진행 상황 · 남은 일: [docs/project/다음작업.md](docs/project/다음작업.md)

## 주요 결과 (2026-11-16)

| 영역 | 결과 | 문서 |
|---|---|---|
| 검색 | 영어 정확도 0.933 · 한국어 0.909, 응답 p95 **146ms** (상품 10만 개, 목표 ≤ 200ms) | [검색.md](docs/project/검색.md) |
| 추천 (오프라인) | 서비스 모델(UME + 인기 + 브랜드)이 인기 추천 대비 H&M MAP@12 **2.7배**, Amazon Recall@12 +46% · 브랜드 일치도 0.2% → 10.7% | [오프라인결과.md](docs/project/오프라인결과.md) |
| 추천 (서빙) | 세션 반영 · 소분류 다양성 재정렬(추천 상품 종류 +19%) · MAB 탐색 슬롯, 응답 p95 **46 ~ 59ms** | [추천모델설계.md](docs/project/추천모델설계.md), [MAB_탐색.md](docs/project/MAB_탐색.md) |
| A/B (모의 고객 1만 명) | UME +1.6%p (p = 0.086), 브랜드충성 +13.2%p · 실용주의자 +6.0%p 유의 | [실제카탈로그_AB_파일럿.md](docs/project/실제카탈로그_AB_파일럿.md) |
| 지속 학습 | 서비스 로그로 이어 학습 → 기존 모델과 MAP@12 · Recall@20 비교 → 후보 등록 (교체는 관리자) | [재학습.md](docs/project/재학습.md) |

## 구조

| 폴더 | 내용 | 담당 |
|---|---|---|
| `backend/` | FastAPI 서비스: 검색 · 추천 · 이벤트 · 모델 레지스트리 · A/B · 지표 API, Alembic 마이그레이션, 사용자 쇼핑 화면(`frontend/`) | 김형우 |
| `recsys/` | 오프라인 학습 · 평가: UME 학습(`ume.py`), 재학습 잡(`retrain.py`), 검색 점수(`search.py`), 평가 지표(`eval/`) | 김유환 |
| `simulator/` | 페르소나 7종 고객 행동 시뮬레이터, 검색어 생성기, 실제 카탈로그 A/B 파일럿 | 김광석 |
| `dashboard/` | 관리자 대시보드 (Streamlit): A/B 테스트 · 시뮬레이터 · 모니터링 | 김광석 |
| `monitoring/` | 지표 창 감시 · 알림 · 재학습 요청 기록 (runner) | 김광석 |
| `data/` | 데이터셋 안내 (원본 · 가공본 · 임베딩은 git 제외) | 김유환 |
| `paper/` | 논문 실험 파이프라인 (사전등록 1 ~ 10차) | 김유환 |
| `docs/project/` | 설계 · 결과 문서 (아래 "문서" 참고) | 전원 |

## 실행

필요: Docker Desktop (RAM 16GB · CPU 4코어 권장). 코드가 이미지에 들어가므로 **main을 갱신하면 다시 빌드**한다.

### 1. 처음 한 번

```bash
cp .env.example .env
docker compose up -d --build            # db · redis · api · dashboard (처음 빌드는 인터넷 필요, 백신 HTTPS 검사는 잠시 끄기)
```

- DB 스키마는 api가 시작할 때 Alembic으로 자동으로 최신이 된다 (`docker compose down -v` 필요 없음).
- 검색 모델(약 900MB)은 첫 실행 때 내려받는다. **받지 못하면 api가 종료된다** (`docker compose ps -a`에서 `Exited (3)`, 로그에 `SSLError`) → 백신 HTTPS 검사를 끄고 `docker compose up -d api`. 받은 뒤 `.env`의 `HF_HUB_OFFLINE=1`로 바꾸면 인터넷 확인을 건너뛴다 (`0`이면 캐시가 있어도 확인하다 실패할 수 있다).
- Windows Git Bash에서는 `/data/...` 경로가 `C:/Program Files/Git/data/...`로 바뀐다 → 명령 앞에 `MSYS_NO_PATHCONV=1`을 붙이거나 PowerShell에서 실행.

### 2. 데이터 넣기 (둘 중 하나)

| 규모 | 방법 |
|---|---|
| **개발용** (상품 1,990 · 고객 6,000) | 드라이브 `msrs_project/dev_subset/` → `data/dev_subset/`, `embeddings/v2/` → `data/embeddings/v2/` 후<br>`docker compose exec api python -m scripts.load_dev_subset --data /data/dev_subset --embeddings /data/embeddings/v2 --reset` |
| **전체 카탈로그** (상품 106,825 · 이벤트 107만) | [로컬실행.md 4절](docs/project/로컬실행.md): zip을 **OneDrive 밖**(예: `C:\msrs_data\full_catalog`)에 풀고 `docker-compose.override.yml`로 연결한 뒤 적재 |

UME 가중치: 드라이브 `msrs_project/results/ume/`의 `hm_test_pca/` · `amazon_test_pca/` → `data/models/ume/` (없으면 등록이 거절되고 인기 모델로 대체된다).

```bash
docker compose restart api                                          # 검색 PCA 파일은 api 시작 때 읽는다 (데이터를 넣은 뒤 한 번)
docker compose exec api python -m scripts.register_ume --activate   # UME 모델 등록 · 운영 지정
```

데모 데이터만 필요하면 `docker compose exec api python -m scripts.seed_sample` (브랜드 3 · 상품 30 · 고객 10).

### 3. 화면

| 화면 | 주소 |
|---|---|
| 사용자 쇼핑 화면 (검색 · 추천 · 상품 상세) | http://localhost:8000/app/ |
| 관리자 대시보드 | http://localhost:8501 |
| API 문서 | http://localhost:8000/docs |

### 4. 재학습 (지속 학습)

```bash
docker compose run --rm trainer                                                     # 새 로그 1만 건 이상이면 학습 · 후보 등록
docker compose run --rm trainer python -m recsys.retrain --min-new-logs 0 --no-register   # 조건 무시 · 평가만
curl -X POST localhost:8000/models/<id>/activate                                     # 후보를 운영 모델로 (관리자 결정)
```

채택 규칙 · 모니터링 runner 연결: [재학습.md](docs/project/재학습.md)

### 5. 테스트

```bash
# 백엔드 (api 컨테이너. 대부분 SQLite로 돌고, 일부는 db 컨테이너가 떠 있어야 한다)
docker compose run --rm -e REDIS_URL= api pytest -q
# 학습 · 평가 (trainer 컨테이너). backend 원본과 recsys 사본이 같은지 보는 테스트 1개는 컨테이너에 원본이 없어 제외
docker compose --profile trainer run --rm --no-deps trainer python -m pytest -q recsys/tests \
  --deselect recsys/tests/test_search.py::test_backend_copies_match
```

Docker 없이 백엔드만: `cd backend && pip install -r requirements-dev.txt && uvicorn app.main:app --reload` (기본 SQLite).

## API

| 메서드 · 경로 | 기능 |
|---|---|
| `POST /search` | 통합 검색: 텍스트(한국어 · 영어) · 이미지 업로드 · 혼합, 데이터셋별 순위 결합, 미번역 단어 알림 |
| `GET /search?q=` · `POST /search/image` | 텍스트 · 이미지 단독 검색 |
| `GET /recommend/{user_id}?k=&session_id=` | 개인화 추천 (A/B 실험 중이면 그룹 모델, 아니면 운영 모델). 응답에 `request_id` · `position` · `impression_id` · 추천 사유 · 탐색 여부 · 단계별 응답 시간 |
| `POST /events` · `POST /events/batch` | 행동 로그 (view · click · add_to_cart · purchase · search), 최대 1,000건 일괄. `event_id` 멱등, `request_id`로 모델 정보 자동 채움 |
| `POST /users` | 고객 생성 (`source` · `source_user_id` 멱등) |
| `GET /items` · `GET /items/{id}` · `GET /items/version` | 상품 조회 · 상세 · 카탈로그 버전 |
| `GET/POST /models`, `POST /models/{id}/activate` | 모델 레지스트리, 모델 교체 |
| `POST /experiments`, `GET /experiments/{id}/metrics` · `/stats` | A/B 실험, 그룹별 지표 · 통계 검정(Wilson 구간 · 두 비율 Z 검정) |
| `GET /metrics/window` · `/metrics/log-delta`, `GET/POST /metrics/snapshots` | 지속 학습용 지표 창(노출 기준 CTR) · 새 로그 수 · 모델별 지표 기록 |
| `GET /features/latency` | Feature Store(Redis) 조회 시간 |

연결 규약(추천 → 노출 → 클릭): [이벤트연결규약.md](docs/project/이벤트연결규약.md). 새 추천 모델은 `backend/app/recommenders/`에 `Recommender`를 상속해 `@register`로 등록하면 위 API에서 바로 쓸 수 있다.

## 문서

| 주제 | 문서 |
|---|---|
| 데이터 | [data/README.md](data/README.md), [데이터스키마.md](docs/project/데이터스키마.md), [카테고리분류.md](docs/project/카테고리분류.md) |
| 검색 | [검색.md](docs/project/검색.md) |
| 추천 | [추천모델설계.md](docs/project/추천모델설계.md), [오프라인기준선.md](docs/project/오프라인기준선.md), [오프라인결과.md](docs/project/오프라인결과.md), [MAB_탐색.md](docs/project/MAB_탐색.md), [재학습.md](docs/project/재학습.md) |
| 평가 · 시뮬레이터 | [지표정의.md](docs/project/지표정의.md), [페르소나_v0.3.md](docs/project/페르소나_v0.3.md), [시뮬레이터_실행기.md](docs/project/시뮬레이터_실행기.md), [시뮬레이터_API_연결.md](docs/project/시뮬레이터_API_연결.md), [실제카탈로그_AB_파일럿.md](docs/project/실제카탈로그_AB_파일럿.md) |
| 운영 · 대시보드 | [로컬실행.md](docs/project/로컬실행.md), [이벤트연결규약.md](docs/project/이벤트연결규약.md), [대시보드_실행기.md](docs/project/대시보드_실행기.md), [관리자API현황.md](docs/project/관리자API현황.md) |

## 협업 규칙

- main에 직접 push하지 않는다. 기능 브랜치 → PR(팀 템플릿) → 리뷰 담당자 1명 승인 후 병합. 본인 PR은 본인이 병합하지 않는다.
- API 키 · 계정 정보 · `.env` · 대용량 데이터 · 모델 가중치는 커밋하지 않는다.
- 결정 사항은 문서(`docs/project/`)와 위키에 남긴다.
