# 데이터

공용 데이터는 구글 드라이브 `msrs_project/`(팀 공유, 바로가기 → `/content/drive/MyDrive/msrs_project`)에 있다. 폴더 구조 · 규칙은 드라이브의 `README.md`, 파일 목록 · 체크섬은 `manifest.json`.
저장소의 `data/`에는 로컬 개발용 데이터만 내려받는다 (git 제외: `data/raw/`, `data/processed/`, `data/dev_subset/`, `data/embeddings/`). 키 파일(kaggle.json 등)은 저장소 · 드라이브에 올리지 않는다.

## 현황 (2026-10-29, v2 적재 진행 중)

| 데이터 | 드라이브 위치 | 역할 | 규모 |
|---|---|---|---|
| Amazon Reviews 2023 — Clothing_Shoes_and_Jewelry 5-core | `processed/amazon_v2`, `images/amazon` | 서비스 카탈로그 (다중 브랜드) | v1: 상품 95,662 · 브랜드 23,183 · 고객 269,056 · 이벤트 2,326,938 (2021-10-01 ~ 2023-09-12). v2 = 패션 외 상품 제외 · 5-core 재적용 (`stats.json`) |
| H&M (Kaggle) | `processed/hm`, `images/hm` | 서비스 카탈로그 (단일 브랜드 "H&M") | 최근 13주(2020-06-24 ~ 09-22) 인기 상품 2만 → 5-core (`stats.json`) |
| Retailrocket (Kaggle) | `raw/retailrocket`, `processed/retailrocket/sim_stats.json` | 시뮬레이터 행동 기준값 (카탈로그 아님) | 전환 퍼널 · 세션 · 재구매 · 인기 집중도 · 시간대 |
| DeepFashion (In-shop) | 예정 | 이미지 검색 학습 · 평가 | 공식 신청 필요 |
| 통합 임베딩 | `embeddings/v2` | 서비스 검색 | Amazon v2 + H&M, 텍스트 MiniLM · 이미지 Marqo-FashionSigLIP → PCA 256 (같은 공간), `catalog.parquet` 행 순서 |
| dev_subset v2 | `dev_subset/` | 로컬 개발 | Amazon v2 상위 20개 브랜드 × 인기 상품 100 + H&M 인기 상품 100, images.zip `{source}/{item_id}.jpg` |

## 처리 규칙 (v2)

- **핵심 패션만:** 여행 가방 · 신발 관리용품 · 쥬얼리 보관함 · 작업복 · 안전용품 · 부속품 · 코스튬 제외 (Amazon 약 9%). H&M은 의류 · 신발 · 액세서리 · 가방 상품군만
- **source:** 데이터셋 사이에 고객이 연결되지 않는다. item_id · user_id는 source 안에서만 유일 → (source, id)로 다룬다. amazon_v2 item_id는 v1 값 유지(이미지 · 원본 임베딩과 같은 id)
- **브랜드 정리:** `brand_norm`(대소문자 · 기호 · 공백 통일)이 같으면 같은 브랜드
- **평점(feedback):** 리뷰 = 구매로 보되 학습 신호를 평점별로 — 4 ~ 5점 +1.0, 3점 +0.5, 1 ~ 2점 −1.0, H&M 구매 +1.0 (`rating` 원값 보존)
- **가격(price_source):** `real` 실제 · `imputed` 카테고리(대상 × 종류) 중앙값 · `scaled` H&M 정규화 가격을 Amazon 같은 종류 실제 가격 분포의 같은 순위 값으로 환산. 화면에서 real 외에는 '예상가'
- **분할 고정(split):** 데이터셋별 마지막 7일 test, 그 전 7일 valid, 나머지 train. 모든 모델 평가는 이 분할만 쓴다
- **H&M product_code:** 같은 옷의 색상 변형 묶음 (결과 중복 묶기에 사용)

## 로컬 백엔드에 넣기

1. 드라이브 `msrs_project/dev_subset/` 전체를 저장소 `data/dev_subset/`로 (임베딩까지 넣으려면 `embeddings/v2/`도 `data/embeddings/v2/`로) 내려받는다
2. 적재

```bash
# Docker
docker compose exec api python -m scripts.load_dev_subset --data /data/dev_subset [--embeddings /data/embeddings/v2] [--reset]
# 로컬 (backend/ 에서)
python -m scripts.load_dev_subset --data ../data/dev_subset [--embeddings ../data/embeddings/v2] [--reset]
```

`--reset`은 모든 테이블을 지우고 다시 만든다. 임베딩 열은 PR #14(pgvector)가 병합된 뒤에 채워진다.

## 데이터 특성 (Amazon 하위 집합)

- 브랜드가 매우 분산: 상위 100개 브랜드가 이벤트의 21%, 브랜드의 50%(11,567개)는 상품 1개 → 다중 브랜드 화면은 상위 50 ~ 100개 브랜드 + "기타"
- 브랜드 충성도가 약함: 고객별 최다 브랜드 비중 중앙값 0.17, 0.5 이상 1.3% → 충성도는 "최다 브랜드 비중"으로 정의, 평가는 시뮬레이터 브랜드 충성형 페르소나로 보완
- 행동 로그 없음: Amazon은 리뷰, H&M은 구매만 → 조회 · 클릭 퍼널은 Retailrocket 통계로 시뮬레이터에서 생성
- 시기가 다름: Amazon 2021 ~ 2023, H&M 2020 → 시간 축은 데이터셋별로 다룬다

## 데이터셋 역할 · 이용 조건

| 데이터셋 | 역할 | 이용 조건 |
|---|---|---|
| Amazon Reviews 2023 — https://amazon-reviews-2023.github.io/ | 메인: 다중 브랜드 추천 · 검색 | 연구 목적 공개 데이터 【확인 필요】, 재배포 금지 |
| H&M Personalized Fashion Recommendations (Kaggle) | 논문 파이프라인 재사용, 한 브랜드로 편입 | 【확인 필요】 |
| DeepFashion | 이미지 검색 학습 · 평가 | 비상업 연구용 |
| Retailrocket | 시뮬레이터 행동 패턴 참고 | 【확인 필요】 |
