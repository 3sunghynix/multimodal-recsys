# 데이터

공용 데이터는 구글 드라이브 `msrs_project/`(팀 공유, 바로가기 → `/content/drive/MyDrive/msrs_project`)에 있다. 폴더 구조 · 규칙은 드라이브의 `README.md`, 파일 목록 · 체크섬은 `manifest.json`.
저장소의 `data/`에는 로컬 개발용 데이터만 내려받는다 (git 제외: `data/raw/`, `data/processed/`, `data/dev_subset/`, `data/embeddings/`). 키 파일(kaggle.json 등)은 저장소 · 드라이브에 올리지 않는다.

## 현황 (2026-10-29, v2 적재 완료)

| 데이터 | 드라이브 위치 | 역할 | 규모 |
|---|---|---|---|
| Amazon Reviews 2023 — Clothing_Shoes_and_Jewelry 5-core | `processed/amazon_v2`, `images/amazon` | 서비스 카탈로그 (다중 브랜드) | 상품 86,827 · 브랜드 19,808 · 고객 241,525 · 이벤트 2,080,580 (2021-10-01 ~ 2023-09-12). 패션 외 7.9% 제외, 실제 가격 95%, 평점 1 ~ 2점 9.6% |
| H&M (Kaggle) | `processed/hm`, `images/hm` | 서비스 카탈로그 (단일 브랜드 "H&M") | 상품 19,998(옷 묶음 10,161) · 고객 245,768 · 이벤트 2,773,076 (2020-06-24 ~ 09-22), 의류 87% |
| Retailrocket (Kaggle) | `raw/retailrocket`, `processed/retailrocket/sim_stats.json` | 시뮬레이터 행동 기준값 (카탈로그 아님) | 조회 → 장바구니 2.5%, 장바구니 → 구매 32%, 재구매 고객 22%, 상위 1% 상품이 조회의 23% |
| DeepFashion (In-shop) | 예정 | 이미지 검색 학습 · 평가 | 공식 신청 필요 |
| 통합 임베딩 | `embeddings/v2` | 서비스 검색 | Amazon v2 86,827 + H&M 19,998, 텍스트 MiniLM · 이미지 Marqo-FashionSigLIP → PCA 256, 설명 분산 0.984 · 0.906 |
| dev_subset v2 | `dev_subset/` | 로컬 개발 | 브랜드 21 · 상품 1,990 · 고객 6,000 · 이벤트 30,255 (Amazon 23,296 · H&M 6,959), images.zip 35MB `{source}/{item_id}.jpg` |

## 처리 규칙 (v2)

- **핵심 패션만:** 여행 가방 · 신발 관리용품 · 쥬얼리 보관함 · 작업복 · 안전용품 · 부속품 · 코스튬 제외 (Amazon 약 9%). H&M은 의류 · 신발 · 액세서리 · 가방 상품군만
- **source:** 데이터셋 사이에 고객이 연결되지 않는다. item_id · user_id는 source 안에서만 유일 → (source, id)로 다룬다. amazon_v2 item_id는 v1 값 유지(이미지 · 원본 임베딩과 같은 id)
- **브랜드 정리:** `brand_norm`(대소문자 · 기호 · 공백 통일)이 같으면 같은 브랜드
- **평점(feedback):** 리뷰 = 구매로 보되 학습 신호를 평점별로 — 4 ~ 5점 +1.0, 3점 +0.5, 1 ~ 2점 −1.0, H&M 구매 +1.0 (`rating` 원값 보존)
- **가격(price_source):** `real` 실제 · `imputed` 카테고리(대상 × 종류) 중앙값 · `scaled` H&M 정규화 가격을 Amazon 같은 종류 실제 가격 분포의 같은 순위 값으로 환산. 화면에서 real 외에는 '예상가'
- **분할 고정(split):** 모든 모델 평가는 이 분할만 쓴다. H&M은 마지막 7일 test(2020-09-16 ~ 22) · 그 전 7일 valid. Amazon은 끝부분이 희소해(9월 주당 수백 건 → 마지막 주 2건) 시간 순 이벤트 마지막 5% test(2023-03-27 ~ 09-12, 10.4만 건) · 그 전 5% valid. Amazon test 고객의 36%는 train 이력이 없어 추천 평가는 이력 있는 고객 기준
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
- 인기 편중: 상품당 이벤트가 H&M 약 139건, Amazon 약 24건 → 인기 기반 추천 · 학습은 source별로 계산
- **두 데이터셋이 임베딩 공간에서 따로 모임:** 상품 이웃 중 다른 데이터셋 비율 0.1 ~ 0.5%(무작위면 19% · 81%), 일반 검색어 상위 20개 중 H&M 51%(카탈로그 19%). 데이터셋별 평균 맞춤으로도 2 ~ 6%로 부족 → 검색은 **데이터셋별로 찾고 점수를 표준화해 합친다**(UME와 같은 원리). 텍스트 검색은 Marqo 텍스트 인코더로 상품 이미지를 찾는 방식과 비교 예정

## 데이터셋 역할 · 이용 조건

| 데이터셋 | 역할 | 이용 조건 |
|---|---|---|
| Amazon Reviews 2023 — https://amazon-reviews-2023.github.io/ | 메인: 다중 브랜드 추천 · 검색 | 연구 목적 공개 데이터 【확인 필요】, 재배포 금지 |
| H&M Personalized Fashion Recommendations (Kaggle) | 논문 파이프라인 재사용, 한 브랜드로 편입 | 【확인 필요】 |
| DeepFashion | 이미지 검색 학습 · 평가 | 비상업 연구용 |
| Retailrocket | 시뮬레이터 행동 패턴 참고 | 【확인 필요】 |
