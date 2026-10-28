# 데이터

공용 데이터는 구글 드라이브 `msrs_project/`(팀 공유, 바로가기 → `/content/drive/MyDrive/msrs_project`)에 있다. 폴더 구조 · 규칙은 드라이브의 `README.md`, 파일 목록 · 체크섬은 `manifest.json`.
저장소의 `data/`에는 로컬 개발용 데이터만 내려받는다 (git 제외: `data/raw/`, `data/processed/`, `data/dev_subset/`, `data/embeddings/`). 키 파일(kaggle.json 등)은 저장소 · 드라이브에 올리지 않는다.

## 현황 (2026-10-28)

| 데이터 | 드라이브 위치 | 상태 | 규모 |
|---|---|---|---|
| Amazon Reviews 2023 — Clothing_Shoes_and_Jewelry 5-core | `raw/amazon_clothing`, `processed/amazon`, `images/amazon`, `embeddings/amazon` | 적재 완료 | 상품 95,662 · 브랜드 23,183 · 고객 269,056 · 이벤트 2,326,938 (2021-10-01 ~ 2023-09-12), 이미지 95,653장 |
| dev_subset (로컬 개발용) | `dev_subset/` | 완료 | 상위 20개 브랜드 × 인기 상품 100개 = 상품 1,894 · 고객 5,000 · 이벤트 23,625 · images.zip 34MB |
| H&M | 논문 폴더 `msrs_paper/` 재사용 | 기존 | 상품 53,232 |
| DeepFashion (In-shop) | 예정 | 공식 신청 필요 | — |
| Retailrocket | 예정 | Kaggle | — |

Amazon 하위 집합 기준: 5-core 평점 → 2021-10-01 이후 → 인기 상품 상위 10만 → 5-core → 브랜드 · 대표 이미지 있는 상품만 → 5-core. 리뷰(평점)를 구매로 간주한다 (`events.type = purchase`, `rating` 포함).
콘텐츠 임베딩: 텍스트 all-MiniLM-L6-v2, 이미지 Marqo-FashionSigLIP → L2 → PCA 256 → L2 (논문과 동일). 질의 변환용 PCA 행렬(`pca_text.npz`, `pca_image.npz`) 포함.

## 로컬 백엔드에 넣기

1. 드라이브 `msrs_project/dev_subset/` 전체를 저장소 `data/dev_subset/`로 내려받는다 (임베딩까지 넣으려면 `embeddings/amazon/`도 `data/embeddings/amazon/`으로)
2. 적재

```bash
# Docker
docker compose exec api python -m scripts.load_dev_subset --data /data/dev_subset [--embeddings /data/embeddings/amazon] [--reset]
# 로컬 (backend/ 에서)
python -m scripts.load_dev_subset --data ../data/dev_subset [--embeddings ../data/embeddings/amazon] [--reset]
```

`--reset`은 모든 테이블을 지우고 다시 만든다. 임베딩 열은 PR #14(pgvector)가 병합된 뒤에 채워진다.

## 데이터 특성 (Amazon 하위 집합)

- 브랜드가 매우 분산: 상위 100개 브랜드가 이벤트의 21%, 브랜드의 50%(11,567개)는 상품 1개
- 브랜드 충성도가 약함: 고객별 최다 브랜드 비중 중앙값 0.17, 0.5 이상인 고객 1.3%
- → 다중 브랜드 화면은 상위 50 ~ 100개 브랜드 + "기타", 브랜드 충성도 평가는 시뮬레이터의 브랜드 충성형 페르소나로 보완

## 데이터셋 역할 · 이용 조건

| 데이터셋 | 역할 | 이용 조건 |
|---|---|---|
| Amazon Reviews 2023 — https://amazon-reviews-2023.github.io/ | 메인: 다중 브랜드 추천 · 검색 | 연구 목적 공개 데이터 【확인 필요】, 재배포 금지 |
| H&M Personalized Fashion Recommendations (Kaggle) | 논문 파이프라인 재사용, 한 브랜드로 편입 | 【확인 필요】 |
| DeepFashion | 이미지 검색 학습 · 평가 | 비상업 연구용 |
| Retailrocket | 시뮬레이터 행동 패턴 참고 | 【확인 필요】 |
