# 데이터

원본은 `data/raw/`, 가공본은 `data/processed/`에 둔다 (둘 다 git 제외). 키 파일(kaggle.json 등)은 저장소 · 공유 드라이브에 올리지 않는다.

| 데이터셋 | 내용 | 역할 | 이용 조건 확인 |
|---|---|---|---|
| Amazon Reviews 2023 (Fashion 계열) — https://amazon-reviews-2023.github.io/ | 여러 브랜드, 이미지 · 텍스트 · 리뷰 이력 | **메인**: 다중 브랜드 추천 · 검색, 브랜드 충성도 | 【확인 필요】 |
| H&M Personalized Fashion Recommendations (Kaggle) | 단일 브랜드, 이미지 · 텍스트 · 구매 이력 | 논문 파이프라인 재사용 검증, 한 브랜드로 편입 | 【확인 필요】 |
| DeepFashion | 이미지 · 속성, 사용자 없음 | 이미지 검색 모델 학습 · 평가 | 【확인 필요】 (비상업 연구용) |
| Retailrocket | 조회 · 장바구니 · 구매 로그, 상품 속성은 해시 | 시뮬레이터 행동 패턴 참고 | 【확인 필요】 |

데이터셋 사이에는 고객이 연결되지 않는다. 모든 데이터는 하나의 상품 · 이벤트 테이블에 `source`, `brand` 열로 구분해 적재한다 (스키마: [docs/project/데이터스키마.md](../docs/project/데이터스키마.md)).

TODO (김유환): 통합 스키마 확정 → Amazon Fashion 하위 집합 선정 · EDA → 적재 스크립트(`backend/scripts/`)
