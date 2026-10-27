# simulator — 페르소나 기반 사용자 시뮬레이터 (담당: 김광석)

실제 사용자가 없으므로 A/B 테스트 · 재학습 자동화는 시뮬레이터가 만든 로그로 돌아간다.

흐름: 시뮬레이션 고객(페르소나 부여) → `GET /recommend/{user_id}` → 노출 상품마다 `POST /events`(view) → 페르소나 규칙에 따라 click · add_to_cart · purchase 이벤트 전송. 이벤트에 `experiment_id`, `variant`, `model_version_id`를 함께 보내야 A/B 지표가 집계된다 (`GET /experiments/{id}/metrics`).

## 정할 것

- 페르소나 6종 정의 (김유환과 합의): 예) 브랜드 충성형, 가격 민감형, 트렌드 추종형, 탐색형, 신상품 선호형, 재구매형 — `personas.yaml`
- 페르소나별 클릭 · 구매 확률 모델 (상품 속성 · 브랜드 · 가격에 대한 선호)
- 시뮬레이션 규모 · 속도, 재현용 seed
