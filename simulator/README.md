# simulator — 페르소나 기반 사용자 시뮬레이터 (담당: 김광석)

실제 사용자가 없으므로 A/B 테스트 · 재학습 자동화는 시뮬레이터가 만든 로그로 돌아간다.

흐름: 시뮬레이션 고객(페르소나 부여) → `GET /recommend/{user_id}` → 노출 상품마다 `POST /events`(view) → 페르소나 규칙에 따라 click · add_to_cart · purchase 이벤트 전송. 이벤트에 `experiment_id`, `variant`, `model_version_id`를 함께 보내야 A/B 지표가 집계된다 (`GET /experiments/{id}/metrics`).

## 현재 구현 — 로컬 실행기 v0.1

`python -m simulator`는 **서버에 요청하지 않고 로컬 파일로만 저장**한다. 위 API 흐름은 후속 단계다. 기존 `personas.yaml` v0.2-draft의 4종(브랜드·가격·탐색·재구매)을 실행하며, 6종 실행 완료나 실제 A/B 테스트 완료로 보고하지 않는다.

- 입력: items.parquet + events.parquet (`split`, `feedback`, source-qualified item_id 필수).
- 초기 선호·구매 이력: **train의 긍정 구매(feedback > 0)** 상품만 사용한 합성 상태. 서로 다른 source의 원본 고객을 연결하지 않는다.
- 상품 선호 점수 → 클릭 확률 → 클릭한 경우 장바구니 → 장바구니에 넣은 경우 구매. 확률은 현실에서 추정하지 않은 설정값이다.
- 모델명·점수·variant는 행동 입력이 아니다. 고객/세션/순위/행동별 seed를 사용하며 초기 이력은 실행 중 고정한다.
- source마다 별도 train 마지막 날짜 기준 7일 인기도 특성도 계산·기록한다. 트렌드형은 여전히 비활성이다.
- 브랜드 비중은 `recsys.eval.online.dominant_brand_share`를 재사용한다. 새 오프라인 평가기를 만들지 않는다.
- 초기 이력은 customers.json에만 보존하며 시뮬레이션 이벤트에 섞지 않는다.

### 로컬 실행 (저장소 루트, Python 3.10 이상)

```bash
python -m pip install -r simulator/requirements.txt
python -m pytest -q simulator/tests
python -m simulator --data data/dev_subset --out simulator/runs/pilot_001
```

기존 출력 폴더는 덮어쓰지 않는다. 두 번째 실행은 pilot_002 등 새 이름을 사용한다. `simulator/runs/`는 Git에서 제외된다. 모델 학습·이미지 임베딩·GPU가 필요하지 않다.

**Mac 기본 Python이 3.9이면 실행하지 말고 Docker 방법을 사용한다.** Docker의 기존 api 이미지를 이용하는 명령은 [실행 가이드](../docs/project/시뮬레이터_실행기.md)에 있다. 실행기에 맞추어 Dockerfile이나 운영 API 설정을 바꾸지 않는다.

### 기본 실행의 추천 목록

기본값은 train 긍정 구매 건수를 source 안에서 평균 순위 백분위로 바꾼 **로컬 인기 목록**이다. 동점은 source/item_id로 결정한다. 전체 제공 카탈로그에서 뽑고 고객 선호 카테고리로 강제 필터하지 않는다. 상품 후보의 0건 항목도 포함한다.

이 목록은 **서버의 popularity 결과와 동일하다고 보장하지 않는다.** synthetic behavior의 기능 확인용이며, UME 비교나 서버 A/B 성과로 보고하지 않는다. 서비스 모델의 품질 비교에는 동일 조건에서 수집한 추천 목록을 명시적으로 넣어야 한다.

### 실제 추천 목록의 로컬 재생

다음 JSONL은 형식 예시다. customer_id는 같은 데이터·설정으로 생성된 customers.json의 ID, items는 `source:processed_item_id`다. **DB items.id나 source_item_id(ASIN)를 그대로 넣지 않는다.** backend 고객 생성/이력 적재·ID 매핑은 아직 이 실행기에 없다.

```json
{"customer_id":"sim-brand_loyal-0000","session_id":"s0000","recommendation_id":"saved-response-001","model":"ume","model_version_id":7,"experiment_id":null,"variant":null,"items":["amazon:93813","hm:12"]}
```

예시 상품이 실제 제공 카탈로그에 없으면 실행은 거절된다. local 목록의 귀속 정보가 없으면 세 필드를 모두 null로 둔다. 실제 실험 응답은 model_version_id·experiment_id·variant를 변형하지 않고 보존하며 같은 고객·실험의 그룹 변경을 거절한다. recommendation_id는 같은 고객의 모든 재생 모델에서 유일해야 한다. 현재 각 모델·고객에 한 세션만 허용한다.

```bash
python -m simulator --data data/dev_subset --recommendations saved_recommendations.jsonl --out simulator/runs/replay_001
```

같은 고객에게 서로 다른 모델의 목록을 재생하는 것은 **paired counterfactual replay**이고 서버의 무작위 A/B 배정 실험이 아니다. 공통 순위별 난수를 쓰므로 독립 표본도 아니다. 행동이 다른 목록의 순서에 어떻게 반응하는지 비교하는 탐색 실험으로만 사용한다.

### 출력과 분모

| 파일 | 내용 |
|---|---|
| manifest.json | 입력·설정·코드 체크섬, source별 기간, 버전, 설정 사본, 실행 모드 |
| customers.json | 합성 고객, 카테고리·브랜드·예산·초기 구매 이력 |
| recommendations.jsonl | 입력 추천 목록과 귀속 정보 원문 |
| decisions.jsonl | 노출별 특성·선호 점수·확률·난수·행동 결정 |
| events.jsonl | 실제 생성한 로컬 view/click/add_to_cart/purchase; 서버 전송 없음 |
| summary.json / SUMMARY.md | 페르소나·모델·귀속별 건수 및 기술 통계 |
| checksums.json | 산출물 체크섬 |

- CTR = click / view. 구매/노출 = purchase / view. 노출 0이면 null.
- 세션 장바구니율/구매율 = 한 번 이상 해당 행동이 있는 세션 수 / 응답 세션 수. 빈 응답도 세션 분모에 포함하고 별도 보고한다.
- 브랜드 비중은 view/click/purchase별 유효 고객 평균과 유효 고객 수를 각각 기록한다. 행동 없으면 null.
- source별 노출 수, real/imputed/scaled/unknown별 노출 수, 예상가 비중을 보존한다.
- 같은 페르소나의 유일한 인기 기준선과 CTR 차이(%p)·상대 상승률을 계산한다. 기준 CTR 0이면 상대 상승률은 null. 고객 집합 일치 여부를 함께 표시한다.
- 생성 시 제외된 고객, 응답 없는 고객, 비활성 페르소나, 빈 응답을 별도로 보고한다. 조건을 만족하는 카테고리가 없으면 제외하고 기록한다. 추출 후 다른 카테고리로 재시도하지 않는다.

### 남은 합의와 후속 작업

초기 이력 길이 10개(브랜드별 최소 한 상품, 최대 train 후보 수), 타임스탬프 UTC 해석은 실행을 위한 제안값이다. history-size는 CLI로 변경하고 manifest에 저장한다. 설정·데이터가 동일하면 고객과 행동이 재현된다. 다른 데이터의 고객 ID가 같다고 동일 선호 상태인 것은 아니다.

고객 생성, 초기 이력 주입, 추천 응답 조회, 노출 귀속 및 서버 중복 방지 계약이 확정되면 API adapter를 후속 PR로 추가한다. 로컬 UUID는 서버 중복 방지 구현이 아니다. 재학습·자동 승자 선정·대시보드는 이 PR 범위가 아니다.


### 리뷰 반영 — 카테고리 추출

train 긍정 구매 상품에서 브랜드별 최소 상품 수 조건을 만족하는 카테고리만 추린 뒤, 카테고리별 긍정 train 구매 이벤트 수에 비례해 선택한다. valid/test 구매 수는 쓰지 않는다. 후보 브랜드·구매 건수·확률·제외 카테고리를 manifest.json의 data.customer_category_sampling에 기록하며 고객에도 선택 확률을 보존한다.

기존 pilot_001(187명 생성·213명 제외)은 보존하고 새 폴더로 실행한다. 초기 고객 분포가 달라지므로 이전·새 실행의 CTR 차이를 모델 개선으로 해석하지 않는다. 현재 dev_subset 별도 검증에서 400명 생성·0명 제외였다. 다른 데이터에서도 항상 전원 생성된다는 보장은 없다.

서버 연결은 다음 PR: items.parquet의 (source, item_id) → source_item_id → DB의 (source, source_item_id)에 대응하는 items.id로 변환한다. initial_purchase_keys의 서버 이력 주입도 고객 생성 API와 함께 정하며, 준비용 구매 이력은 실험의 새 구매 전환에 포함하지 않도록 계약을 맞춘다.
