# simulator — 페르소나 기반 사용자 시뮬레이터 (담당: 김광석)

실제 사용자가 없으므로 A/B 테스트 · 재학습 자동화는 시뮬레이터가 만든 로그로 돌아간다.

흐름: 시뮬레이션 고객(페르소나 부여) → `GET /recommend/{user_id}` → 노출 상품마다 `POST /events`(view) → 페르소나 규칙에 따라 click · add_to_cart · purchase 이벤트 전송. 이벤트에 `experiment_id`, `variant`, `model_version_id`를 함께 보내야 A/B 지표가 집계된다 (`GET /experiments/{id}/metrics`).

## 서버 연결 2단계 (A1, v1) — `simulator.server_run`

`python -m simulator.server_run` 은 **실제 서버 API** 를 돌아 시뮬레이터 → 추천 → 이벤트 → 서버 DB 한 바퀴를 완성한다. `catalog_pilot.py` (서비스 함수 직접 호출 · rollback) 과 달리 **서버 로그가 영구 저장**되므로 재학습 · 대시보드 · A/B 집계가 이 로그로 돈다.

전제: 이벤트연결규약 (PR #54 ~ #59) 서버 구현 완료 — `POST /users` 멱등, `GET /recommend` 가 `request_id` · `position` 포함, `POST /events/batch` 가 `request_id` 자동 채움 · `event_id` 멱등.

```bash
# 소규모 end-to-end 확인 (약 1초) — 기본 --base-url 은 http://localhost:8000
python -m simulator.server_run --base-url http://localhost:8000 --customers 10 --sessions 2 --k 5 --run-id dev

# 대량 실행 (고객 1만 · 로그 100만 급)
python -m simulator.server_run --base-url http://... --customers 10000 --sessions 10 --k 10 \
                               --click-rate 0.10 --purchase-rate 0.02 --run-id prod-$(date +%s)
```

로컬 포트가 8000 대신 8001 로 매핑돼 있으면 (`docker-compose.override.yml` 참고) `--base-url http://localhost:8001` 로.

- **결정론**: `source_user_id` · `session_id` · `event_id` 모두 `uuid5(run_id, 인덱스...)` 로 생성. 같은 `--run-id` 로 재실행하면 같은 고객 · 같은 세션이 만들어진다.
- **`run_id` 길이 제한**: `session_id = {run_id}:c{customer_idx}:s{session_idx}` 가 서버 제한 **64자** 를 넘지 않아야 한다 (안 그러면 `/events` 전부 422). 시뮬레이터가 실행 시작 전에 미리 검증 — 넘으면 바로 에러로 멈춘다. 참고: 고객 1만 · 세션 10 이면 접미사 9자, `--run-id` 는 **55자 이하** 로.
- **재실행은 지원 X**: 추천 호출 시 서버가 **새 `request_id`** 를 발급하므로, 같은 `event_id` + 다른 `request_id` → 톡방 결정 2 에 의해 **409 error**. 중간에 죽거나 대량 데이터가 더 필요하면 **새 `--run-id`** 로 처음부터 다시 돌린다 (이어서 돌리는 체크포인트 기능 없음). 네트워크 재시도 중단 복구는 batch 안에서만 작동.
- **A/B 참여**: 서버에 running 실험이 있으면 자동으로 세션 격리 + variant 배정. 시뮬레이터는 variant 선택 안 함.
- **행동 모델**: 지금은 최소 랜덤 — `click_rate` · `purchase_rate` 파라미터. persona · 카테고리 선호 등 풍부한 행동은 `engine.py` 를 `server_run.py` 에 교체 삽입하는 후속 PR (김광석 담당).

결과 `--out/summary.json`:
```json
{"customers": 10, "sessions": 2, "recommend_calls": 20, "events_created": 115, "events_error": 0, "elapsed_sec": 1.4}
```
`/metrics/window` 로 바로 집계 — views · clicks · CTR 이 노출 기준으로 나옴.

테스트: `python -m pytest simulator/tests/test_server_adapter.py -v` → 8 passed (mocked HTTP).

---

## 현재 구현 — 로컬 실행기 v0.1

`python -m simulator`는 **서버에 요청하지 않고 로컬 파일로만 저장**한다. 위 API 흐름은 후속 단계다. `personas.yaml` v0.3의 7종(트렌드세터·실용주의자·가성비추구·브랜드충성·충동구매·신중탐색·눈팅형)을 실행한다. 기본 로컬 파일럿은 700명을 12:18:18:12:8:17:15 비율로 생성한다. 실제 A/B 테스트 완료로 보고하지 않는다.

- 입력: items.parquet + events.parquet (`split`, `feedback`, source-qualified item_id 필수).
- `items.parquet`에 `category_l1`·`category_l2`·`category_l3`가 모두 있으면 `대분류 > 중분류 > 소분류`를 선호 카테고리로 사용한다. 셋 중 일부만 있으면 입력 오류다. 세 열이 없는 이전 데이터는 `cat_target > cat_type`으로 읽지만, 활성 페르소나의 선호 카테고리와 일치하지 않으면 실행을 멈추고 새 분류 열이 포함된 데이터를 요구한다.
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

## 실제 카탈로그 A/B 파일럿

`python -m simulator.catalog_pilot`은 전체 카탈로그 DB와 실행 중인 API로 인기(A) vs UME(B)를 비교한다. API 컨테이너 안에서 실행한다. 실행 방법 · 출력 · 해석 주의: [실제카탈로그_AB_파일럿.md](../docs/project/실제카탈로그_AB_파일럿.md)

## 서버 연결 준비 (읽기 전용)

`python -m simulator.server`는 API 준비 상태와 상품 ID 매핑을 기록한다.
기존 고객 추천을 명시적으로 조회할 수 있지만 고객 생성·초기 이력 적재·이벤트 전송은 하지 않는다.
실행 명령과 백엔드 계약 목록: [시뮬레이터 API 연결](../docs/project/시뮬레이터_API_연결.md).

## 별도 합성 규모·A/B 실험

`python -m simulator.scale --out simulator/runs/scale_001`은 기본값으로 합성 상품 50,000개·고객 10,000명과 100만 건 이상의 로컬 행동 로그를 생성한다. 기존 `python -m simulator`의 실제 데이터셋 기반 파일럿과 입력·결과를 섞지 않는다. 상품과 카테고리는 합성 데이터이며 서비스 카탈로그나 UME 모델 결과가 아니다.

규모 실험은 인기 전략 A와 카테고리 인기 전략 B를 고객 단위로 배정한다. A/B Z-test의 고객별 결과는 **첫 세션에서 한 번 이상 클릭했는지**이며, 전체 세션 클릭/노출 CTR은 별도 기술 통계다. `scale_summary.json`을 관리자 화면에서 확인할 수 있다. 이전에 만든 `scale_summary.json`은 지표 정의가 다르므로 새 버전으로 다시 생성한다.
