# 시뮬레이터 서버 연결 1단계 — 읽기 전용 점검

## 확인한 기준

팀 main `7dd498b` (2026-10-31) 기준이다. 기존 로컬 실행기와 행동 확률은 변경하지 않는다.

- GET /recommend/{user_id}: 존재. model_version_id · experiment_id · variant를 반환한다.
- POST /users: 없음.
- POST /events: 존재하지만 클라이언트 중복 방지 키가 없다.
- 추천 응답: request_id · impression_id가 없다.
- 초기 이력과 새 실험 행동을 구분하는 이벤트 필드가 없다.

따라서 이번 코드는 서버 연결의 준비 단계다. 고객 생성·초기 이력 적재·행동 이벤트 전송·A/B 실행은 포함하지 않는다.

## 왜 필요한가

parquet의 amazon:93813은 DB 상품 ID가 아니다. 다음 경로로 연결한다.

`(source, parquet item_id) → items.parquet의 source_item_id → DB의 (source, source_item_id) → DB items.id`

H&M source_item_id의 앞자리 0을 유지한다. source가 다르면 같은 번호도 다른 상품이다. 중복 키는 오류, 매핑되지 않은 상품은 목록으로 보고한다. 추천 응답에 매핑되지 않은 상품이 있으면 일부만 제거하거나 다른 상품으로 대체하지 않고 중단한다.

## 구현

- simulator/server.py: GET-only HTTP 클라이언트, OpenAPI 준비 상태 확인, 상품 ID 매핑, 선택적으로 기존 고객 추천 수집.
- backend/scripts/export_simulator_items.py: DB 상품 ID 대응을 JSON으로 내보낸다. DB 초기화나 쓰기·commit은 하지 않는다.
- 초기 이력 로딩이나 이벤트 전송 기능은 없다. OpenAPI에 필드가 있다고 중복 방지 동작이 검증된 것으로 취급하지 않는다.
- 기존 python -m simulator의 동작과 로컬 요약 형식은 그대로다.
- 추천 수집 결과는 진단 전용이다. 초기 이력 일치가 확인되지 않았으므로 기존 로컬 replay 입력과 다른 형식으로 저장하며, CTR·효과·승자 판정을 만들지 않는다.

## Mac Docker 실행 — 팀 main을 받은 새 작업 브랜치에서

Docker Desktop과 기존 API·DB가 실행 중이고 data/dev_subset/items.parquet가 준비돼 있어야 한다.
저장소 루트에서 실행한다. simulator/runs/는 Git 커밋 대상이 아니다.
기존 api 이미지를 다시 빌드하지 않아도 새 scripts를 read-only 마운트해서 실행한다.

```bash
mkdir -p simulator/runs

docker compose run --rm --no-deps \
  -v "$PWD/backend/scripts:/app/scripts:ro" \
  -v "$PWD/simulator/runs:/runs" \
  api python -m scripts.export_simulator_items --out /runs/db_items_api_001.json

docker compose run --rm --no-deps \
  -v "$PWD/simulator:/work/simulator:ro" \
  -v "$PWD/recsys:/work/recsys:ro" \
  -v "$PWD/simulator/runs:/runs" \
  -w /work api sh -c 'python -m pip install -r simulator/requirements.txt && python -m simulator.server --base-url http://api:8000 --data /data/dev_subset --db-items /runs/db_items_api_001.json --out /runs/api_preflight_001'
```

파일이 이미 존재하면 이름을 바꾼다. 기존 파일을 덮어쓰지 않는다.
반드시 연결할 API와 **동일 DB**에서 방금 만든 snapshot을 사용한다. snapshot은 live API와 DB가 동일하다는 인증서가 아니다.

### 산출물

api_preflight_001/ 안에:

- readiness.json: 조회 시각, API 기능 목록, 누락 기능, 매핑 건수·누락 상품, 입력 체크섬.
- item_mapping.json: parquet 상품 키 → DB 상품 ID.
- server_recommendations.jsonl: 기본 실행에서는 빈 파일. --bindings 옵션을 줄 때만 수집.
- checksums.json: 위 파일 무결성 확인용.

현재 main이면 create_user_post, event_dedup_key_advertised, recommendation_link_id_advertised, setup_event_marker_advertised가 누락으로 나오는 것이 예상 결과다. 이것은 실행 실패가 아닌 미완료 계약 보고다.

### 선택: 기존 고객 추천 응답만 수집

사용자가 명시한 기존 서버 고객에 한해서 --bindings JSON 파일을 줄 수 있다.
형식: [{"customer_id": "diagnostic-001", "server_user_id": 1}].
이 고객은 자동 생성되지 않으며 초기 이력 일치도 보장하지 않는다. 실제 고객 이력이나 합성 이력을 변경하지 않는다.
수집은 GET /recommend만 호출하고, raw 서버 응답과 변환 상품 키를 별도로 보존한다. 초기 이력 미검증 표시가 붙는다. 새로운 시뮬레이터 고객으로 재사용하지 않는다.

## 다음 PR 전에 형우·유환과 확정할 계약

1. 고객 생성: source=sim, 실행별 고유 source_user_id, 재시도 시 같은 고객 반환 또는 명확한 충돌 응답.
2. 상품 매핑: API로 source_item_id까지 제공하거나, 같은 DB에서 내보낸 snapshot을 쓰는 절차 합의.
3. 초기 이력: purchase로 개인화 이력에 반영하되 is_setup 또는 별도 초기화 API로 표시한다. 새 실험 전환·인기도·재학습 트리거 집계에 섞이지 않게 한다. 초기 순서·시각도 합의한다.
4. 추천 연결: request_id + 노출 상품별 impression_id, 고객·상품·순위·모델·실험·그룹을 서버에 연결한다.
5. 이벤트 중복 방지: event_id UUID의 DB 유일성, 같은 키 재전송의 멱등 응답, 다른 내용의 키 재사용 거절. 네트워크 timeout 재시도도 검증한다.
6. A/B 격리: sim 실행 ID와 persona를 집계 필터에 포함하고, 실제 사용자·설정용 이력과 분리한다. 배정·노출 분모와 반복 행동 단위도 합의한다.

별도 승인된 이벤트 전송 구현과 통합 테스트가 있어야 delivery_enabled를 켤 수 있다. 이번 코드에는 켜는 옵션이 없다.

## 검증

- simulator 전체 테스트 40개 통과 (기존 24 + 연결 준비 16).
- backend 전체 테스트 31개 통과. DB snapshot export의 읽기 전용·앞자리 0 보존·덮어쓰기 방지 검증 포함.
- HTTP 동작은 GET transport 대역과 API schema로 검사한다. 실제 사용자 노트북 Docker·실제 DB 실행은 별도로 확인한다.

## 발표용 설명

“시뮬레이터의 로컬 상품 번호와 서버 DB 상품 번호가 다르므로 source와 원본 상품 식별자를 이용한 대응을 구현했습니다. 서버 API 준비 상태와 누락 계약을 파일로 기록하며, 기존 고객의 추천 응답은 순서와 모델·실험 정보를 보존해 조회할 수 있습니다. 고객 생성·초기 이력·중복 방지 계약은 아직 없어, 이번 단계에서는 서버에 행동을 전송하거나 A/B 효과를 주장하지 않습니다.”
