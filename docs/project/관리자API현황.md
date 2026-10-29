# 관리자 페이지 API 현황 (기초안 v0.1)

> 김광석의 관리자 페이지 설계 · 김형우와의 API 합의를 위한 출발점. 화면 구성은 김광석이 확정하며, 아래 "필요한 것"은 예시다.
> 기준: 확인한 upstream main e2c657b (PR #19 평가·기준선, #20 통합 검색 반영 시점). 문서: `docker compose up` 후 http://localhost:8000/docs

## 1. 지금 있는 API

| 메서드 · 경로 | 하는 일 | 관련 화면 |
|---|---|---|
| `GET /models` | 등록 모델 목록 (이름 · 버전 · 상태 active/candidate/retired · 파라미터) | 모델 실험실 |
| `GET /models/available` | 코드에 등록된 추천 모델 종류 | 모델 실험실 (등록 폼) |
| `POST /models` | 모델 버전 등록 | 모델 실험실 |
| `POST /models/{id}/activate` | 운영 모델 교체 (기존 active → candidate) | 모델 실험실 · **교체 버튼** |
| `POST /experiments` | A/B 실험 생성 (그룹별 모델 · 비율). 진행 중인 실험은 자동 중지 | A/B 테스트 |
| `GET /experiments` | 실험 목록 | A/B 테스트 |
| `POST /experiments/{id}/stop` | 실험 중지 | A/B 테스트 |
| `GET /experiments/{id}/metrics` | 그룹별 고객 수 · 노출(view) · 클릭 · 구매 · CTR (누적) | A/B 테스트 |
| `GET /events/count` | 이벤트 종류별 전체 개수 | 데이터 현황 |
| `GET /recommend/{user_id}` | 고객별 추천, 서버 A/B 배정 정보 반환 | 시뮬레이터·서비스 |
| `POST /search` (#20) | q·선택 file·text_weight·k로 통합 검색. query_en·untranslated·items 반환 | 서비스 검색 |
| `GET /search` · `POST /search/image` | 단순 키워드·기존 이미지 전용 경로. 통합 검색 화면은 POST /search 사용 | 기존 호환 경로 |

오프라인 평가: #19의 `recsys/eval`·`recsys/run_eval.py`는 main에 있다. 관리자 평가 작업·결과 저장 API가 구현됐다는 뜻은 아니다. 이 PR의 reference는 검증용이며 별도 운영 평가를 만들지 않는다.

DB에는 있지만 API가 없는 것: `metric_snapshots` 테이블(지표 기록).

## 2. 화면별로 지금 가능한 것 / 부족한 것

| 화면 | 지금 가능 | 부족한 것 (추가 API 예시) |
|---|---|---|
| 개요 | 운영 모델(`GET /models`), 진행 중 실험(`GET /experiments`) | 핵심 지표 카드 · 최근 알림 → 아래 지표 · 알림 API |
| 모델 실험실 | 모델 목록 · 등록 · 교체 | 모델별 오프라인 지표 조회 `GET /models/{id}/metrics` (metric_snapshots), 지표 기록 `POST /metrics/snapshots` (평가 스크립트가 호출), 교체 이력 · 롤백 (교체 이력 테이블 + `POST /models/{id}/rollback` 또는 이력의 이전 모델 activate) |
| A/B 테스트 | 생성 · 중지 · 누적 그룹 지표 | 지표 추이 `GET /experiments/{id}/metrics/timeseries?interval=hour\|day`, 신뢰구간 · 판정 결과, "승자 적용"(= 실험 중지 + 승자 모델 activate 한 번에) |
| 알림 · 재학습 | 없음 | 알림 규칙 `GET/PUT /alert-rules`, 알림 이력 `GET /alerts`, 새 로그 수 `GET /retrain/status`, 재학습 실행 · 이력 `POST /retrain/jobs` · `GET /retrain/jobs` (학습은 Colab에서 하므로 "요청 → 결과 등록" 방식인지 서버에서 직접 돌릴지 결정 필요) |
| 시뮬레이터 | 없음 | 실행 `POST /simulator/runs`(페르소나 비율 · 고객 수 · 실험 id), 상태 `GET /simulator/runs/{id}`, 중지 (시뮬레이터는 별도 프로세스로 API를 호출하는 구조 — 김유환 작업) |
| 데이터 현황 | 이벤트 종류별 개수 | source별 상품 · 브랜드 수, 상위 브랜드, 일별 이벤트 유입 `GET /stats/catalog` · `GET /stats/events?interval=day` |

## 3. 결정이 필요한 것 (김광석 · 김형우)

- 지표 추이를 이벤트에서 그때그때 계산할지, 주기적으로 `metric_snapshots`에 쌓을지
- 재학습을 어디서 돌릴지: 서버(CPU, 느림) vs Colab(수동 · 반자동) → "재학습 자동화"의 범위
- 알림 전달 방식: 화면 표시만 vs 슬랙 · 메일
- 관리자 인증: 데모 범위에서는 없음으로 할지
