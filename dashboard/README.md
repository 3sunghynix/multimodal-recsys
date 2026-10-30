# dashboard — Streamlit 관리자 화면 v0.1 (김광석)

학교 미션의 Streamlit 대시보드(localhost:8501) 요구에 맞춘 첫 읽기 전용 구현이다. 사용자 쇼핑 데모(localhost:8000/app/)는 backend/frontend가 담당한다. 두 화면은 별도 서비스다.

## 구현한 범위

- **모델 실험실:** source·평가 파일·최대 4개 모델 선택, MAP/Recall/Coverage 등 기록 지표 비교, 막대 차트·상세 표·CSV 다운로드.
- 기본 수치는 `docs/project/오프라인결과.md` PR #23의 **문서 보고값 스냅샷**이다. 출처 커밋·반올림 여부를 표시한다. 실제 JSON 결과를 가져오면 `recsys.ume`와 `recsys.run_eval` 출력을 읽는다.
- 비교는 **동일 파일·source·split의 평가 집합** 안에서만 한다. 인기 기준값이 없거나 0이면 상대 개선율을 꾸며내지 않는다. raw 결과가 제공한 차이의 95% 구간만 표시한다.
- **시뮬레이터 결과:** `summary.json`을 읽어 생성·제외 고객, 페르소나별 건수·CTR을 표시한다. 로컬 합성 결과이며 서버 A/B가 아님을 명시한다. 실행 버튼은 비활성이다.
- **개요:** 사용자가 누를 때만 기존 `GET /health`, `GET /models`를 조회한다. 스냅샷 조회 시각과 모델 레지스트리를 읽으며 변경 요청을 보내지 않는다.
- **데이터 현황:** dev_subset parquet의 메타데이터 행 수를 읽는다. 원본 이벤트·고객 행을 표시하지 않는다.
- **A/B 테스트, 알림·재학습:** 후속 단계 안내. p-value·온라인 신뢰구간·작업 상태를 생성하지 않는다.

모델 교체·롤백·재학습·시뮬레이터 실행 컨트롤은 disabled다. 이름이 같은 오프라인 모델과 DB model_version_id가 같은 버전이라고 가정하지 않는다. 승인·이력·매핑을 합의한 후 별도 변경으로 연결한다.

## Docker 실행

저장소 루트에서:

```bash
mkdir -p simulator/runs results/ume
docker compose up --build -d dashboard
```

Safari에서 http://127.0.0.1:8501 을 연다. 전체 서비스는 `docker compose up --build -d`로 함께 시작한다. 기존 DB reset·재적재가 필요하지 않다. 첫 dashboard 빌드는 Streamlit 패키지를 내려받는다.

```bash
docker compose logs --tail=50 dashboard
docker compose stop dashboard
```

환경 설정:

| 변수 | Docker 기본값 | 의미 |
|---|---|---|
| API_BASE_URL | http://api:8000 | 읽기 전용 API 주소. 컨테이너 안 localhost는 API가 아니다 |
| OFFLINE_RESULTS_DIR | /results | 호스트 results/ume, 평가 JSON 읽기 전용 |
| SIM_RUNS_DIR | /runs | 호스트 simulator/runs, 로컬 실행 요약 읽기 전용 |
| DEV_DATA_DIR | /data/dev_subset | 개발 데이터 메타데이터 읽기 전용 |

결과 JSON을 `results/ume/`에 두거나 화면에서 직접 업로드한다. data/models의 가중치·원본 고객 데이터·대용량 추천 배열은 이 화면 입력이 아니다. 입력 JSON은 8MB 이하로 제한한다.

## Python 개발·테스트

Python 3.10 이상 가상환경, 저장소 루트:

```bash
python -m pip install -r dashboard/requirements-dev.txt
python -m pytest -q dashboard/tests
python -m streamlit run dashboard/app.py --theme.base=light
```

AppTest는 실제 앱을 실행해 데이터셋 전환·빈 선택·메뉴·미연결 상태를 확인한다. 모델 품질을 재학습하거나 지표를 재계산하지 않는다. 테스트는 외부 API 대신 가짜 응답을 사용한다.

## 과제 요구와 후속 범위

학교 문서에는 검색 MRR/NDCG, 추천 HitRate/Coverage, A/B 전환율·p-value·95% 구간이 요구된다. v0.1의 PR #23 보고값은 주로 MAP/NDCG/Recall@12·Recall@300이다. **Recall을 HitRate로 바꿔 표시하거나, 없는 K의 지표를 다른 K에서 추정하지 않는다.** 검색 평가 결과·과제 K의 지표·고객 의존성을 반영한 온라인 검정은 후속 연결이 필요하다.

프레임워크 선택·그림 배치·원본 파일 형식은 [실행·발표 메모](../docs/project/대시보드_실행기.md)에 정리했다.


## 리뷰 반영 — 원본 결과 우선 표시

기본 선택은 source별 results/ume/hm_test_pca.json, results/ume/amazon_test_pca.json이다. 없을 때만 해당 source의 문서 보고값을 사용한다. 파일은 있지만 source·split·형식이 잘못되면 오류를 표시하며 조용히 문서값으로 바꾸지 않는다. 원본 수치와 vs_popularity의 95% 구간을 표시한다.

두 파일은 Drive msrs_project/results/ume/에서 내려받아 저장소 results/ume/에 둔다. PR에는 포함하지 않는다. 여러 모델이 있으면 문서의 서비스 설정 모델을 기본 비교 후보로 먼저 배치한다.

Amazon에는 브랜드 일치도@12 카드를 추가한다. 퍼센트와 인기 대비 차이(%p)로 표시하며, 고객별 최다 브랜드 비중과 구분한다.
