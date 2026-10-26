# 김광석 탐색적 실험 결과 및 복구 코드

원본 Colab 노트북을 세 실험으로 분리했다. **모든 결과는 탐색적**이며 사전 등록 확증으로 표현하지 않는다.

| 가설 | 지표·비교 | 차이 | 95% 고객 부트스트랩 구간 |
| --- | --- | ---: | --- |
| H_TextMention | 상품군 가중 구매 비율, 미언급−언급 C2−C4 | −0.013112 | [−0.020492, −0.006295] |
| H_TextMask | 구매 비율, 마스킹−원본 C2−C4 | +0.003260 | [−0.001102, +0.007368] |
| H_VisualConsistency | 층 조정 고객 평균, high−low C2−C4 | +0.006600 | [−0.009907, +0.022624] |

각 실험의 REPORT.md에 설계·표본·결과·한계·재현 절차를 기록했다. 집계 방식과 평가 표본이 다르므로 세 숫자를 직접 같은 효과 크기로 비교하지 않는다.

## 파일

- H_*/results/: ZIP에서 보존한 CSV/JSON 및 1,000개 부트스트랩 값(각 8KB 정도)
- H_*/settings/: 저장된 계획·설정·임계값
- H_TextMask/training_metadata/: 12회 학습 기록
- H_*/original_execution_log.txt: 원본 업로드 노트북의 텍스트 출력
- PROVENANCE.json: 원본 노트북/ZIP 해시, 셀 출처 및 33개 원본 파일 해시
- ../../notebooks/kwangsuk_*.ipynb: 실험별 복구 노트북 (실제 위치는 paper/notebooks/)

대용량 데이터·임베딩·추천 후보·고객별 원본 그룹 파일은 Drive에 남겨 둔다. 이 저장소의 결과 요약만으로 전체 예측 재채점은 불가능하다.

## 복구 내용

H_TextMention과 H_VisualConsistency의 분석 셀은 남아 있었다. H_TextMask의 임베딩·학습·평가 셀도 남아 있었지만 입력 준비 셀은 없었다. 해당 셀을 대화 및 settings/experiment_plan.json의 정규식으로 재구성했고 노트북에 명시했다.

새 노트북은 데이터 경로를 명시하고 원본 핵심 코드 커밋 e6b73dc4a8aee549b4310cf4f3266e373b8632ca를 사용한다. 실행 결과·횟수는 비워 두었으며, 원본 출력은 별도 로그로 보존했다. 관련 없는 팀 전체 분석·라벨 생성·다운로드 셀은 제외했다.

## 실행

1. 각 노트북을 Colab에서 열고 새 세션을 시작한다.
2. DRIVE_ROOT를 공유 데이터의 실제 경로에 맞춘다. manifest가 없으면 중단한다.
3. H_TextMention은 기존 A/main 예측을 읽는다.
4. H_TextMask는 기본 RUN_TRAINING=False로 기존 실험을 재채점한다. 새 전체 재현은 True로 바꾼다.
5. H_VisualConsistency는 기본 REBUILD_GROUPS=False로 저장된 고객 그룹을 읽는다. 그룹 재생성은 True로 바꾼다.
6. 새 결과는 My Drive/kwangsuk_reproductions의 새 폴더에 저장된다.

마스킹 재현 시 역사적 Parquet 파일의 바이트 동일성은 가정하지 않는다. 모델 다운로드의 현재 revision/라이브러리/GPU 차이 때문에 새 학습 결과가 완전히 같다고 보장하지 않는다. 원본 라이브러리 버전은 training_metadata에 기록되어 있다.

## 로컬 검증 (원본 데이터·GPU 없이)

저장소 루트에서:

```bash
python paper/experiment_results/kwangsuk/verify_results.py
python paper/experiment_results/kwangsuk/test_recovered_notebooks.py
```

NumPy와 pandas가 필요하다. 첫 명령은 해시·표 계산·95% 구간·학습 메타·노트북 구문을 확인한다. 둘째는 합성 데이터로 복구 셀을 실행해 고객 순서 정렬, 대비값 부호, 과거 이력 필터링, 중복 제거 및 부트스트랩 계산을 점검한다. 합성 표는 메모리 어댑터를 사용하므로 실제 Parquet 직렬화는 검사하지 않는다.

**검증 범위:** 원본 H&M 데이터나 대용량 예측 파일은 이번 제공 자료에 없어 전체 실험을 재채점하거나 GPU 재학습하지 않았다. 저장된 수치의 정합성과 복구된 계산 흐름을 검증했다.
