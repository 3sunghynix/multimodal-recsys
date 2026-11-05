# CTR·HitRate 모니터와 재학습 요청 기록

닫힌 집계 구간 JSON을 주기적으로 읽는다. DB에서 집계를 만드는 생산자와 실제 UME 학습기는 별도로 연결해야 한다. 현재 구현은 임계값 알림 → 재학습 요청·후보 버전 기록까지다.

```bash
python -m monitoring.runner --input monitoring/example_window.json --state simulator/runs/monitor_demo/monitor_state.json
python -m monitoring.runner --input monitoring/example_window.json --state simulator/runs/monitor_demo/monitor_state.json --watch
```

두 번째 명령은 policy.json의 interval_seconds(60초)마다 확인한다. 같은 window_id는 한 번만 반영한다. 실제 데이터 생산자가 다음 닫힌 구간으로 input을 갱신해야 한다. example_window는 기능 확인용이며 실제 운영 수치가 아니다.

CTR=클릭 수/노출 수. HitRate=정해진 K에서 정답을 1개 이상 맞힌 고객 수/평가 고객 수. K·정답·고객 집합·노출/클릭 중복 처리 정의는 metric_definition_version에 고정한다. 조회 로그만으로 정답 HitRate를 만들지 않는다. 초기 이력(is_setup)은 분모와 새 로그 수에서 제외한다.

상태 파일 하나는 모델 버전/source/cohort/지표 정의 한 조합에만 쓴다. 구간은 시간대가 있는 시작·종료 시각이며 서로 겹치면 거절한다. 한 파일에 여러 프로세스가 쓰지 않는다. 노출 1,000건·평가 고객 100명 이상에서 CTR 2% 또는 HitRate 5% 미만이 3구간 연속이면 운영 알림을 기록한다. 임계값은 팀 합의 전 제안이고 통계적 유의성 판정이 아니다.

새 non-setup 로그가 직전 요청 이후 10,000건 이상이면 requested 후보를 기록한다. 같은 구간 재실행으로 요청을 중복 생성하지 않는다. 실제 학습은 실행하지 않는다. 외부 학습 완료 후 finish_training()에 산출물·동일 고정 평가의 후보/기준 MAP@12·평가 식별자를 전달하면 파일 해시와 버전을 기록한다. 기준보다 나쁘면 rejected, 그렇지 않으면 candidate_pending_review다. 운영 모델은 자동으로 교체하지 않는다.

후속: 서버 지표 생산자, 학습 worker, 실제 레지스트리 ID 연결, 실패 재시도·승인·롤백·변경 이력.
