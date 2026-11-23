# CTR·HitRate 모니터와 재학습 요청 기록

닫힌 집계 구간을 주기적으로 읽어 임계값 알림 → 재학습 요청 → trainer 실행 → 후보 버전 기록까지 한다. 운영 모델 교체는 하지 않는다.

- `monitoring/runner.py`: 판정 로직 (`observe` · `finish_training` · `finish_training_report`). 파일 입력(`--input`)으로도 돈다.
- `monitoring/ct.py`: 서버 연결 루프 (A3). `/models` · `/metrics/window` · `/metrics/log-delta`를 읽고, 재학습이 필요하면 trainer를 실행해 결과를 기록한다.

## 서버 연결 CT 루프 (`monitoring.ct`)

repo 루트에서, api가 떠 있는 상태로 실행한다 (호스트 python 표준 라이브러리만 사용, trainer는 docker compose로 띄움).

```bash
python -m monitoring.ct --state simulator/runs/monitor_ct/monitor_state.json --dry-run   # trainer 명령만 기록
python -m monitoring.ct --state simulator/runs/monitor_ct/monitor_state.json             # 한 번
python -m monitoring.ct --state simulator/runs/monitor_ct/monitor_state.json --watch     # interval_seconds(60초)마다
```

| 단계 | 내용 |
|---|---|
| 운영 모델 | `GET /models`에서 status=active. UME가 아니면 재학습을 요청하지 않는다 (`last_note`) |
| 알림 | `window_minutes`(60분) 단위 닫힌 구간마다 `GET /metrics/window?model_version_id=<운영 모델>` → `observe(request_jobs=False)`. 처음 실행하면 직전 구간 하나만 읽고, 밀린 구간은 tick당 `max_windows_per_tick`까지 채운다. 모델이 바뀌면 스트림(`streams`)을 따로 둔다 |
| 재학습 요청 | `GET /metrics/log-delta?since_event_id=S&types=click,add_to_cart,purchase`, S = max(상태 값, 운영 모델 `params.trained_until_event_id`). **학습에 쓰는 타입(`training_event_types`)만** 센다 — 노출 view(세션마다 10개)가 임계값을 채우지 않게. 새 로그 ≥ `training_log_threshold`(1만)이고 진행 중인 요청이 없으면 `requested`. `reset_detected`면 알림 후 기준점을 다시 잡는다 |
| 학습 | `trainer_command` + `--base <운영 모델 id> --min-new-logs <임계값> --tolerance 0.02 --recall-tolerance 0.10`. 출력 마지막의 보고 JSON을 `finish_training_report()`로 기록 |
| 결과 | `candidate_pending_review`(trainer가 후보 등록, `registered_version_id`) · `rejected`(학습했지만 채택 데이터셋 없음) · `skipped`(trainer가 조건 미달로 건너뜀) · `failed`(종료 코드 ≠ 0 또는 보고 없음 → 다음 tick에 재시도, `trainer_max_attempts`(2)회 후 `trainer_failed` 알림, `trainer_retry_cooldown_minutes`(60분) 뒤 새 요청). 학습했으면 기준점을 `max_event_id`로 올려 같은 로그로 다시 학습하지 않는다 |
| 산출물 | 채택된 데이터셋마다 `text/weights.npz` · `image/weights.npz` sha256. trainer 컨테이너 경로 `/data/...`는 `trainer_path_map`으로 호스트 `data/...`로 바꾼다 |

**카운터와 trainer의 관계.** trainer는 운영 모델의 `trained_until_event_id`부터 센다. CT 루프는 후보가 거절돼도 기준점을 `max_event_id`로 올리므로(같은 로그로 반복 학습 방지), 첫 거절 이후에는 CT 카운트 ≤ trainer 카운트다. trainer의 `--min-new-logs` 확인은 항상 통과하므로 동작에는 문제가 없다. 후보를 교체하면 운영 모델의 `trained_until_event_id`가 커져 두 값이 다시 맞는다. trainer가 학습 타입만 세도록 바뀌기 전까지는 trainer 쪽 숫자가 view를 포함해 더 크다.

**복구.** 손으로 고칠 필요는 없다: trainer 실패는 60분 뒤 자동으로 다시 요청된다. 바로 다시 돌리려면 상태 파일에서 마지막 `failed` job의 `failed_at`을 지우거나 `python -m monitoring.ct --state ...`를 새 상태 파일로 실행한다 (기준점은 운영 모델 `trained_until_event_id`에서 다시 시작).

채택 규칙은 trainer(`recsys.retrain.accept`)와 같다: **새 MAP@12 ≥ 기준 × (1 − 0.02) 그리고 새 Recall@20 ≥ 기준 × (1 − 0.10)**. runner가 데이터셋마다 다시 계산해 trainer 판정과 다르면 `rule_mismatch: true`를 남긴다 (두 함수가 같은 결과를 내는지 테스트로 고정). 허용 범위는 `policy.json`의 `training_map_tolerance` · `training_recall_tolerance`.

상태 파일은 대시보드 모니터링 페이지(`simulator/runs/*/monitor_state.json`)가 그대로 읽는다. 후보 확인 · 교체는 같은 페이지의 '모델 버전 · 재학습 후보'에서 한다.

## 파일 입력 runner (`monitoring.runner`)

```bash
python -m monitoring.runner --input monitoring/example_window.json --state simulator/runs/monitor_demo/monitor_state.json
python -m monitoring.runner --input monitoring/example_window.json --state simulator/runs/monitor_demo/monitor_state.json --watch
```

두 번째 명령은 policy.json의 interval_seconds(60초)마다 확인한다. 같은 window_id는 한 번만 반영한다. 실제 데이터 생산자가 다음 닫힌 구간으로 input을 갱신해야 한다. example_window는 기능 확인용이며 실제 운영 수치가 아니다.

CTR=클릭 수/노출 수. HitRate=정해진 K에서 정답을 1개 이상 맞힌 고객 수/평가 고객 수. K·정답·고객 집합·노출/클릭 중복 처리 정의는 metric_definition_version에 고정한다. 조회 로그만으로 정답 HitRate를 만들지 않는다. 초기 이력(is_setup)은 분모와 새 로그 수에서 제외한다.

상태 파일 하나는 모델 버전/source/cohort/지표 정의 한 조합에만 쓴다. 구간은 시간대가 있는 시작·종료 시각이며 서로 겹치면 거절한다. 한 파일에 여러 프로세스가 쓰지 않는다. 노출 1,000건·평가 고객 100명 이상에서 CTR 2% 또는 HitRate 5% 미만이 3구간 연속이면 운영 알림을 기록한다. 임계값은 팀 합의 전 제안이고 통계적 유의성 판정이 아니다.

새 non-setup 로그가 직전 요청 이후 10,000건 이상이면 requested 후보를 기록한다. 같은 구간 재실행으로 요청을 중복 생성하지 않는다. 실제 학습은 실행하지 않는다. 외부 학습 완료 후 finish_training()에 산출물·동일 고정 평가의 후보/기준 MAP@12(+ 선택: Recall@k)·평가 식별자를 전달하면 파일 해시와 버전을 기록한다. trainer와 같은 규칙(MAP@12 2% · Recall 10% 허용)을 통과하면 candidate_pending_review, 아니면 rejected다. 운영 모델은 자동으로 교체하지 않는다.

후속: 데이터셋별 지표 구간(`/metrics/window`가 지금은 source=all만 지원), 상태 파일 대신 DB 기록, 롤백 · 변경 이력, 고정 허용 범위 대신 부트스트랩 구간 하한 판정.
