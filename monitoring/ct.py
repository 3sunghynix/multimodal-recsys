"""CT 루프 (A3): 서버 지표 → runner 판정 → trainer 실행 → finish_training 기록. 운영 교체는 하지 않는다.

한 번(tick)에 하는 일
  1. GET /models 로 운영 중(active) 모델을 찾는다.
  2. 닫힌 집계 구간마다 GET /metrics/window(model_version_id=운영 모델) → observe(): CTR · HitRate 알림만 기록.
     모델이 바뀌면 identity가 달라지므로 스트림(streams)을 따로 둔다.
  3. GET /metrics/log-delta?since_event_id=<마지막 학습 이벤트> → 새 비 setup 로그가 임계값 이상이면 재학습 요청.
     기준점 = max(상태에 저장된 값, 운영 모델 params.trained_until_event_id) — trainer와 같은 카운터.
  4. 요청이 있으면 trainer_command(기본 docker compose --profile trainer run --rm -T trainer python -m recsys.retrain)를
     실행하고, 마지막에 출력되는 보고 JSON을 finish_training_report()로 기록한다 (MAP@12 ×0.98 · Recall ×0.90 규칙).
     실패하면 failed로 두고 trainer_max_attempts까지 다음 tick에 다시 시도한다.

python -m monitoring.ct --state simulator/runs/monitor_ct/monitor_state.json            # 한 번 (repo 루트에서)
python -m monitoring.ct --state simulator/runs/monitor_ct/monitor_state.json --watch    # interval_seconds마다
python -m monitoring.ct --state ... --dry-run                                           # trainer 명령만 기록, 실행 안 함

표준 라이브러리만 쓴다 (호스트에서 실행 — trainer는 docker compose로 띄운다).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from monitoring.runner import finish_training_report, observe

OPEN = ('requested', 'failed')


def http_api(base: str, timeout: float = 30):
    def get(path, params=None):
        url = base.rstrip('/') + path + ('?' + urllib.parse.urlencode(params) if params else '')
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    return get


def run_command(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return p.returncode, p.stdout, p.stderr


def parse_report(stdout: str) -> dict:
    """trainer는 학습 로그를 찍은 뒤 마지막에 보고 dict를 json.dumps(indent=2)로 출력한다 → 마지막 '{' 줄부터 읽는다."""
    lines = stdout.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].rstrip() == '{':
            try:
                obj = json.loads('\n'.join(lines[i:]))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError('trainer report JSON not found in stdout')


def closed_windows(now: datetime, minutes: int, last_end: str | None, limit: int):
    """now 이전에 끝난 [start, end) 구간들. 처음이면 직전 구간 하나만 (과거를 거슬러 채우지 않음)."""
    step = timedelta(minutes=minutes)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    end = epoch + ((now - epoch) // step) * step
    start = datetime.fromisoformat(last_end) if last_end else end - step
    out = []
    while start + step <= end and len(out) < limit:
        out.append((start, start + step))
        start += step
    return out


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _open_job(state, max_attempts):
    for job in state.get('jobs', []):
        if job['status'] == 'requested' or (job['status'] == 'failed' and job.get('attempts', 0) < max_attempts):
            return job
    return None


def tick(state: dict, api, policy: dict, now: datetime, run=run_command, dry_run: bool = False) -> dict:
    state = json.loads(json.dumps(state))
    state.setdefault('alerts', []); state.setdefault('jobs', []); streams = state.setdefault('streams', {})
    max_attempts = policy.get('trainer_max_attempts', 2)

    active = [m for m in api('/models') if m.get('status') == 'active']
    if not active:
        state['last_note'] = 'no active model'
        return state
    model = active[-1]

    # 2. 지표 구간 → 알림 (학습 요청은 3에서 log-delta로)
    for start, end in closed_windows(now, policy.get('window_minutes', 60), state.get('last_window_end'),
                                     policy.get('max_windows_per_tick', 24)):
        w = api('/metrics/window', {'start': _iso(start), 'end': _iso(end), 'model_version': f"{model['name']}-{model['version']}",
                                    'source': 'all', 'cohort': policy.get('cohort', 'all'), 'model_version_id': model['id']})
        key = '|'.join(str(w[k]) for k in ('model_version', 'source', 'cohort', 'metric_definition_version'))
        before = len(streams.get(key, {}).get('alerts', []))
        streams[key] = observe(streams.get(key, {}), w, policy, request_jobs=False)
        state['alerts'] += [{**a, 'stream': key} for a in streams[key]['alerts'][before:]]
        state['last_window_end'] = _iso(end)
        # 대시보드 모니터링 페이지가 읽는 키 (runner 상태 파일과 같은 모양)
        state.update(identity=streams[key]['identity'], latest=streams[key]['latest'], stream=key)

    # 3. 새 로그 → 재학습 요청 (trainer와 같은 기준점)
    since = max(int(state.get('since_event_id', 0)), int(model.get('params', {}).get('trained_until_event_id', 0)))
    d = api('/metrics/log-delta', {'since_event_id': since})
    if d.get('reset_detected'):
        state['alerts'].append({'kind': 'log_counter_reset', 'since_event_id': since, 'max_event_id': d['max_event_id'],
                                'at': _iso(now)})
        since = int(d['max_event_id'])
        d = {**d, 'delta': 0}
    state['since_event_id'] = since
    state['latest_log_delta'] = {'since_event_id': since, 'delta': d['delta'], 'max_event_id': d['max_event_id'], 'at': _iso(now)}
    threshold = policy['training_log_threshold']
    if model['name'] != 'ume':
        state['last_note'] = f"active model {model['name']} is not ume — retraining not requested"
    elif d['delta'] >= threshold and _open_job(state, max_attempts) is None \
            and not any(j['status'] == 'failed' and j.get('since_event_id') == since for j in state['jobs']):
        state['jobs'].append({'version': f"candidate-{len(state['jobs']) + 1:06d}", 'status': 'requested',
                              'reason': 'new_nonsetup_log_threshold', 'since_event_id': since, 'logs_delta': d['delta'],
                              'max_event_id': d['max_event_id'], 'base_model_version_id': model['id'],
                              'requested_at': _iso(now), 'attempts': 0, 'active_model_unchanged': True})

    # 4. trainer 실행 → 결과 기록
    job = _open_job(state, max_attempts)
    if job is None:
        return state
    cmd = list(policy['trainer_command']) + ['--base', str(job['base_model_version_id']), '--min-new-logs', str(threshold),
                                             '--tolerance', str(policy.get('training_map_tolerance', 0.02)),
                                             '--recall-tolerance', str(policy.get('training_recall_tolerance', 0.10))]
    job['command'] = cmd
    if dry_run:
        job['dry_run'] = True
        return state
    job['attempts'] = job.get('attempts', 0) + 1
    job['started_at'] = _iso(now)
    code, out, err = run(cmd)
    try:
        if code != 0:
            raise RuntimeError(f'trainer exit {code}')
        report = parse_report(out)
        version = job['version']
        state = finish_training_report(state, version, report, policy.get('trainer_path_map'), policy)
        job = next(j for j in state['jobs'] if j['version'] == version)
        if job.get('trained_through_event_id') is not None:
            state['since_event_id'] = max(state['since_event_id'], int(job['trained_through_event_id']))
        elif report.get('since_event_id') is not None:
            state['since_event_id'] = max(state['since_event_id'], int(report['since_event_id']))
    except (RuntimeError, ValueError) as e:
        job.update(status='failed', error=str(e), stderr_tail=(err or '')[-2000:])
        if job['attempts'] >= max_attempts:
            state['alerts'].append({'kind': 'trainer_failed', 'version': job['version'], 'error': str(e), 'at': _iso(now)})
    return state


def save(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--api', default='http://localhost:8000')
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--policy', type=Path, default=Path(__file__).with_name('policy.json'))
    p.add_argument('--watch', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='trainer 명령을 기록만 하고 실행하지 않음')
    a = p.parse_args()
    policy = json.loads(a.policy.read_text(encoding='utf-8'))
    api = http_api(a.api)
    while True:
        state = json.loads(a.state.read_text(encoding='utf-8')) if a.state.exists() else {}
        updated = tick(state, api, policy, datetime.now(timezone.utc), dry_run=a.dry_run)
        if updated != state:
            save(a.state, updated)
            print(json.dumps({'alerts': updated['alerts'][-3:], 'jobs': updated['jobs'][-1:],
                              'log_delta': updated.get('latest_log_delta')}, ensure_ascii=False), flush=True)
        if not a.watch:
            break
        time.sleep(policy['interval_seconds'])


if __name__ == '__main__':
    main()
