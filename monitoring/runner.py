"""Poll closed metric windows from JSON; persist alerts and local training requests.

Counters must exclude is_setup events. State is for one stream/model/data cohort.
No DB writes or automatic model activation. Server polling + trainer invocation live in monitoring.ct.
"""
import argparse
import json
import math
from pathlib import Path
import time


def integer(value,name):
    if type(value) is not int or value<0: raise ValueError(f'{name}: nonnegative integer required')
    return value


def observe(state, window, policy, request_jobs=True):
    """One closed, non-overlapping window per unique monotonic window_end.

    request_jobs=False: alerts only. monitoring.ct uses /metrics/log-delta for the training trigger
    (same counter as the trainer), so window-based requests would double-count."""
    from datetime import datetime
    state=json.loads(json.dumps(state))
    if not isinstance(window.get('window_id'),str) or not window['window_id']:
        raise ValueError('window_id required')
    stamp=datetime.fromisoformat(window['window_end'])
    if stamp.tzinfo is None: raise ValueError('window_end must contain timezone')
    start=datetime.fromisoformat(window['window_start'])
    if start.tzinfo is None or start>=stamp:raise ValueError('Require aware window_start < window_end')
    identity=[window[k] for k in ('model_version','source','cohort','metric_definition_version')]
    if state.get('identity',identity)!=identity:raise ValueError('Use a separate state for each model/source/cohort/definition')
    state['identity']=identity
    if state.get('last_window_id')==window['window_id']: return state
    if state.get('last_window_end') and stamp<=datetime.fromisoformat(state['last_window_end']):
        raise ValueError('Windows must arrive in chronological order')
    if state.get('last_window_end') and start<datetime.fromisoformat(state['last_window_end']):
        raise ValueError('Metric windows must not overlap')
    if window.get('setup_excluded') is not True:raise ValueError('Explicit setup_excluded=true required')
    views=integer(window['views'],'views');clicks=integer(window['clicks'],'clicks')
    users=integer(window['evaluated_users'],'evaluated_users');hits=integer(window['hit_users'],'hit_users')
    logs=integer(window['cumulative_new_logs'],'cumulative_new_logs')
    if clicks>views or hits>users:raise ValueError('Numerators exceed denominators')
    if logs<state.get('last_cumulative_logs',0):raise ValueError('Log counter decreased; use a new stream state')
    for key in ('minimum_views','minimum_users','consecutive_windows','training_log_threshold'):
        if type(policy[key]) is not int or policy[key]<=0:raise ValueError(f'{key}: positive integer required')
    for key in ('ctr_minimum','hitrate_minimum'):
        if not isinstance(policy[key],(int,float)) or not math.isfinite(policy[key]) or not 0<=policy[key]<=1:raise ValueError('Invalid threshold')
    values={'ctr':clicks/views if views else None,'hitrate':hits/users if users else None}
    eligible={'ctr':views>=policy['minimum_views'],'hitrate':users>=policy['minimum_users']}
    streak=state.setdefault('streak',{});alerts=state.setdefault('alerts',[])
    for metric,value in values.items():
        below=eligible[metric] and value is not None and value<policy[metric+'_minimum']
        streak[metric]=streak.get(metric,0)+1 if below else 0
        if streak[metric]==policy['consecutive_windows']:
            alerts.append({'window_id':window['window_id'],'metric':metric,'value':value,'threshold':policy[metric+'_minimum'],
                           'kind':'operational_threshold_not_significance_test'})
    jobs=state.setdefault('jobs',[])
    watermark=state.get('training_watermark',0)
    # First observation is measured since stream origin (counter starts at zero).
    if request_jobs and logs-watermark>=policy['training_log_threshold']:
        jobs.append({'version':f"candidate-{len(jobs)+1:06d}",'status':'requested',
                     'logs_through':logs,'window_id':window['window_id'],'active_model_unchanged':True,
                     'reason':'new_nonsetup_log_threshold'})
        state['training_watermark']=logs
    state.update(last_window_id=window['window_id'],last_window_end=window['window_end'],last_cumulative_logs=logs,
                 latest={'values':values,'eligible':eligible,'views':views,'evaluated_users':users})
    return state


def accept(candidate_map, baseline_map, candidate_recall=None, baseline_recall=None,
           map_tolerance=0.02, recall_tolerance=0.10):
    """Same rule as recsys.retrain.accept (kept dependency-free here; equality is tested):
    candidate MAP@12 >= baseline x (1 - map_tolerance) and, when given,
    candidate Recall@k >= baseline x (1 - recall_tolerance). Missing MAP -> accept (trainer behavior)."""
    if candidate_map is None or baseline_map is None:
        return True
    ok_map=candidate_map>=baseline_map*(1-map_tolerance)
    ok_recall=candidate_recall is None or baseline_recall is None or candidate_recall>=baseline_recall*(1-recall_tolerance)
    return bool(ok_map and ok_recall)


def tolerances(policy):
    policy=policy or {}
    out=(policy.get('training_map_tolerance',0.02),policy.get('training_recall_tolerance',0.10))
    for t in out:
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not 0<=t<1:raise ValueError('Invalid acceptance tolerance')
    return out


def _score(value,name,optional=False):
    if value is None and optional:return None
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:raise ValueError(f'Valid {name} required')
    return value


def _digest(path):
    from hashlib import sha256
    return sha256(Path(path).read_bytes()).hexdigest()


def finish_training(state, version, artifact, candidate_score, baseline_score, evaluation_key,
                    candidate_recall=None, baseline_recall=None, policy=None):
    """Record completed external training; score both models on the same fixed evaluation.

    Acceptance matches the trainer: MAP@12 within training_map_tolerance (default 2%) of the baseline and
    Recall@k within training_recall_tolerance (default 10%) when given.
    A passing candidate is only reviewable; activation is a separate backend action.
    This function does not run the trainer or trust process exit as quality evidence.
    """
    if not isinstance(evaluation_key,str) or not evaluation_key:raise ValueError('Fixed evaluation identity required')
    for score in (candidate_score,baseline_score):_score(score,'MAP@12')
    for score in (candidate_recall,baseline_recall):_score(score,'Recall@k',optional=True)
    map_tol,recall_tol=tolerances(policy)
    artifact=Path(artifact)
    if not artifact.is_file(): raise ValueError('Completed model artifact required')
    digest=_digest(artifact)
    result=json.loads(json.dumps(state)); matches=[j for j in result.get('jobs',[]) if j['version']==version]
    if len(matches)!=1 or matches[0]['status']!='requested':raise ValueError('Unknown or completed job')
    ok=accept(candidate_score,baseline_score,candidate_recall,baseline_recall,map_tol,recall_tol)
    matches[0].update(status='candidate_pending_review' if ok else 'rejected',artifact=str(artifact),
                     sha256=digest,candidate_map12=candidate_score,baseline_map12=baseline_score,evaluation_key=evaluation_key,
                     candidate_recall=candidate_recall,baseline_recall=baseline_recall,map_tolerance=map_tol,
                     recall_tolerance=recall_tol,active_model_unchanged=True)
    return result


def finish_training_report(state, version, report, path_map=None, policy=None):
    """Record a recsys.retrain report (stdout JSON / report.json) for one requested job.

    Per dataset: re-check the trainer's decision with the same rule, hash text/image weights of accepted
    datasets. Job status: candidate_pending_review if the trainer registered a candidate version,
    rejected if it trained but accepted nothing, skipped if it did not train. Never activates.
    path_map maps trainer container paths to local ones, e.g. {"/data/": "data/"}.
    """
    map_tol,recall_tol=tolerances(policy)
    result=json.loads(json.dumps(state)); matches=[j for j in result.get('jobs',[]) if j['version']==version]
    if len(matches)!=1 or matches[0]['status'] not in ('requested','failed'):raise ValueError('Unknown or completed job')
    job=matches[0]
    if report.get('status')!='trained':
        job.update(status='skipped',reason=report.get('reason','trainer did not train'),active_model_unchanged=True)
        return result

    def local(p):
        for prefix,to in (path_map or {}).items():
            if p.startswith(prefix):return to+p[len(prefix):]
        return p
    sources={}
    for name,r in sorted(report.get('sources',{}).items()):
        entry={'status':r.get('status')}
        if r.get('status')=='trained':
            k=next((key[len('base_recall@'):] for key in r if key.startswith('base_recall@')),None)
            vals={'candidate_map12':_score(r.get('new_map@12'),'MAP@12',True),'baseline_map12':_score(r.get('base_map@12'),'MAP@12',True),
                  'candidate_recall':_score(r.get(f'new_recall@{k}'),'Recall@k',True),'baseline_recall':_score(r.get(f'base_recall@{k}'),'Recall@k',True)}
            ok=accept(vals['candidate_map12'],vals['baseline_map12'],vals['candidate_recall'],vals['baseline_recall'],map_tol,recall_tol)
            entry.update(vals,recall_k=k,evaluation_key=r.get('evaluation_key'),eval_users=r.get('eval_users'),
                         accepted=ok,trainer_accepted=r.get('accepted'))
            if r.get('accepted') is not None and bool(r['accepted'])!=ok:entry['rule_mismatch']=True
            if r.get('accepted'):
                files={m:Path(local(r['dir']))/m/'weights.npz' for m in ('text','image')}
                missing=[str(f) for f in files.values() if not f.is_file()]
                if missing:raise ValueError(f'Completed model artifact required: {missing}')
                entry.update(artifact=local(r['dir']),sha256={m:_digest(f) for m,f in files.items()})
        else:
            entry['reason']=r.get('reason')
        sources[name]=entry
    registered=report.get('registered_version_id')
    job.update(status='candidate_pending_review' if registered else 'rejected',registered_version_id=registered,
               trainer_out=local(report.get('out','')),trained_through_event_id=report.get('max_event_id'),
               sources=sources,map_tolerance=map_tol,recall_tolerance=recall_tol,active_model_unchanged=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Closed-window snapshot JSON from metrics producer')
    p.add_argument('--state',type=Path,required=True);p.add_argument('--watch',action='store_true')
    p.add_argument('--policy',type=Path,default=Path(__file__).with_name('policy.json'))
    a=p.parse_args();policy=json.loads(a.policy.read_text(encoding='utf-8'))
    while True:
        state=json.loads(a.state.read_text(encoding='utf-8')) if a.state.exists() else {}
        updated=observe(state,json.loads(a.input.read_text(encoding='utf-8')),policy)
        if updated!=state:
            a.state.parent.mkdir(parents=True,exist_ok=True)
            temporary=a.state.with_suffix('.tmp')
            temporary.write_text(json.dumps(updated,ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(a.state)
            print(json.dumps({'alerts':updated['alerts'],'jobs':updated['jobs']},ensure_ascii=False),flush=True)
        if not a.watch:break
        interval=policy['interval_seconds']
        if type(interval) is not int or interval<1:raise ValueError('Positive polling interval required')
        time.sleep(interval)

if __name__=='__main__':main()
