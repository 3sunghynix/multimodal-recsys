"""Poll closed metric windows from JSON; persist alerts and local training requests.

Counters must exclude is_setup events. State is for one stream/model/data cohort.
No DB writes, automatic model activation, or hidden training commands.
"""
import argparse
import json
import math
from pathlib import Path
import time


def integer(value,name):
    if type(value) is not int or value<0: raise ValueError(f'{name}: nonnegative integer required')
    return value


def observe(state, window, policy):
    """One closed, non-overlapping window per unique monotonic window_end."""
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
    if logs-watermark>=policy['training_log_threshold']:
        jobs.append({'version':f"candidate-{len(jobs)+1:06d}",'status':'requested',
                     'logs_through':logs,'window_id':window['window_id'],'active_model_unchanged':True,
                     'reason':'new_nonsetup_log_threshold'})
        state['training_watermark']=logs
    state.update(last_window_id=window['window_id'],last_window_end=window['window_end'],last_cumulative_logs=logs,
                 latest={'values':values,'eligible':eligible,'views':views,'evaluated_users':users})
    return state


def finish_training(state, version, artifact, candidate_score, baseline_score, evaluation_key):
    """Record completed external training; score both models on the same fixed evaluation.

    A passing candidate is only reviewable; activation is a separate backend action.
    This function does not run the trainer or trust process exit as quality evidence.
    """
    if not isinstance(evaluation_key,str) or not evaluation_key:raise ValueError('Fixed evaluation identity required')
    for score in (candidate_score,baseline_score):
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Valid MAP@12 required')
    artifact=Path(artifact)
    if not artifact.is_file(): raise ValueError('Completed model artifact required')
    from hashlib import sha256
    digest=sha256(artifact.read_bytes()).hexdigest()
    result=json.loads(json.dumps(state)); matches=[j for j in result.get('jobs',[]) if j['version']==version]
    if len(matches)!=1 or matches[0]['status']!='requested':raise ValueError('Unknown or completed job')
    matches[0].update(status='candidate_pending_review' if candidate_score>=baseline_score else 'rejected',artifact=str(artifact),
                     sha256=digest,candidate_map12=candidate_score,baseline_map12=baseline_score,evaluation_key=evaluation_key,
                     active_model_unchanged=True)
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
