"""monitoring.ct — 서버 · trainer 없이 가짜 API · 가짜 trainer로 한 바퀴를 확인한다."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from monitoring.ct import closed_windows, parse_report, tick

POLICY=json.loads((Path(__file__).parents[1]/'policy.json').read_text(encoding='utf-8'))
NOW=datetime(2026,11,23,10,30,tzinfo=timezone.utc)


class FakeAPI:
    def __init__(self,max_event_id=20000,delta=12000,model=None,views=2000,clicks=10):
        self.max_event_id,self.delta,self.views,self.clicks=max_event_id,delta,views,clicks
        self.model=model or {'id':3,'name':'ume','version':'v1','status':'active','params':{'trained_until_event_id':8000}}
        self.calls=[]

    def __call__(self,path,params=None):
        self.calls.append((path,params))
        if path=='/models':return [{'id':1,'name':'popularity','version':'v1','status':'candidate','params':{}},self.model]
        if path=='/metrics/window':
            return dict(window_id=f"{params['start']}__{params['end']}",window_start=params['start'],window_end=params['end'],
                        model_version=params['model_version'],source='all',cohort='all',metric_definition_version='user_click_through_v1',
                        setup_excluded=True,views=self.views,clicks=self.clicks,evaluated_users=200,hit_users=5,cumulative_new_logs=self.max_event_id)
        if path=='/metrics/log-delta':
            s=params['since_event_id'];self.types=params.get('types')
            return dict(since_event_id=s,max_event_id=self.max_event_id,delta=self.delta if s<self.max_event_id else 0,
                        reset_detected=self.max_event_id<s)
        raise AssertionError(path)


def report(tmp_path,accepted=True,registered=7):
    for src in ('hm','amazon'):
        for m in ('text','image'):
            f=tmp_path/'data/models/ume/retrain_x'/src/m/'weights.npz';f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(src.encode()+m.encode())
    src=lambda new_map:{'status':'trained','dir':'/data/models/ume/retrain_x/hm','eval_users':5000,'base_recall@20':0.153,'new_recall@20':0.1516,
                        'base_map@12':0.0425,'new_map@12':new_map,'evaluation_key':'leave-last-out:holdout=0.1:upto=20000:users=5000','accepted':accepted}
    r={'base_version_id':3,'since_event_id':8000,'new_logs':12000,'max_event_id':20000,'status':'trained',
       'out':'/data/models/ume/retrain_x','sources':{'hm':src(0.0428 if accepted else 0.03),'amazon':{'status':'skipped','reason':'학습 표본 1 < 100'}}}
    if registered:r['registered_version_id']=registered
    return r


def trainer(stdout,code=0,calls=None):
    def run(cmd):
        if calls is not None:calls.append(cmd)
        return code,stdout,'boom' if code else ''
    return run


@pytest.fixture
def policy(tmp_path):
    return {**POLICY,'trainer_path_map':{'/data/':str(tmp_path/'data')+'/'}}


def test_closed_windows_first_run_and_catch_up():
    assert closed_windows(NOW,60,None,24)==[(datetime(2026,11,23,9,tzinfo=timezone.utc),datetime(2026,11,23,10,tzinfo=timezone.utc))]
    assert len(closed_windows(NOW,60,'2026-11-23T05:00:00+00:00',24))==5
    assert len(closed_windows(NOW,60,'2026-11-23T05:00:00+00:00',2))==2
    assert closed_windows(NOW,60,'2026-11-23T10:00:00+00:00',24)==[]


def test_parse_report_after_training_logs():
    out='[hm] 상품 19,998 · 이벤트 1\nepoch 1 loss {0.3}\n'+json.dumps({'status':'skipped','reason':'x'},indent=2)+'\n'
    assert parse_report(out)['status']=='skipped'
    with pytest.raises(ValueError):parse_report('no json here')


def test_full_loop_requests_trains_and_records_candidate(tmp_path,policy):
    calls=[];api=FakeAPI()
    out='training...\n'+json.dumps(report(tmp_path),indent=2)
    s=tick({},api,policy,NOW,run=trainer(out,calls=calls))
    job=s['jobs'][0]
    assert job['status']=='candidate_pending_review' and job['registered_version_id']==7 and job['active_model_unchanged']
    assert job['sources']['hm']['accepted'] and len(job['sources']['hm']['sha256']['text'])==64
    assert 'rule_mismatch' not in job['sources']['hm'] and job['sources']['amazon']['status']=='skipped'
    assert ('/metrics/log-delta',{'since_event_id':8000,'types':'click,add_to_cart,purchase'}) in api.calls          # 운영 모델의 trained_until_event_id부터
    cmd=calls[0];assert cmd[:4]==['docker','compose','--profile','trainer'] and cmd[cmd.index('--base')+1]=='3'
    assert cmd[cmd.index('--tolerance')+1]=='0.02'
    assert s['since_event_id']==20000
    # 다음 tick: 새 로그 없음 → 요청 · 실행 없음, 같은 구간 다시 안 읽음
    s2=tick(s,api,policy,NOW,run=trainer('',calls=calls))
    assert len(s2['jobs'])==1 and len(calls)==1
    assert [c for c in api.calls if c[0]=='/metrics/window'].__len__()==1


def test_below_threshold_does_not_request(policy):
    s=tick({},FakeAPI(delta=9999),policy,NOW,run=trainer('should not run',code=1))
    assert s['jobs']==[] and s['latest_log_delta']['delta']==9999


def test_trained_but_rejected(tmp_path,policy):
    out=json.dumps(report(tmp_path,accepted=False,registered=None),indent=2)
    s=tick({},FakeAPI(),policy,NOW,run=trainer(out))
    assert s['jobs'][0]['status']=='rejected' and not s['jobs'][0]['sources']['hm']['accepted']
    assert s['since_event_id']==20000                                           # 같은 로그로 다시 학습하지 않음


def test_trainer_failure_retries_then_cools_down_and_retries(tmp_path,policy):
    from datetime import timedelta
    api=FakeAPI();calls=[]
    s=tick({},api,policy,NOW,run=trainer('',code=1,calls=calls))
    assert s['jobs'][0]['status']=='failed' and s['jobs'][0]['attempts']==1 and not any(a.get('kind')=='trainer_failed' for a in s['alerts'])
    s=tick(s,api,policy,NOW,run=trainer('',code=1,calls=calls))
    assert s['jobs'][0]['attempts']==2 and any(a.get('kind')=='trainer_failed' for a in s['alerts'])
    later=NOW+timedelta(minutes=30)                                            # 대기 시간(60분) 안: 새 요청 · 실행 없음
    s=tick(s,api,policy,later,run=trainer('',code=1,calls=calls))
    assert len(calls)==2 and len(s['jobs'])==1
    out=json.dumps(report(tmp_path),indent=2)                                  # 대기 시간이 지나면 손대지 않아도 다시 요청 · 실행
    s=tick(s,api,policy,NOW+timedelta(minutes=61),run=trainer(out,calls=calls))
    assert len(calls)==3 and len(s['jobs'])==2 and s['jobs'][1]['status']=='candidate_pending_review'


def test_log_delta_counts_training_types_only(policy):
    api=FakeAPI(delta=0)
    tick({},api,policy,NOW)
    assert api.types=='click,add_to_cart,purchase'


def test_skipped_report_and_non_ume_and_reset(tmp_path,policy):
    out=json.dumps({'base_version_id':3,'since_event_id':8000,'new_logs':5,'max_event_id':20000,'status':'skipped','reason':'새 로그 5 < 10000'},indent=2)
    assert tick({},FakeAPI(),policy,NOW,run=trainer(out))['jobs'][0]['status']=='skipped'
    pop=FakeAPI(model={'id':1,'name':'popularity','version':'v1','status':'active','params':{}})
    s=tick({},pop,policy,NOW,run=trainer('x',code=1))
    assert s['jobs']==[] and 'not ume' in s['last_note']
    s=tick({'since_event_id':50000},FakeAPI(max_event_id=100),policy,NOW,run=trainer('x',code=1))
    assert s['alerts'][-1]['kind']=='log_counter_reset' and s['since_event_id']==100 and s['jobs']==[]


def test_low_ctr_windows_alert_without_job(policy):
    api=FakeAPI(delta=0,views=2000,clicks=10)                                   # CTR 0.5% < 2%
    s=tick({'last_window_end':'2026-11-23T07:00:00+00:00'},api,policy,NOW)
    assert any(a['metric']=='ctr' for a in s['alerts'] if 'metric' in a) and s['jobs']==[]


def test_dry_run_records_command_only(policy):
    s=tick({},FakeAPI(),policy,NOW,run=trainer('x',code=1),dry_run=True)
    assert s['jobs'][0]['status']=='requested' and s['jobs'][0]['dry_run'] and s['jobs'][0]['attempts']==0


def test_state_has_dashboard_keys(tmp_path,policy):
    pd=pytest.importorskip("pandas")
    s=tick({},FakeAPI(),policy,NOW,run=trainer(json.dumps(report(tmp_path),indent=2)))
    assert s['identity'][0]=='ume-v1' and 'values' in s['latest'] and s['last_window_end']=='2026-11-23T10:00:00+00:00'
    flat=pd.json_normalize(s['jobs'])
    assert 'sources.hm.accepted' in flat.columns


def test_real_run_after_dry_run_clears_flag(tmp_path,policy):
    s=tick({},FakeAPI(),policy,NOW,run=trainer('x',code=1),dry_run=True)
    s=tick(s,FakeAPI(),policy,NOW,run=trainer(json.dumps(report(tmp_path),indent=2)))
    assert s['jobs'][0]['status']=='candidate_pending_review' and 'dry_run' not in s['jobs'][0]


def ranker_report(tmp_path,accepted=True,new_map=0.0495,ref='base_refit'):
    r=report(tmp_path)
    d=tmp_path/'data/models/ume/retrain_x/ranker/hm'
    if accepted:
        d.mkdir(parents=True,exist_ok=True);(d/'model.joblib').write_bytes(b'm');(d/'meta.json').write_text('{}')
    rk={'status':'trained','accepted':accepted,'compared_with':ref,'eval_users':5000,'rule_map@12':0.0430,'new_map@12':new_map,
        'base_map@12':0.0560,'base_refit_map@12':0.0480,'evaluation_key':'ranker:time_week:until=2020-09-08:users=5000'}
    if ref=='base_refit':rk['base_seen_eval_window']='2020-09-08 00:00:00'
    if accepted:rk['path']='/data/models/ume/retrain_x/ranker/hm'
    r['rankers']={'hm':rk,'amazon':{'status':'failed','error':'RuntimeError: x'}}
    return r


def test_ranker_entries_rechecked_and_hashed(tmp_path,policy):
    """B1 랭커 재학습: runner도 같은 MAP@12 규칙으로 다시 판정 (trainer의 비교 대상 기준), 채택된 랭커 파일 sha256."""
    s=tick({},FakeAPI(),policy,NOW,run=trainer(json.dumps(ranker_report(tmp_path),indent=2)))
    rk=s['jobs'][0]['rankers']
    assert rk['hm']['accepted'] and rk['hm']['compared_with']=='base_refit' and rk['hm']['baseline_map12']==0.0480
    assert len(rk['hm']['sha256']['model.joblib'])==64 and 'rule_mismatch' not in rk['hm'] and rk['hm']['base_seen_eval_window']
    assert rk['amazon']=={'status':'failed','reason':'RuntimeError: x'}


def test_ranker_rule_mismatch_flagged(tmp_path,policy):
    """trainer는 채택했는데 같은 규칙으로는 거절(기준 랭커 0.0560 대비 0.0495)이면 rule_mismatch."""
    s=tick({},FakeAPI(),policy,NOW,run=trainer(json.dumps(ranker_report(tmp_path,ref='base'),indent=2)))
    rk=s['jobs'][0]['rankers']['hm']
    assert rk['accepted'] is False and rk['trainer_accepted'] and rk['rule_mismatch']


def test_report_without_rankers_has_no_key(tmp_path,policy):
    s=tick({},FakeAPI(),policy,NOW,run=trainer(json.dumps(report(tmp_path),indent=2)))
    assert 'rankers' not in s['jobs'][0]
