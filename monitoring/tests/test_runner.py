import json
from pathlib import Path

import pytest
from monitoring.runner import observe, finish_training

POLICY=json.loads((Path(__file__).parents[1]/'policy.json').read_text(encoding='utf-8'))


def window(i, **changes):
    w=dict(window_id=f'w{i}',window_start=f'2026-10-31T0{i}:00:00+00:00',window_end=f'2026-10-31T0{i+1}:00:00+00:00',
           model_version='ume-v1',source='amazon',cohort='fixed-users-v1',metric_definition_version='v1',
           setup_excluded=True,views=1000,clicks=10,evaluated_users=100,hit_users=1,cumulative_new_logs=i*5000)
    w.update(changes);return w


def test_three_windows_alert_and_request_only_once():
    state={}
    for i in range(3):state=observe(state,window(i),POLICY)
    assert len(state['alerts'])==2
    assert len(state['jobs'])==1 and state['jobs'][0]['status']=='requested'
    assert state['jobs'][0]['active_model_unchanged']
    assert observe(state,window(2),POLICY)==state


def test_denominators_and_identity_are_guarded():
    state=observe({},window(0),POLICY)
    state=observe(state,window(1,views=1,clicks=0,evaluated_users=1,hit_users=0),POLICY)
    assert state['streak']=={'ctr':0,'hitrate':0}
    for change in [dict(source='hm'),dict(setup_excluded=False),dict(clicks=1001),dict(cumulative_new_logs=0),dict(window_start='2026-10-31T00:00:00+00:00')]:
        with pytest.raises(ValueError):observe(state,window(2,**change),POLICY)


def test_rejected_and_reviewable_versions_do_not_activate(tmp_path):
    state=observe({},window(0,cumulative_new_logs=10000),POLICY)
    artifact=tmp_path/'model.npz';artifact.write_bytes(b'test-artifact')
    bad=finish_training(state,'candidate-000001',artifact,.1,.2,'amazon:test:users-v1')
    assert bad['jobs'][0]['status']=='rejected'
    good=finish_training(state,'candidate-000001',artifact,.3,.2,'amazon:test:users-v1')
    assert good['jobs'][0]['status']=='candidate_pending_review'
    assert good['jobs'][0]['active_model_unchanged'] and len(good['jobs'][0]['sha256'])==64
    with pytest.raises(ValueError):finish_training(good,'candidate-000001',artifact,.3,.2,'fixed')


def test_acceptance_matches_trainer_tolerance(tmp_path):
    state=observe({},window(0,cumulative_new_logs=10000),POLICY)
    artifact=tmp_path/'model.npz';artifact.write_bytes(b'x')
    # 2% 이내 하락은 통과 (예전 runner 규칙 candidate>=baseline 에서는 rejected)
    assert finish_training(state,'candidate-000001',artifact,.197,.2,'k',policy=POLICY)['jobs'][0]['status']=='candidate_pending_review'
    assert finish_training(state,'candidate-000001',artifact,.195,.2,'k',policy=POLICY)['jobs'][0]['status']=='rejected'
    # Recall 보조 안전장치: 10% 넘게 하락하면 MAP@12가 좋아도 거절
    assert finish_training(state,'candidate-000001',artifact,.3,.2,'k',.089,.1,POLICY)['jobs'][0]['status']=='rejected'
    assert finish_training(state,'candidate-000001',artifact,.3,.2,'k',.091,.1,POLICY)['jobs'][0]['status']=='candidate_pending_review'


def test_accept_rule_equals_trainer():
    retrain=pytest.importorskip('recsys.retrain')
    from monitoring.runner import accept
    grid=[None,0.0,0.0039,0.0041,0.0425,0.0428,0.1,0.2]
    for b in grid:
        for n in grid:
            for br,nr in [(None,None),(0.0198,0.0192),(0.153,0.12),(0.1,0.091)]:
                assert accept(n,b,nr,br)==retrain.accept(b,n,br,nr)


def test_window_jobs_can_be_disabled():
    state={}
    for i in range(3):state=observe(state,window(i),POLICY,request_jobs=False)
    assert state.get('jobs')==[] and len(state['alerts'])==2
