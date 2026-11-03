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
