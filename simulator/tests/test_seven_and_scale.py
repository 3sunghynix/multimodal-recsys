import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from simulator.ab import compare_customers, wilson
from simulator.config import load_config
from simulator.scale import generate

CONFIG = Path(__file__).parents[1] / 'personas.yaml'
NAMES = {'트렌드세터','실용주의자','가성비추구','브랜드충성','충동구매','신중탐색','눈팅형'}


def test_seven_requested_names_and_distinct_behavior():
    cfg=load_config(CONFIG)
    assert {p['name_ko'] for p in cfg['personas']} == NAMES
    assert len(cfg['pilot']['active_personas']) == 7
    assert len({tuple(p['behavior_overrides'].values()) for p in cfg['personas']}) == 7
    assert not {'repeat_purchaser','new_item_seeking','explorer','price_sensitive'} & set(cfg['pilot']['active_personas'])


def test_scale_is_reproducible_and_uses_seven_personas(tmp_path):
    r=generate(tmp_path/'a', products=260, customers=140, min_events=1400)
    again=generate(tmp_path/'b', products=260, customers=140, min_events=1400)
    assert r == again and r['events'] >= 1400
    assert (tmp_path/'a/events.jsonl').read_bytes() == (tmp_path/'b/events.jsonl').read_bytes()
    products=pd.read_parquet(tmp_path/'a/items.parquet')
    assert products[['category_l1','category_l2','category_l3']].notna().all().all()
    people=pd.read_parquet(tmp_path/'a/customers.parquet')
    window=people.loc[people.persona=='window_shopper']
    assert 0 < sum(len(items)==0 for items in window.initial_history_item_ids) < len(window)
    assert all(len(items)<=2 for items in window.initial_history_item_ids)
    events=[json.loads(s) for s in (tmp_path/'a/events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len({e['event_id'] for e in events}) == len(events)
    assert all(not e['server_sent'] for e in events)
    assert any(e['persona']=='window_shopper' for e in events)
    cfg=load_config(CONFIG)
    shopper=next(p for p in cfg['personas'] if p['id']=='window_shopper')
    assert shopper['behavior_overrides']=={'click_base_probability':.02,'cart_probability':.05,'purchase_probability':.2}
    assert {'search','view','click','cart','purchase'} <= {e['type'] for e in events}
    assert all(g['views']>=0 for g in r['groups'])
    with pytest.raises(FileExistsError):generate(tmp_path/'a')


def test_config_ratios_control_customer_counts(tmp_path):
    cfg=load_config(CONFIG)
    for p in cfg['personas']:p['sampling_weight']=0
    cfg['personas'][0]['sampling_weight']=3
    cfg['personas'][1]['sampling_weight']=1
    path=tmp_path/'personas.yaml';path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    generate(tmp_path/'out',path,products=260,customers=140,min_events=1400)
    counts=pd.read_parquet(tmp_path/'out/customers.parquet').persona.value_counts().to_dict()
    assert counts=={'trendsetter':105,'practical':35}


def outcomes(a,b,n=1000):
    return [{'customer_id':f'{arm}-{i}', 'variant':arm, 'clicked_first_session':i<s} for arm,s in [('A',a),('B',b)] for i in range(n)]


def test_z_test_known_difference_and_zero_case():
    r=compare_customers(outcomes(100,150))
    assert r['metric']=='first_session_customer_click_rate'
    assert r['difference_B_minus_A']==pytest.approx(.05)
    assert r['p_value']==pytest.approx(.0007232327,rel=1e-5)
    assert 0 < r['ci95'][0] < r['ci95'][1]
    zero=compare_customers(outcomes(0,0))
    assert zero['p_value'] is None and zero['status'].startswith('insufficient')
    assert wilson(0,100)[1] > 0


def test_ab_rejects_repeated_or_paired_customer():
    rows=outcomes(10,20,n=100)
    rows[-1]['customer_id']=rows[0]['customer_id']
    with pytest.raises(ValueError):compare_customers(rows)


def test_scale_ab_uses_first_session_and_reports_all_session_ctr(tmp_path):
    report=generate(tmp_path/'out',products=260,customers=140,min_events=2800)
    assert report['sessions_per_customer']==2
    events=[json.loads(line) for line in (tmp_path/'out/events.jsonl').read_text(encoding='utf-8').splitlines()]
    for persona,result in report['ab_by_persona'].items():
        for arm in ('A','B'):
            first_clickers={e['customer_id'] for e in events if e['persona']==persona and
                            e['variant']==arm and e['session']==0 and e['type']=='click'}
            assert result[arm]['clicked_customers']==len(first_clickers)
            group=next(g for g in report['groups'] if g['persona']==persona and g['variant']==arm)
            assert group['ctr']==pytest.approx(group['clicks']/group['views'])
