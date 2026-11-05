"""Streaming synthetic scale fixture, separate from real Amazon/H&M training data.

python -m simulator.scale --out simulator/runs/scale_001
Produces >=50k products, >=10k customers, >=1m local events by default.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .config import load_config
from .data import derived_seed, file_hash
from .engine import click_probability
from .ab import compare_customers


def generate(out, config_path=None, products=50000, customers=10000, min_events=1000000, seed=42):
    for name,value in [('products',products),('customers',customers),('min_events',min_events),('seed',seed)]:
        if type(value) is not int or value < (0 if name=='seed' else 1):
            raise ValueError(f'{name}: valid integer required')
    if products < 260 or customers < 14:
        raise ValueError('At least 260 products (10 per leaf category) and 14 customers required')
    cfg=load_config(config_path or Path(__file__).with_name('personas.yaml'))
    out=Path(out)
    if out.exists(): raise FileExistsError(f'Choose new output directory: {out}')
    specs=[p for p in cfg['personas'] if p['id'] in cfg['pilot']['active_personas']]
    weights=np.array([p['sampling_weight'] for p in specs],float)
    if weights.sum() <= 0: raise ValueError('At least one positive persona weight required')
    quotas=np.floor(customers*weights/weights.sum()).astype(int)
    for j in np.argsort(-(customers*weights/weights.sum()-quotas))[:customers-int(quotas.sum())]: quotas[j]+=1
    # Fixed observation length for every customer, determined before seeing outcomes.
    k=10
    sessions=math.ceil(min_events/(customers*k))
    rng=np.random.default_rng(seed)
    # Representative leaves from the agreed three-level taxonomy (PR #36).
    leaves={'상의':['티셔츠','셔츠·블라우스'], '하의':['청바지','바지'],
            '이너웨어':['속옷','양말·스타킹'], '원피스·세트':['원피스'],
            '아우터':['재킷'], '신발':['운동화','부츠'],
            '가방·지갑':['가방'], '액세서리':['귀걸이','목걸이']}
    categories=[(target,kind,leaf) for target in ['여성','남성'] for kind,names in leaves.items() for leaf in names]
    rows=[]
    for i in range(products):
        c=categories[i%len(categories)]
        rows.append({'item_id':f'synthetic:{i}','source':'synthetic','brand':f'brand_{i%100}',
                     'category_l1':c[0],'category_l2':c[1],'category_l3':c[2],
                     'price':round(float(rng.uniform(10,200)),2),'price_source':'synthetic',
                     'popularity':float(rng.random())})
    catalog=pd.DataFrame(rows)
    top=catalog.sort_values(['popularity','item_id'],ascending=[False,True]).head(k).index.tolist()
    by_category={j:catalog.loc[(catalog.category_l1==c[0])&(catalog.category_l2==c[1])&(catalog.category_l3==c[2])].sort_values('popularity',ascending=False).head(k).index.tolist() for j,c in enumerate(categories)}
    # Deterministic assignment independent of outcomes and persona preference draws.
    people=[]
    for spec,n in zip(specs,quotas):
        indices=sorted(range(int(n)),key=lambda i:derived_seed(seed,'assignment',spec['id'],i))
        arm_a=set(indices[:int(n)//2])
        rule=spec.get('category_sampling',{})
        cat_weights=np.array([rule.get('matching_multiplier',1) if any(t.lower() in ' > '.join(c).lower() for t in rule.get('matching_terms',[])) else 1 for c in categories],float)
        cat_weights/=cat_weights.sum()
        budget_by_category={j:float(catalog.loc[(catalog.category_l1==c[0]) &
                                                 (catalog.category_l2==c[1]) &
                                                 (catalog.category_l3==c[2]),'price'].quantile(spec.get('budget_quantile',.5)))
                            for j,c in enumerate(categories)}
        for i in range(int(n)):
            cr=np.random.default_rng(derived_seed(seed,'customer',spec['id'],i))
            category_index=int(cr.choice(len(categories),p=cat_weights))
            history_rule=spec.get('initial_history',{})
            history_size=int(cr.choice(history_rule['size_choices'],p=history_rule['weights'])) if history_rule else 10
            history_candidates=by_category[category_index]
            history_indices=cr.choice(history_candidates,size=min(history_size,len(history_candidates)),replace=False).tolist()
            people.append({'customer_id':f"scale-{spec['id']}-{i:05d}",'persona':spec['id'],'variant':'A' if i in arm_a else 'B',
                           'category_index':category_index,
                           'budget':budget_by_category[category_index],
                           'brand':f'brand_{int(cr.integers(100))}',
                           'initial_history_item_ids':[rows[j]['item_id'] for j in history_indices]})
    out.mkdir(parents=True,exist_ok=False)
    catalog.to_parquet(out/'items.parquet',index=False)
    pd.DataFrame(people).to_parquet(out/'customers.parquet',index=False)
    totals=Counter(); group=Counter(); outcomes=[]
    with (out/'events.jsonl').open('w',encoding='utf-8') as f:
        def emit(person,session,action,item=None,rank=None):
            record={'event_id':str(uuid.uuid5(uuid.NAMESPACE_URL,f"{seed}/{person['customer_id']}/{session}/{rank}/{action}")),
                    'customer_id':person['customer_id'],'persona':person['persona'],'variant':person['variant'],
                    'strategy':'popularity' if person['variant']=='A' else 'category_popularity',
                    'session':session,'type':action,'item_id':item,'rank':rank,'source':'synthetic',
                    'query':' > '.join(categories[person['category_index']]) if action=='search' else None,
                    'server_sent':False}
            f.write(json.dumps(record,ensure_ascii=False)+'\n'); totals[action]+=1;group[(person['persona'],person['variant'],action)]+=1
        by_id={s['id']:s for s in specs}
        for person in people:
            spec=by_id[person['persona']]; override=spec['behavior_overrides']; clicked_first_session=False
            selection=top if person['variant']=='A' else by_category[person['category_index']]
            history_items=set(person['initial_history_item_ids'])
            br=np.random.default_rng(derived_seed(seed,'behavior',person['customer_id']))
            for session in range(sessions):
                emit(person,session,'search')
                for rank,index in enumerate(selection,1):
                    item=rows[index]; category=tuple(item[f'category_l{j}'] for j in (1,2,3))
                    affinity=float(category==categories[person['category_index']])
                    feats={'category_affinity':affinity,'familiarity_affinity':affinity,'price_affinity':max(0,1-item['price']/person['budget']),
                           'brand_affinity':float(item['brand']==person['brand']),
                           'novelty_affinity':float(item['item_id'] not in history_items and item['brand']!=person['brand']),
                           'repurchase_affinity':float(item['item_id'] in history_items), 'popularity_affinity':item['popularity']}
                    score=sum(v*feats[key] for key,v in spec['preference_weights'].items())
                    draws=br.random(3)
                    emit(person,session,'view',item['item_id'],rank)
                    if draws[0]<click_probability(score,cfg,spec):
                        if session==0: clicked_first_session=True
                        emit(person,session,'click',item['item_id'],rank)
                        if draws[1]<override['cart_probability']:
                            emit(person,session,'cart',item['item_id'],rank)
                            if draws[2]<override['purchase_probability']:emit(person,session,'purchase',item['item_id'],rank)
            outcomes.append({'customer_id':person['customer_id'],'persona':person['persona'],'variant':person['variant'],
                             'clicked_first_session':clicked_first_session})
    ab={}
    for s in specs:
        cohort=[r for r in outcomes if r['persona']==s['id']]
        if {r['variant'] for r in cohort}=={'A','B'}: ab[s['id']]=compare_customers(cohort)
    report={'kind':'synthetic_scale_fixture','calibrated':False,'products':products,'customers':customers,
            'category_levels':3,'sessions_per_customer':sessions,'events':sum(totals.values()),'event_counts':dict(totals),
            'seed':seed,'config':cfg,'strategies':{'A':'popularity','B':'category_popularity'},'ab_by_persona':ab,
            'groups':[{'persona':s['id'],'variant':v,'views':group[s['id'],v,'view'],'clicks':group[s['id'],v,'click'],
                       'ctr':group[s['id'],v,'click']/group[s['id'],v,'view'] if group[s['id'],v,'view'] else None,
                       'purchases':group[s['id'],v,'purchase']} for s in specs for v in ('A','B')],
            'requirements_met':{'products_50000':products>=50000,'customers_10000':customers>=10000,'events_1000000':sum(totals.values())>=1000000},
            'limitations':['Synthetic products are not the real service catalog and have no image embeddings',
                            'Category-popularity comparator is not UME', 'First-session customer click test; CTR covers all views and clicks separately',
                            'Unadjusted per-persona tests; no real-user claim or automatic winner']}
    (out/'scale_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'checksums.json').write_text(json.dumps({p.name:file_hash(p) for p in out.iterdir() if p.is_file()},indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--products',type=int,default=50000);p.add_argument('--customers',type=int,default=10000)
    p.add_argument('--min-events',type=int,default=1000000);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();r=generate(a.out,a.config,a.products,a.customers,a.min_events,a.seed)
    print(json.dumps({k:r[k] for k in ['products','customers','events','requirements_met']},indent=2))

if __name__=='__main__':main()
