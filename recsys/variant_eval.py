"""C1 색상 변형 묶기 영향: H&M 서비스 로그 leave-last-out (trainer와 같은 분할), 운영 가중치(hm_test_pca).

실행: docker compose --profile trainer run --rm trainer python -m recsys.variant_eval   (결과: docs/project/추천모델설계.md "색상 변형 묶기")

순위 = UME z(text)+z(image). 묶기 = 같은 product_code(article_id 앞 7자리)는 위에서 cap개까지 (cap 1 · 2 · 3).
지표: 정확 일치 MAP@12 · Recall@12, 같은 상품 코드면 맞음으로 보는 코드 단위 Recall@12, 목록 내 중복 코드 수.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/app")
from app.db import SessionLocal
from app.models import Item
from app.services.catalog_index import get_index
from recsys.eval.metrics import accuracy_at_k
from recsys.retrain import _tower_from_npz, load_events, split_holdout
from recsys.ume import DEFAULT, build_sequences, tower_vectors

K, POOL, HOLD, MAXU = 12, 300, 0.1, 5000
db = SessionLocal()
ix = get_index(db)
rows = np.flatnonzero(ix.sources == "hm")
ids = ix.ids[rows]
sid = dict(db.query(Item.id, Item.source_item_id).filter(Item.source == "hm").all())
code = np.array([sid[int(i)][:7] for i in ids], object)
ev = load_events(db, 2**31 - 1)
ev = ev[ev.item_id.isin(ids)].copy()
ev["item_id"] = np.searchsorted(ids, ev.item_id.values)
train, test = split_holdout(ev, HOLD)
targets = test.groupby("user_id").item_id.apply(set)
seen = set(train.user_id)
users = np.array([u for u in targets.index if u in seen], np.int64)[:MAXU]
truth = pd.DataFrame([(u, i) for u in users for i in targets[u]], columns=["user_id", "item_id"])
base = Path("/data/models/ume/hm_test_pca")
p = {**DEFAULT, **{k: v for k, v in json.loads((base / "text/config.json").read_text()).items() if k in DEFAULT}}
seq = build_sequences(train, p["max_hist"])
catalog = np.arange(len(rows))
X = {"text": ix.text[rows], "image": ix.image[rows]}
z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)  # noqa: E731
S = 0
for m in ("text", "image"):
    w = dict(np.load(base / m / "weights.npz"))
    U, V = tower_vectors(_tower_from_npz(w, p, "cpu"), X[m], seq, users, catalog, p["max_hist"], "cpu")
    S = S + z(U @ V.T)


def collapse(order, cap=1):
    out, used = [], {}
    for j in order:
        if used.get(code[j], 0) >= cap:
            continue
        used[code[j]] = used.get(code[j], 0) + 1; out.append(j)
        if len(out) == K:
            return out
    rest = [j for j in order if j not in set(out)]
    return out + rest[:K - len(out)]


res = {}
for name, s in (("ume", S),):
    order = np.argsort(-s, axis=1)[:, :POOL]
    for mode in ("as_is", "cap1", "cap2", "cap3"):
        rec = order[:, :K] if mode == "as_is" else np.array([collapse(o, int(mode[-1])) for o in order])
        acc = accuracy_at_k(rec, users, truth, ks=(K,))[0]
        code_hit = np.mean([len({code[j] for j in r} & {code[t] for t in targets[u]}) > 0 for r, u in zip(rec, users)])
        dup = np.mean([K - len({code[j] for j in r}) for r in rec])
        res[f"{name}/{mode}"] = {"map@12": round(acc["map@12"], 5), "recall@12": round(acc["recall@12"], 5),
                                 "code_recall@12": round(float(code_hit), 5), "dup_codes_per_list": round(float(dup), 3)}
res["eval_users"] = int(len(users))
res["short_lists"] = int(sum(len({code[j] for j in o}) < K for o in np.argsort(-S, axis=1)[:, :POOL]))
same = np.mean([any(code[t] in set(code[train[train.user_id == u].item_id.values]) for t in targets[u]) for u in users[:2000]])
res["truth_code_in_history_share"] = round(float(same), 3)
print(json.dumps(res, indent=1, ensure_ascii=False))
