"""베이스라인과 학습 없는 방식. 결과 형식은 Two-Tower와 같다 (candidates.npy, users.npy, run_info.json).

B1: 직전 1주 인기 상품 / B2: 재구매(최근 순) + 인기 상품
K0: 텍스트 임베딩 이력 평균과 가까운 상품 / K2: 텍스트 + Marqo 이미지 (이어 붙임)
"""
import time

import numpy as np
import pandas as pd

from .data_prep import load_split
from .two_tower import item_features, user_history
from .utils import environment_info, local_tmp_dir, now, publish_dir, read_json, write_json

KNN_SPECS = {"K0": {"text": True, "image": None, "shuffle": None},
             "K2": {"text": True, "image": "marqo", "shuffle": None}}


def popular_items(history, start, n_items):
    """직전 1주 판매량 순. 모자라면 전체 기간 판매량 순, 그래도 모자라면 나머지 상품으로 채운 전체 순위."""
    last_week = history[(history["t_dat"] >= start - pd.Timedelta(weeks=1)) & (history["t_dat"] < start)]
    order = list(last_week["item_idx"].value_counts().index)
    seen = set(order)
    order += [x for x in history["item_idx"].value_counts().index if x not in seen]
    seen = set(order)
    order += [x for x in range(n_items) if x not in seen]
    return np.array(order, dtype=np.int32)


def repurchase_popular(history, users, popular, k):
    r = history[history["user"].isin(users)].sort_values(["user", "t_dat"], ascending=[True, False])
    r = r.drop_duplicates(["user", "item_idx"])
    past = r.groupby("user")["item_idx"].apply(list)
    top = popular[:2 * k]
    out = np.empty((len(users), k), dtype=np.int32)
    for i, u in enumerate(users):
        mine = past.get(u, [])[:k]
        mine_set = set(mine)
        out[i] = mine + [x for x in top if x not in mine_set][:k - len(mine)]
    return out


def knn_mean(X, history, users, max_hist, k, device, chunk=4096):
    import torch
    import torch.nn.functional as F
    from .two_tower import mean_features
    flat, offsets, _ = user_history(history, users, max_hist)
    Xt = F.normalize(torch.from_numpy(X).to(device), dim=1)
    U = F.normalize(mean_features(Xt, flat, offsets, device), dim=1)
    out = [(U[s:s + chunk] @ Xt.T).topk(k, dim=1).indices.int().cpu().numpy() for s in range(0, len(U), chunk)]
    return np.concatenate(out)


def run(cfg, paths, name, split, group="C", device=None, log=print):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = paths.run_dir(group, split, f"{name}_seed0")
    if (out_dir / "run_info.json").exists():
        log(f"[skip] {out_dir} 이미 완료")
        return read_json(out_dir / "run_info.json")
    k, max_hist = cfg["eval"]["k"], cfg["train"]["default"]["max_hist"]
    stats = read_json(paths.split_dir(split) / "stats.json")
    history, _, targets = load_split(paths, split)
    start = pd.Timestamp(stats["test_start"])
    history = history[history["t_dat"] < start]
    users = np.sort(targets["user"].unique())
    t0 = time.time()
    n_items = len(pd.read_parquet(paths.processed / "items.parquet", columns=["idx"]))
    popular = popular_items(history, start, n_items)
    if name == "B1":
        cands = np.tile(popular[:k], (len(users), 1))
    elif name == "B2":
        cands = repurchase_popular(history, users, popular, k)
    else:
        cands = knn_mean(item_features(paths, KNN_SPECS[name], seed=0), history, users, max_hist, k, device)
    info = {"name": f"{name}_seed0", "cond": name, "seed": 0, "split": split, "group": group, "mode": "final",
            "n_users": int(len(users)), "train_seconds": round(time.time() - t0, 1), "created_at": now(),
            **environment_info()}
    tmp = local_tmp_dir()
    np.save(tmp / "candidates.npy", cands.astype(np.int32))
    np.save(tmp / "users.npy", users.astype(np.int64))
    write_json(tmp / "run_info.json", info)
    publish_dir(tmp, out_dir)
    log(f"[{name}] 저장 → {out_dir}")
    return info
