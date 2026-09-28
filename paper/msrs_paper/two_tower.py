"""ID 없는 Two-Tower.

상품 타워: 상품의 임베딩 입력(텍스트 / 이미지 / 둘을 이어 붙임) → MLP
고객 타워: 고객이 산 상품들의 '상품 타워 입력'을 평균 → MLP
섞기 대응표는 item_features() 한 곳에서만 적용하므로 두 타워에 똑같이 들어간다.

학습 표본: 주 단위. 각 라벨 주의 구매를 정답으로, 그 주 이전 구매를 고객 이력으로 쓴다 (미래 정보 누수 없음).
- tune 모드: 라벨 주가 검증 주 직전에서 끝나고, 검증 주로 채점한다.
- final 모드: 라벨 주에 검증 주까지 포함하고, 테스트 주 후보를 만든다.
"""
import time

import numpy as np
import pandas as pd

from . import metrics
from .config import train_params
from .data_prep import load_split
from .utils import environment_info, local_tmp_dir, now, publish_dir, read_json, set_seed, write_json


def item_features(paths, spec, seed):
    parts = []
    if spec["text"]:
        parts.append(np.load(paths.embeddings / "text_minilm.npy"))
    if spec["image"]:
        img = np.load(paths.embeddings / f"img_{spec['image']}.npy")
        if spec["shuffle"]:
            perm = np.load(paths.embeddings / f"shuffle_{spec['shuffle']}_seed{seed}.npy")
            img = img[perm]
        parts.append(img)
    return np.concatenate(parts, axis=1).astype(np.float32)


def user_history(rows, users, max_hist):
    """users 순서대로, 최근 구매 순 중복 없는 상품 max_hist개 (flat, offsets)."""
    r = rows[rows["user"].isin(users)].sort_values(["user", "t_dat"], ascending=[True, False])
    r = r.drop_duplicates(["user", "item_idx"])
    r = r[r.groupby("user").cumcount() < max_hist]
    pos = pd.Series(np.arange(len(users)), index=users)
    key = pos.loc[r["user"].values].values
    order = np.argsort(key, kind="stable")
    flat = r["item_idx"].values[order].astype(np.int64)
    counts = np.bincount(key, minlength=len(users))
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return flat, offsets, counts


def build_samples(history, period_end, label_weeks, max_hist):
    flats, offs, pair_key, pair_item = [], [], [], []
    n_keys, n_flat = 0, 0
    for k in range(label_weeks):
        wk_end = period_end - pd.Timedelta(weeks=k)
        wk_start = wk_end - pd.Timedelta(weeks=1)
        lab = history[(history["t_dat"] >= wk_start) & (history["t_dat"] < wk_end)][["user", "item_idx"]]
        lab = lab.drop_duplicates()
        prior = history[history["t_dat"] < wk_start]
        users = np.intersect1d(lab["user"].unique(), prior["user"].unique())
        flat, off, _ = user_history(prior, users, max_hist)
        lab = lab[lab["user"].isin(users)]
        pos = pd.Series(np.arange(len(users)), index=users)
        pair_key.append(pos.loc[lab["user"].values].values + n_keys)
        pair_item.append(lab["item_idx"].values)
        flats.append(flat)
        offs.append(off + n_flat)
        n_keys += len(users)
        n_flat += len(flat)
    return {"flat": np.concatenate(flats), "offsets": np.concatenate(offs),
            "pair_key": np.concatenate(pair_key).astype(np.int64),
            "pair_item": np.concatenate(pair_item).astype(np.int64)}


def mean_features(Xt, flat, offsets, device, chunk=200_000):
    import torch
    import torch.nn.functional as F
    flat_t = torch.from_numpy(flat).to(device)
    out = []
    for s in range(0, len(offsets), chunk):
        e = min(len(offsets), s + chunk)
        lo, hi = offsets[s], (offsets[e] if e < len(offsets) else len(flat))
        off = torch.from_numpy(offsets[s:e] - lo).to(device)
        out.append(F.embedding_bag(flat_t[lo:hi], Xt, off, mode="mean"))
    return torch.cat(out)


def build_model(d_in, p):
    import torch.nn as nn

    def mlp():
        return nn.Sequential(nn.Linear(d_in, p["hidden"]), nn.ReLU(), nn.Dropout(p["dropout"]),
                             nn.Linear(p["hidden"], p["emb_dim"]))

    model = nn.Module()
    model.user, model.item = mlp(), mlp()
    return model


def train(X, samples, p, seed, device, log=print):
    import torch
    import torch.nn.functional as F
    set_seed(seed)
    Xt = torch.from_numpy(X).to(device)
    U = mean_features(Xt, samples["flat"], samples["offsets"], device)
    keys = torch.from_numpy(samples["pair_key"]).to(device)
    items = torch.from_numpy(samples["pair_item"]).to(device)
    model = build_model(X.shape[1], p).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n, bs = len(keys), p["batch_size"]
    for epoch in range(p["epochs"]):
        model.train()
        order = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for s in range(0, n, bs):
            b = order[s:s + bs]
            kb, ib = keys[b], items[b]
            u = F.normalize(model.user(U[kb]), dim=1)
            v = F.normalize(model.item(Xt[ib]), dim=1)
            logits = u @ v.T / p["temperature"]
            same = ib[None, :] == ib[:, None]          # 같은 상품이 배치에 또 있으면 부정 샘플에서 제외
            same.fill_diagonal_(False)
            logits = logits.masked_fill(same, -1e4)
            loss = F.cross_entropy(logits, torch.arange(len(b), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        log(f"epoch {epoch + 1}/{p['epochs']} loss {total / n:.4f}")
    return model


def predict(model, X, flat, offsets, k, device, chunk=4096):
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).to(device)
        V = F.normalize(model.item(Xt), dim=1)
        Uin = mean_features(Xt, flat, offsets, device)
        out = []
        for s in range(0, len(Uin), chunk):
            u = F.normalize(model.user(Uin[s:s + chunk]), dim=1)
            out.append((u @ V.T).topk(k, dim=1).indices.int().cpu().numpy())
    return np.concatenate(out)


def run(cfg, exps, paths, cond, seed, split, group, mode="final", hp_name=None, device=None, log=print):
    """실험 1회. 드라이브에 run_info.json이 이미 있으면 건너뛴다 (중단 후 이어 돌리기 가능)."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "final":
        name, out_dir, overrides = f"{cond}_seed{seed}", paths.run_dir(group, split, f"{cond}_seed{seed}"), None
    else:
        name = f"{cond}_seed{seed}_{hp_name}"
        out_dir = paths.outputs / "tuning" / split / name
        overrides = cfg["train"]["candidates"][hp_name]
    if (out_dir / "run_info.json").exists():
        log(f"[skip] {out_dir} 이미 완료")
        return read_json(out_dir / "run_info.json")

    p = train_params(cfg, overrides)
    spec = exps["conditions"][cond]
    stats = read_json(paths.split_dir(split) / "stats.json")
    history, valid_targets, test_targets = load_split(paths, split)
    start = pd.Timestamp(stats["valid_start"] if mode == "tune" else stats["test_start"])
    targets = valid_targets if mode == "tune" else test_targets
    history = history[history["t_dat"] < start]

    t0 = time.time()
    X = item_features(paths, spec, seed)
    samples = build_samples(history, start, p["label_weeks"], p["max_hist"])
    log(f"[{name}] {split}/{mode} 입력 {X.shape}, 학습 쌍 {len(samples['pair_key']):,}, device {device}")
    model = train(X, samples, p, seed, device, log)
    users = np.sort(targets["user"].unique())
    flat, offsets, counts = user_history(history, users, p["max_hist"])
    cands = predict(model, X, flat, offsets, cfg["eval"]["k"], device)

    info = {"name": name, "cond": cond, "spec": spec, "seed": seed, "split": split, "group": group, "mode": mode,
            "hp_name": hp_name, "params": p, "n_items": int(X.shape[0]), "input_dim": int(X.shape[1]),
            "n_train_pairs": int(len(samples["pair_key"])), "n_users": int(len(users)),
            "users_without_history": int((counts == 0).sum()), "train_seconds": round(time.time() - t0, 1),
            "created_at": now(), **environment_info()}
    if mode == "tune":
        info["valid_recall@300"] = metrics.recall_user_mean(cands, users, targets, cfg["eval"]["k"])
        log(f"[{name}] valid Recall@300 = {info['valid_recall@300']:.4f}")

    tmp = local_tmp_dir()
    np.save(tmp / "candidates.npy", cands.astype(np.int32))
    np.save(tmp / "users.npy", users.astype(np.int64))
    write_json(tmp / "run_info.json", info)
    publish_dir(tmp, out_dir)
    log(f"[{name}] 저장 → {out_dir}")
    return info


def run_group(cfg, exps, paths, group, log=print):
    for block in exps["groups"][group]:
        for seed in block["seeds"]:
            for cond in block["conds"]:
                run(cfg, exps, paths, cond, seed, block["split"], group, log=log)
