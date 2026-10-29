"""UME: 텍스트 Two-Tower와 이미지 Two-Tower를 따로 학습하고, 고객마다 두 점수를 표준화해 더한다 (논문 방식, 서비스용).

설계: docs/project/추천모델설계.md. 논문 코드(paper/msrs_paper/two_tower.py)와 같은 구조, 다른 점만 적는다.

모델 (모달리티마다 하나, ID 없음)
  상품 타워: 상품 콘텐츠 임베딩 → Linear(512) → ReLU → Dropout → Linear(128) → L2
  고객 타워: 고객의 최근 긍정 상품 최대 50개 콘텐츠 임베딩 평균 → 같은 구조의 별도 MLP → L2
  학습: 배치 안의 다른 상품을 부정 예시로 쓰는 softmax (temperature 0.05). 같은 상품이 배치에 또 있으면 부정에서 제외
학습 표본 (이벤트 단위 — 논문은 주 단위)
  고객의 긍정 이벤트(feedback > 0)를 시간순으로 놓고, 이벤트마다 (그보다 **이전 시각**의 긍정 이벤트 최대 50개 → 그 상품).
  같은 시각(H&M은 같은 날)의 다른 구매는 이력에 넣지 않는다 (같은 장바구니가 정답을 알려 주는 누수 방지).
  정답 가중치 = feedback (평점 3점 0.5, 그 외 1). feedback_weight=False면 모두 1 (E4 비교).
추천
  고객 이력(평가 때는 학습 이력의 마지막 50개) → 고객 벡터, 그 데이터셋 전체 상품과 내적
  UME = z(텍스트 점수) + z(이미지 점수), z는 고객마다 전체 상품 점수 기준 (w = 1)
  브랜드 가산(E3): + β · [상품 브랜드 = 고객 이력의 최다 브랜드]   (z 단위)

실행 (Colab, 저장소 루트):
    python -m recsys.ume --root /content/drive/MyDrive/msrs_project --source hm --split valid --out results/ume
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recsys import baselines
from recsys.eval import build_eval_set
from recsys.run_eval import image_embedding_by_item, load_dataset, metrics_for

DEFAULT = dict(hidden=512, emb_dim=128, dropout=0.1, lr=5e-4, weight_decay=0.0, temperature=0.05,
               batch_size=2048, epochs=8, max_hist=50, seed=42)
DATA_DIR = {"amazon": "amazon_v2", "hm": "hm"}
POP_RECENT = {"hm": 7, "amazon": None}          # 기준선에서 가장 좋았던 인기 설정 (오프라인기준선.md)


# ---------------------------------------------------------------- 입력

def item_features(root: Path, source: str, n_items: int, kind: str = "pca"):
    """{"text": (n_items, d), "image": (n_items, d)} 상품 id로 인덱싱. 임베딩 없는 상품은 0 벡터.

    kind=pca: embeddings/v2 (서비스 DB와 같은 256차원), raw: embeddings/{source}/*_raw.f16.npy (원본 384 · 768).
    """
    out = {}
    if kind == "pca":
        cat = pd.read_parquet(root / "embeddings" / "v2" / "catalog.parquet")
        sel = cat[cat.source == source]
        for m in ("text", "image"):
            mat = np.load(root / "embeddings" / "v2" / f"{m}_pca256.npy", mmap_mode="r")
            x = np.zeros((n_items, mat.shape[1]), np.float32)
            x[sel.item_id.values] = mat[sel.row.values]
            out[m] = x
    elif kind == "raw":
        d = root / "embeddings" / source
        ids = np.load(d / "item_ids.npy")
        keep = ids < n_items
        for m, name in (("text", "text_minilm_raw.f16.npy"), ("image", "image_marqo_raw.f16.npy")):
            mat = np.load(d / name, mmap_mode="r")
            x = np.zeros((n_items, mat.shape[1]), np.float32)
            x[ids[keep]] = np.asarray(mat[keep], np.float32)
            out[m] = x
    else:
        raise ValueError(kind)
    return out


@dataclass
class Sequences:
    """긍정 이벤트를 (고객, 시각) 순으로 펼친 배열. 이력 = items[start:end]."""
    items: np.ndarray        # (E,) 상품 id
    users: np.ndarray        # (E,) 고객 id
    weight: np.ndarray       # (E,) feedback
    start: np.ndarray        # (E,) 이 이벤트를 정답으로 할 때 이력 시작
    end: np.ndarray          # (E,) 이력 끝 (같은 시각 이벤트 제외)

    def samples(self):
        """이력이 1개 이상인 이벤트 = 학습 표본 번호."""
        return np.flatnonzero(self.end > self.start)

    def last_history(self, users: np.ndarray, max_hist: int):
        """고객마다 마지막 max_hist개 이력 (start, end). 이력이 없으면 길이 0."""
        u = pd.Series(self.users)
        first = u.groupby(u).head(1)
        last_end = u.groupby(u).size()
        s = pd.Series(first.index.values, index=first.values).reindex(users)
        n = last_end.reindex(users)
        e = (s + n).fillna(0).astype(np.int64).values
        st = np.maximum(s.fillna(0).astype(np.int64).values, e - max_hist)
        return st, e


def build_sequences(history: pd.DataFrame, max_hist: int) -> Sequences:
    pos = history[history.feedback > 0].sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)
    idx = np.arange(len(pos))
    first = idx - pos.groupby("user_id").cumcount().values               # 고객의 첫 이벤트 위치
    end = idx - pos.groupby(["user_id", "ts"]).cumcount().values          # 같은 시각 이벤트 앞까지
    start = np.maximum(first, end - max_hist)
    return Sequences(pos.item_id.values.astype(np.int64), pos.user_id.values, pos.feedback.values.astype(np.float32),
                     start.astype(np.int64), end.astype(np.int64))


# ---------------------------------------------------------------- 모델

def _mlp(d_in, p):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d_in, p["hidden"]), nn.ReLU(), nn.Dropout(p["dropout"]), nn.Linear(p["hidden"], p["emb_dim"]))


def _bag(flat, start, length, device):
    """(start, length) 묶음들 → embedding_bag 입력 (인덱스, offsets)."""
    import torch
    offsets = torch.cumsum(length, 0) - length
    total = int(length.sum())
    pos = torch.arange(total, device=device) - torch.repeat_interleave(offsets, length)
    return flat[torch.repeat_interleave(start, length) + pos], offsets


def train_tower(X: np.ndarray, seq: Sequences, p: dict, device: str, feedback_weight: bool = True, log=print):
    """모달리티 하나의 Two-Tower 학습. 반환: {"user": nn.Module, "item": nn.Module}."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    Xt = torch.from_numpy(X).to(device)
    flat = torch.from_numpy(seq.items).to(device)
    sid = seq.samples()
    starts = torch.from_numpy(seq.start[sid]).to(device)
    lens = torch.from_numpy(seq.end[sid] - seq.start[sid]).to(device)
    labels = torch.from_numpy(seq.items[sid]).to(device)
    w = torch.from_numpy(seq.weight[sid] if feedback_weight else np.ones(len(sid), np.float32)).to(device)
    user, item = _mlp(X.shape[1], p).to(device), _mlp(X.shape[1], p).to(device)
    opt = torch.optim.Adam(list(user.parameters()) + list(item.parameters()), lr=p["lr"], weight_decay=p["weight_decay"])
    gen = torch.Generator().manual_seed(p["seed"])
    n, bs = len(sid), p["batch_size"]
    for epoch in range(p["epochs"]):
        user.train(); item.train()
        order = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for s in range(0, n, bs):
            b = order[s:s + bs]
            idx, off = _bag(flat, starts[b], lens[b], device)
            u = F.normalize(user(F.embedding_bag(idx, Xt, off, mode="mean")), dim=1)
            v = F.normalize(item(Xt[labels[b]]), dim=1)
            logits = u @ v.T / p["temperature"]
            same = labels[b][None, :] == labels[b][:, None]
            same.fill_diagonal_(False)
            logits = logits.masked_fill(same, -1e4)
            ce = F.cross_entropy(logits, torch.arange(len(b), device=device), reduction="none")
            loss = (ce * w[b]).sum() / w[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        log(f"  epoch {epoch + 1}/{p['epochs']} loss {total / n:.4f}")
    user.eval(); item.eval()
    return {"user": user, "item": item}


def tower_vectors(model, X, seq: Sequences, users, catalog, max_hist, device, chunk=8192):
    """정규화된 고객 벡터 (len(users), d)와 상품 벡터 (len(catalog), d) — numpy."""
    import torch
    import torch.nn.functional as F
    Xt = torch.from_numpy(X).to(device)
    flat = torch.from_numpy(seq.items).to(device)
    st, en = seq.last_history(users, max_hist)
    U = []
    with torch.no_grad():
        V = F.normalize(model["item"](Xt[torch.from_numpy(catalog).to(device)]), dim=1).cpu().numpy()
        for s in range(0, len(users), chunk):
            a = torch.from_numpy(st[s:s + chunk]).to(device)
            ln = torch.from_numpy(en[s:s + chunk] - st[s:s + chunk]).to(device)
            idx, off = _bag(flat, a, ln, device)
            U.append(F.normalize(model["user"](F.embedding_bag(idx, Xt, off, mode="mean")), dim=1).cpu().numpy())
    return np.concatenate(U), V


def popularity_score(history: pd.DataFrame, catalog: np.ndarray, recent_days: int | None) -> np.ndarray:
    """상품별 인기 점수 (n_catalog,): 학습 이력 긍정 이벤트 수(최근 N일)의 log1p를 카탈로그 전체로 표준화."""
    h = history[history.feedback > 0]
    if recent_days is not None:
        h = h[h.ts >= h.ts.max() - pd.Timedelta(days=recent_days)]
    cnt = np.log1p(h.item_id.value_counts().reindex(catalog).fillna(0).values).astype(np.float32)
    return (cnt - cnt.mean()) / (cnt.std() + 1e-6)


def recommend(U: dict, V: dict, catalog: np.ndarray, k: int, device: str, boost=None, item_bias=None, chunk=2048):
    """U · V: {"text": ..., "image": ...}. 모달리티 하나면 원래 점수, 둘이면 고객별 z 합.
    boost: (고객별 최다 브랜드 (n_users,), 상품 브랜드 (n_catalog,), β) 또는 None.
    item_bias: 모든 고객에게 같은 상품별 가산 (n_catalog,) — 예: γ · 인기 점수. 반환: (n_users, k) 상품 id."""
    import torch
    mods = list(U)
    Vt = {m: torch.from_numpy(V[m]).to(device) for m in mods}
    bias = None if item_bias is None else torch.from_numpy(np.ascontiguousarray(item_bias, np.float32)).to(device)[None, :]
    out = []
    for s in range(0, len(U[mods[0]]), chunk):
        total = 0
        for m in mods:
            sc = torch.from_numpy(U[m][s:s + chunk]).to(device) @ Vt[m].T
            total = total + (sc if len(mods) == 1 else (sc - sc.mean(1, keepdim=True)) / (sc.std(1, keepdim=True) + 1e-6))
        if boost is not None:
            top_brand, item_brand, beta = boost
            tb = torch.from_numpy(np.ascontiguousarray(top_brand[s:s + chunk])).to(device)[:, None]
            ib = torch.from_numpy(item_brand).to(device)[None, :]
            total = total + beta * ((ib == tb) & (ib >= 0)).float()
        if bias is not None:
            total = total + bias
        out.append(total.topk(k, dim=1).indices.cpu().numpy())
    return catalog[np.concatenate(out)]


def mix_popular(rec: np.ndarray, pop: np.ndarray, n_pop: int, k: int) -> np.ndarray:
    """앞 (k − n_pop)개는 rec, 나머지는 rec에 없는 인기 상품 순서대로 (논문의 인기 슬롯; E2)."""
    out = np.empty((len(rec), k), np.int64)
    for i in range(len(rec)):
        head = rec[i, :k - n_pop]
        tail = pop[~np.isin(pop, head)][:n_pop]
        out[i] = np.concatenate([head, tail])
    return out


def save_tower(model, path: Path, p: dict, meta: dict):
    """서빙용 산출물: MLP 가중치 npz (numpy로 계산 가능) + config.json."""
    path.mkdir(parents=True, exist_ok=True)
    w = {f"{side}.{k}": v.detach().cpu().numpy() for side in ("user", "item") for k, v in model[side].state_dict().items()}
    np.savez(path / "weights.npz", **w)
    (path / "config.json").write_text(json.dumps({**p, **meta}, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 실험 (E0 · E1 · E2 · E3)

def run(root: Path, source: str, split: str = "valid", kind: str = "pca", out: Path | None = None,
        feedback_weight: bool = True, epochs: int | None = None, betas=(0.5, 1.0, 2.0), gammas=(0.5, 1.0, 2.0, 4.0),
        device: str | None = None, ks=(12, 300), log=print):
    """E0 · E1 · E2 · E3을 한 번에: 두 타워 학습 → 인기 · 텍스트 · 이미지 · UME · 인기 슬롯 · 브랜드 가산을 같은 지표로 채점."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    p = {**DEFAULT, **({"epochs": epochs} if epochs else {})}
    events, items, brand_of, n_items = load_dataset(root / "processed" / DATA_DIR[source])
    es = build_eval_set(events, split)
    catalog = np.sort(items.item_id.values.astype(np.int64))
    X = item_features(root, source, n_items, kind)
    seq = build_sequences(es.history, p["max_hist"])
    log(f"[{source} · {split} · {kind}] 학습 표본 {len(seq.samples()):,} · 평가 고객 {len(es.users):,} · 상품 {len(catalog):,} · {device}")

    U, V, timing = {}, {}, {}
    for m in ("text", "image"):
        t = time.time()
        model = train_tower(X[m], seq, p, device, feedback_weight, log)
        timing[f"train_{m}_s"] = round(time.time() - t, 1)
        U[m], V[m] = tower_vectors(model, X[m], seq, es.users, catalog, p["max_hist"], device)
        if out:
            save_tower(model, out / f"{source}_{split}_{kind}" / m, p,
                       {"source": source, "split": split, "input": kind, "modality": m, "feedback_weight": feedback_weight,
                        "trained_until": str(es.history.ts.max())})
        log(f"  {m} 학습 {timing[f'train_{m}_s']}s")

    emb = image_embedding_by_item(root / "embeddings" / "v2", source, n_items)
    k0, K = ks[0], max(ks)
    recs = {"popularity": baselines.popularity(es.history, es.users, K, POP_RECENT[source]),
            "text": recommend({"text": U["text"]}, {"text": V["text"]}, catalog, K, device),
            "image": recommend({"image": U["image"]}, {"image": V["image"]}, catalog, K, device)}
    t = time.time()
    recs["ume"] = recommend(U, V, catalog, K, device)
    timing["ume_recommend_ms_per_user"] = round((time.time() - t) / len(es.users) * 1000, 3)
    # E2: 화면(상위 k0)에 인기 상품 2 · 4자리, 후보(K)는 논문 설정 인기 (K − K/15) + UME K/15 (300이면 280 + 20)
    for n_pop in (2, 4):
        head = mix_popular(recs["ume"][:, :k0], recs["popularity"][0], n_pop, k0)
        recs[f"ume+pop{n_pop}@{k0}"] = np.stack([np.concatenate([h, u[~np.isin(u, h)][:K - k0]])      # 뒤는 앞에 없는 UME 후보
                                                 for h, u in zip(head, recs["ume"])])
    n_ume = K // 15
    recs[f"pop{K - n_ume}+ume{n_ume}"] = np.stack([np.concatenate([p_[:K - n_ume], u_[~np.isin(u_, p_[:K - n_ume])][:n_ume]])
                                                   for p_, u_ in zip(recs["popularity"], recs["ume"])])
    # E2b: 인기 점수 가산 — 최종 = UME z 합 + γ · 인기 점수(표준화). 슬롯과 달리 순위 전체에 인기가 섞인다
    pop_z = popularity_score(es.history, catalog, POP_RECENT[source])
    for g in gammas:
        recs[f"ume+popscore{g}"] = recommend(U, V, catalog, K, device, item_bias=g * pop_z)
    # E3: 브랜드 가산 (브랜드가 여럿인 데이터셋만), 인기 점수와 함께도
    if source == "amazon":
        b = pd.DataFrame({"user_id": es.history.user_id.values, "brand": brand_of[es.history.item_id.values]})
        b = b[b.brand >= 0]
        top = b.groupby(["user_id", "brand"]).size().reset_index(name="c").sort_values(["user_id", "c"], ascending=[True, False])
        top_brand = top.drop_duplicates("user_id").set_index("user_id").brand.reindex(es.users).fillna(-1).astype(np.int64).values
        boost = lambda beta: (top_brand, brand_of[catalog], beta)  # noqa: E731
        for beta in betas:
            recs[f"ume+brand{beta}"] = recommend(U, V, catalog, K, device, boost=boost(beta))
            for g in gammas:
                recs[f"ume+popscore{g}+brand{beta}"] = recommend(U, V, catalog, K, device, boost=boost(beta), item_bias=g * pop_z)

    result = {"source": source, "split": split, "input": kind, "feedback_weight": feedback_weight, "params": p,
              "eval": es.summary(), "timing": timing, "models": {}}
    for name, rec in recs.items():
        srt = np.sort(rec, axis=1)
        assert rec.shape[1] == K and (srt[:, 1:] != srt[:, :-1]).all(), f"{name}: 추천 목록 길이 · 중복 오류"
        result["models"][name] = m = metrics_for(rec, es, brand_of, len(catalog), emb, ks)
        log(f"  {name:<18} R@{k0} {m[f'recall@{k0}']:.4f}  NDCG@{k0} {m[f'ndcg@{k0}']:.4f}  MAP@{k0} {m[f'map@{k0}']:.4f}  "
            f"R@{K} {m[f'recall@{K}']:.4f}  브랜드일치@{k0} {m[f'brand_alignment@{k0}']:.4f}")
    if out:
        out.mkdir(parents=True, exist_ok=True)
        tag = f"{source}_{split}_{kind}" + ("" if feedback_weight else "_nofb")
        (out / f"{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="드라이브 msrs_project 경로")
    ap.add_argument("--source", required=True, choices=list(DATA_DIR))
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--input", default="pca", choices=["pca", "raw"], help="E0: 서비스 256차원 vs 원본 임베딩")
    ap.add_argument("--no-feedback-weight", action="store_true", help="E4: 정답 가중치 없이")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    run(a.root, a.source, a.split, a.input, a.out, not a.no_feedback_weight, a.epochs)


if __name__ == "__main__":
    main()
