"""데이터셋 하나(processed/amazon_v2 또는 processed/hm)를 고정 분할로 평가한다. 지금은 인기 기반 기준 모델.

실행 (저장소 루트에서, Colab이면 드라이브 경로):
    python -m recsys.run_eval --data /content/drive/MyDrive/msrs_project/processed/hm --source hm \
        --embeddings /content/drive/MyDrive/msrs_project/embeddings/v2 --out results/hm_popularity.json
출력: 지표 평균 · 평가 집합 요약 JSON (관리자 화면 · metric_snapshots에 그대로 쓰는 형식)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys import baselines
from recsys.eval import (accuracy_at_k, brand_alignment_at_k, brand_diversity_at_k, build_eval_set, coverage_at_k,
                         intra_list_diversity_at_k, popularity_percentile_at_k, user_brand_loyalty)

KS = (12, 300)


def image_embedding_by_item(emb_dir: Path, source: str, n_items: int):
    """embeddings/v2 → 상품 id로 인덱싱하는 이미지 임베딩 (없는 상품은 0 벡터)."""
    cat = pd.read_parquet(emb_dir / "catalog.parquet")
    img = np.load(emb_dir / "image_pca256.npy")
    sel = cat[cat.source == source]
    out = np.zeros((n_items, img.shape[1]), np.float32)
    out[sel.item_id.values] = img[sel.row.values]
    return out


def load_dataset(data: Path):
    """events, items(item_id · brand_norm), 상품 id → 브랜드 코드 배열, 상품 id 배열 크기."""
    events = pd.read_parquet(data / "events.parquet")
    items = pd.read_parquet(data / "items.parquet", columns=["item_id", "brand_norm"])
    n_items = int(max(items.item_id.max(), events.item_id.max())) + 1
    brand_of = np.full(n_items, -1, np.int64)
    brand_of[items.item_id.values] = pd.factorize(items.brand_norm)[0]
    return events, items, brand_of, n_items


def metrics_for(rec, es, brand_of, n_catalog, emb=None, ks=None):
    """추천 행렬 하나의 지표 묶음 (모든 모델이 같은 함수로 채점). emb: 상품 id로 인덱싱되는 이미지 임베딩."""
    ks = ks or KS
    out, _ = accuracy_at_k(rec, es.users, es.truth, ks)
    for k in ks:
        out[f"coverage@{k}"] = coverage_at_k(rec, k, n_catalog)
        out[f"brand_diversity@{k}"] = brand_diversity_at_k(rec, k, brand_of)
        out[f"popularity_pct@{k}"] = popularity_percentile_at_k(rec, k, es.history)
        out[f"brand_alignment@{k}"] = brand_alignment_at_k(rec, es.users, k, es.history, brand_of)
    if emb is not None:
        out[f"intra_list_diversity@{ks[0]}"] = intra_list_diversity_at_k(rec, ks[0], emb)
    return out


def evaluate(data: Path, source: str, model: str = "popularity", split: str = "test", positive_min: float = 1.0,
             embeddings: Path | None = None, recent_days: int | None = None):
    events, items, brand_of, n_items = load_dataset(data)
    es = build_eval_set(events, split, positive_min)
    if model == "popularity":
        rec = baselines.popularity(es.history, es.users, max(KS), recent_days)
    else:
        raise ValueError(f"모르는 모델: {model}")

    emb = image_embedding_by_item(embeddings, source, n_items) if embeddings is not None else None
    out = {"source": source, "model": model, "params": {"recent_days": recent_days}, "eval": es.summary(),
           "metrics": metrics_for(rec, es, brand_of, len(items), emb)}
    loyalty = user_brand_loyalty(es.history, brand_of)
    out["data"] = {"user_brand_loyalty_median": float(loyalty.median()), "loyal_user_share(>=0.5)": float((loyalty >= 0.5).mean())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--source", required=True, choices=["amazon", "hm"])
    ap.add_argument("--model", default="popularity")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--positive-min", type=float, default=1.0)
    ap.add_argument("--recent-days", type=int, default=None)
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = evaluate(a.data, a.source, a.model, a.split, a.positive_min, a.embeddings, a.recent_days)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
