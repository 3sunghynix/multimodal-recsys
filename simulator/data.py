"""Source-qualified raw IDs, train-only setup, and source-specific popularity."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def item_key(source, item_id):
    return f"{source}:{int(item_id)}"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def derived_seed(seed, *parts):
    """Stable across processes; deliberately excludes model/variant names."""
    value = json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


@dataclass
class Prepared:
    items: pd.DataFrame
    train: pd.DataFrame
    positive: pd.DataFrame
    stats: dict


def prepare(items, events):
    items, events = items.copy(), events.copy()
    for label, df, columns in (
        ("items", items, {"source", "item_id", "brand_norm", "cat_target", "cat_type", "price", "price_source"}),
        ("events", events, {"source", "item_id", "split", "feedback", "ts", "type"}),
    ):
        if columns - set(df.columns):
            raise ValueError(f"{label}: missing columns {sorted(columns - set(df.columns))}")
        if df[["source", "item_id"]].isna().any().any():
            raise ValueError(f"{label}: missing source/item_id")
        if not set(df.source) <= {"amazon", "hm"}:
            raise ValueError(f"{label}: source must be amazon or hm")
        ids = pd.to_numeric(df.item_id, errors="raise")
        if not np.isfinite(ids).all() or (ids < 0).any() or (ids % 1 != 0).any():
            raise ValueError(f"{label}: nonnegative integer item_id required")
        df["item_id"] = ids.astype("int64")
        df["item_key"] = [item_key(s, i) for s, i in zip(df.source, df.item_id)]
    if items.item_key.duplicated().any():
        raise ValueError("Duplicate (source, item_id) in catalog")
    # Never read valid/test timestamps, feedback or actions for feature construction.
    train = events.loc[events.split == "train"].copy()
    if train.empty:
        raise ValueError("No split=train rows; do not substitute valid/test")
    train["ts"] = pd.to_datetime(train.ts, errors="raise", utc=True)
    train["feedback"] = pd.to_numeric(train.feedback, errors="raise")
    if train.ts.isna().any() or not np.isfinite(train.feedback).all():
        raise ValueError("Missing train timestamp or nonfinite feedback")
    if not set(train.item_key) <= set(items.item_key):
        raise ValueError("Train events refer to products missing from the provided catalog")
    items["brand_norm"] = items.brand_norm.fillna("").astype(str).str.strip()
    items["category"] = items.cat_target.fillna("").astype(str).str.strip() + " > " + items.cat_type.fillna("").astype(str).str.strip()
    items["price"] = pd.to_numeric(items.price, errors="coerce")
    ready = (np.isfinite(items.price) & (items.price > 0) & items.brand_norm.ne("")
             & items.cat_target.fillna("").astype(str).str.strip().ne("")
             & items.cat_type.fillna("").astype(str).str.strip().ne(""))
    stats = {"catalog_rows": len(items), "blocked_catalog_items_missing_price_brand_or_category": int((~ready).sum()),
             "train_rows": len(train), "ignored_nontrain_rows": len(events) - len(train),
             "timestamp_interpretation": "UTC; naive timestamps interpreted as UTC (proposal)",
             "catalog_scope": "only supplied items; local dev subset is not full service catalog"}
    items = items.loc[ready].sort_values(["source", "item_id"]).reset_index(drop=True)
    positive = train.loc[(train.feedback > 0) & (train.type == "purchase") & train.item_key.isin(items.item_key)].copy()
    if positive.empty:
        raise ValueError("No positive train purchase rows with usable product features")
    counts = positive.item_key.value_counts()
    items["positive_train_count"] = items.item_key.map(counts).fillna(0).astype(int)
    items["popularity_affinity"] = 0.5
    windows = {}
    for source in sorted(items.source.unique()):
        history = train.loc[train.source == source]
        if history.empty:
            windows[source] = {"status": "no_train_history", "degenerate": True}
            continue
        end = history.ts.max().normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=7)
        recent = positive.loc[(positive.source == source) & (positive.ts >= start) & (positive.ts < end)]
        counts = recent.item_key.value_counts()
        mask = items.source == source
        values = items.loc[mask, "item_key"].map(counts).fillna(0)
        equal = values.nunique() <= 1
        ranks = pd.Series(0.5, index=values.index) if equal else values.rank(method="average", pct=True)
        items.loc[mask, "popularity_affinity"] = ranks
        windows[source] = {"start_inclusive": start.isoformat(), "end_exclusive": end.isoformat(),
                           "positive_rows": len(recent), "degenerate": bool(equal)}
    stats["popularity_windows"] = windows
    return Prepared(items, train, positive, stats)


def load_data(directory):
    directory = Path(directory)
    return prepare(pd.read_parquet(directory / "items.parquet"), pd.read_parquet(directory / "events.parquet"))
