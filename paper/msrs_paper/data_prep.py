"""H&M 원본 → 상품 표, 시간 기준 분할, 정답(타깃) 표.

모든 분할(main, alt)은 같은 상품 목록(items.parquet의 idx 순서)을 공유한다.
신규 상품·재구매 판정은 2018년부터의 전체 구매 기록 기준이다.
"""
import re

import numpy as np
import pandas as pd

from .utils import write_json

APPEARANCE_ORDER = ["solid", "pattern", "texture"]
TARGET_COLS = ["user", "item_idx", "is_repurchase", "is_new_item", "pgroup", "appearance", "product_code"]


def customer_to_int(ids):
    """64자리 16진수 customer_id의 마지막 16자리를 int64로 (Kaggle에서 흔히 쓰는 방식, 충돌 사실상 없음)."""
    return np.array([int(s[-16:], 16) for s in ids], dtype=np.uint64).view(np.int64)


def load_transactions(csv_path, chunksize=5_000_000):
    parts = []
    for ch in pd.read_csv(csv_path, usecols=["t_dat", "customer_id", "article_id"],
                          dtype={"customer_id": str, "article_id": np.int64}, chunksize=chunksize):
        parts.append(pd.DataFrame({
            "t_dat": pd.to_datetime(ch["t_dat"].values),
            "user": customer_to_int(ch["customer_id"].values),
            "article_id": ch["article_id"].values.astype(np.int64),
        }))
    return pd.concat(parts, ignore_index=True)


def _invert(groups):
    return {v: k for k, values in groups.items() for v in values}


def check_plan(articles, plan):
    """매핑에 없는 값이 있으면 멈춘다. analysis_plan.yaml을 실제 값으로 먼저 확정해야 한다."""
    problems = {}
    for col, groups, exclude in (("graphical_appearance_name", plan["appearance_groups"], plan["appearance_exclude"]),
                                 ("product_group_name", plan["product_groups"], plan["product_group_exclude"])):
        known = set(_invert(groups)) | set(exclude)
        missing = sorted(set(articles[col].dropna().unique()) - known)
        if missing:
            problems[col] = missing
    return problems


def build_item_table(articles, plan):
    a = articles.copy()
    a["appearance"] = a["graphical_appearance_name"].map(_invert(plan["appearance_groups"]))
    a["pgroup"] = a["product_group_name"].map(_invert(plan["product_groups"]))
    pattern_re = re.compile(plan["pattern_keywords_regex"], re.IGNORECASE)
    desc = a["detail_desc"].fillna("")
    a["desc_mentions_pattern"] = desc.str.contains(pattern_re)
    a["text"] = (a["prod_name"].fillna("") + ". " + a["product_type_name"].fillna("") + ". "
                 + a["colour_group_name"].fillna("") + ". " + desc).str.strip()
    return a


def select_items(tx, items, image_ids, window_weeks, alt_shift_weeks):
    """두 분할의 전체 기간(가장 이른 학습 시작 ~ 마지막 날)에 팔린 상품 중 이미지가 있는 것."""
    max_date = tx["t_dat"].max()
    earliest = max_date - pd.Timedelta(weeks=window_weeks + 2 + alt_shift_weeks) + pd.Timedelta(days=1)
    sold = tx.loc[tx["t_dat"] >= earliest, "article_id"].unique()
    keep = items[items["article_id"].isin(sold) & items["article_id"].isin(image_ids)]
    keep = keep.sort_values("article_id").reset_index(drop=True)
    keep.insert(0, "idx", np.arange(len(keep), dtype=np.int32))
    cols = ["idx", "article_id", "product_code", "prod_name", "product_type_name", "product_group_name",
            "graphical_appearance_name", "colour_group_name", "pgroup", "appearance", "desc_mentions_pattern", "text"]
    return keep[cols]


def split_dates(max_date, window_weeks, shift_weeks):
    end = max_date - pd.Timedelta(weeks=shift_weeks)            # 테스트 마지막 날 (포함)
    test_start = end - pd.Timedelta(days=6)
    valid_start = test_start - pd.Timedelta(days=7)
    train_start = valid_start - pd.Timedelta(weeks=window_weeks)
    return {"train_start": train_start, "valid_start": valid_start, "test_start": test_start, "end": end}


def _targets(period_rows, tx_full, first_sale, items, period_start):
    t = period_rows[["user", "item_idx", "article_id"]].drop_duplicates(["user", "item_idx"])
    before = tx_full.loc[(tx_full["t_dat"] < period_start) & tx_full["user"].isin(t["user"].unique()),
                         ["user", "article_id"]].drop_duplicates()
    before["is_repurchase"] = True
    t = t.merge(before, on=["user", "article_id"], how="left")
    t["is_repurchase"] = t["is_repurchase"].notna()
    t["is_new_item"] = (t["article_id"].map(first_sale) >= period_start).values
    meta = items.set_index("idx")[["pgroup", "appearance", "product_code"]]
    t = t.join(meta, on="item_idx")
    return t[TARGET_COLS].reset_index(drop=True)


def build_split(tx, items, window_weeks, min_user_purchases, shift_weeks):
    """한 분할의 train / valid 원자료와 valid·test 정답 표."""
    d = split_dates(tx["t_dat"].max(), window_weeks, shift_weeks)
    idx_of = pd.Series(items["idx"].values, index=items["article_id"].values)
    first_sale = tx.groupby("article_id")["t_dat"].min()

    in_items = tx["article_id"].isin(idx_of.index)
    rows = tx.loc[in_items & (tx["t_dat"] >= d["train_start"]) & (tx["t_dat"] <= d["end"])].copy()
    rows["item_idx"] = idx_of.loc[rows["article_id"]].values.astype(np.int32)

    train = rows[rows["t_dat"] < d["valid_start"]]
    counts = train.groupby("user").size()
    users = counts.index[counts >= min_user_purchases]
    rows = rows[rows["user"].isin(users)]

    train = rows[rows["t_dat"] < d["valid_start"]]
    valid = rows[(rows["t_dat"] >= d["valid_start"]) & (rows["t_dat"] < d["test_start"])]
    test = rows[rows["t_dat"] >= d["test_start"]]

    valid_targets = _targets(valid, tx, first_sale, items, d["valid_start"])
    test_targets = _targets(test, tx, first_sale, items, d["test_start"])
    history = pd.concat([train, valid])[["user", "item_idx", "t_dat"]].reset_index(drop=True)

    stats = {k: str(v.date()) for k, v in d.items()}
    stats.update({
        "n_items": int(len(items)), "n_users": int(len(users)),
        "n_train_rows": int(len(train)), "n_valid_rows": int(len(valid)), "n_test_rows": int(len(test)),
        "n_valid_users": int(valid_targets["user"].nunique()), "n_test_users": int(test_targets["user"].nunique()),
        "n_test_targets": int(len(test_targets)),
        "test_repurchase_share": float(test_targets["is_repurchase"].mean()),
        "test_new_item_share": float(test_targets["is_new_item"].mean()),
        "test_cells": test_targets.groupby(["pgroup", "appearance"]).size().rename("n").reset_index()
                                  .to_dict(orient="records"),
    })
    return history, valid_targets, test_targets, stats


def save_split(out_dir, history, valid_targets, test_targets, stats):
    out_dir.mkdir(parents=True, exist_ok=True)
    history.to_parquet(out_dir / "history.parquet", index=False)
    valid_targets.to_parquet(out_dir / "valid_targets.parquet", index=False)
    test_targets.to_parquet(out_dir / "test_targets.parquet", index=False)
    write_json(out_dir / "stats.json", stats)


def load_split(paths, split):
    d = paths.split_dir(split)
    return (pd.read_parquet(d / "history.parquet"), pd.read_parquet(d / "valid_targets.parquet"),
            pd.read_parquet(d / "test_targets.parquet"))


def load_items(paths):
    return pd.read_parquet(paths.processed / "items.parquet")
