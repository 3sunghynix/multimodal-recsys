"""눈으로 확인하는 도구: 임베딩 최근접 이웃, 외관 그룹 라벨 표본 검사."""
import zipfile

import numpy as np

from .embeddings import read_image


def _grid(zip_path, article_rows, out_path, titles=None, thumb=160):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rows, n_cols = len(article_rows), max(len(r) for r in article_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 2.0), squeeze=False)
    with zipfile.ZipFile(zip_path) as zf:
        for i, row in enumerate(article_rows):
            for j in range(n_cols):
                ax = axes[i][j]
                ax.axis("off")
                if j < len(row):
                    img = read_image(zf, int(row[j]))
                    img.thumbnail((thumb, thumb))
                    ax.imshow(img)
                    ax.set_title(str(row[j]) if titles is None else titles[i][j], fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def neighbor_grid(items, emb, zip_path, out_path, n_query=5, n_neighbors=6, seed=42):
    """무작위 상품 n_query개와 임베딩 최근접 이웃. 첫 열이 기준 상품."""
    rng = np.random.default_rng(seed)
    q = rng.choice(len(items), n_query, replace=False)
    sims = emb[q] @ emb.T
    nb = np.argsort(-sims, axis=1)[:, :n_neighbors + 1]
    aids = items.sort_values("idx")["article_id"].values
    rows = [aids[r] for r in nb]
    return _grid(zip_path, rows, out_path)


def label_sample(items, zip_path, out_dir, per_group=40, cols=8, seed=42):
    """외관 그룹별 표본 이미지 격자와, 사람이 채울 확인표(csv). 원 라벨도 함께 기록해 사후 집계 용이."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    lookup = items.set_index("article_id")["graphical_appearance_name"]
    records = []
    for app, g in items.dropna(subset=["appearance"]).groupby("appearance"):
        pick = g.iloc[rng.choice(len(g), min(per_group, len(g)), replace=False)]
        aids = pick["article_id"].values
        rows = [aids[i:i + cols] for i in range(0, len(aids), cols)]
        titles = [[f"{a}\n{lookup.at[a]}" for a in r] for r in rows]
        _grid(zip_path, rows, out_dir / f"label_check_{app}.png", titles)
        for a in aids:
            records.append({"appearance": app, "graphical_appearance_name": lookup.at[a],
                            "article_id": int(a), "label_correct": ""})
    pd.DataFrame(records).to_csv(out_dir / "label_check.csv", index=False)
    return out_dir


def _wilson(k, n, z=1.96):
    """Wilson 이항 신뢰구간. 표본이 작아 정규 근사보다 안정적."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - h) / d, (c + h) / d


def label_agreement_report(csv_path, items=None, out_dir=None):
    """채운 label_check.csv를 읽어 외관 그룹별·원 라벨별 일치율(Wilson 95% CI)을 계산.

    csv_path: label_sample()이 만든 csv를 label_correct 열에 1/0으로 채운 파일.
    items: 옛 스키마(원 라벨이 csv에 없음)일 때 join 용도. None이면 csv의 값만 사용.
    반환: {"by_group": DataFrame, "by_raw": DataFrame, "total": dict, "sentence": str, "n_missing": int}
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "graphical_appearance_name" not in df.columns:
        if items is None:
            raise ValueError("csv에 graphical_appearance_name이 없습니다. items를 넘겨 join하세요.")
        df = df.merge(items[["article_id", "graphical_appearance_name"]], on="article_id", how="left")
    n_total = len(df)
    df = df[pd.to_numeric(df["label_correct"], errors="coerce").isin([0, 1])].copy()
    df["label_correct"] = df["label_correct"].astype(int)
    n_missing = n_total - len(df)
    if len(df) == 0:
        raise ValueError(f"{csv_path}: label_correct에 0/1이 채워지지 않았습니다 (총 {n_total}행)")

    def summary(g):
        k, n = int(g["label_correct"].sum()), len(g)
        lo, hi = _wilson(k, n)
        return pd.Series({"n": n, "correct": k, "rate": round(k / n, 4),
                          "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})

    by_group = df.groupby("appearance")[["label_correct"]].apply(summary).reset_index()
    by_raw = df.groupby(["appearance", "graphical_appearance_name"])[["label_correct"]].apply(summary).reset_index()
    total = summary(df).to_dict()
    total["appearance"] = "TOTAL"

    parts = [f"{r['appearance']} {int(r['correct'])}/{int(r['n'])} = {r['rate'] * 100:.1f}% "
             f"[{r['ci_lo'] * 100:.1f}, {r['ci_hi'] * 100:.1f}]" for _, r in by_group.iterrows()]
    sentence = ("외관 그룹별 라벨 일치율(사람 검수, Wilson 95% CI): " + " · ".join(parts) +
                f" · 전체 {total['rate'] * 100:.1f}% ({int(total['correct'])}/{int(total['n'])}).")

    if out_dir is not None:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        by_group.to_csv(out_dir / "label_agreement_by_group.csv", index=False)
        by_raw.to_csv(out_dir / "label_agreement_by_raw_label.csv", index=False)
        (out_dir / "label_agreement_sentence.txt").write_text(sentence, encoding="utf-8")
    return {"by_group": by_group, "by_raw": by_raw, "total": total,
            "sentence": sentence, "n_missing": n_missing}
