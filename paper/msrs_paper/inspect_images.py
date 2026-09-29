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
    """외관 그룹별 표본 이미지 격자와, 사람이 채울 확인표(csv)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    records = []
    for app, g in items.dropna(subset=["appearance"]).groupby("appearance"):
        pick = g.iloc[rng.choice(len(g), min(per_group, len(g)), replace=False)]
        aids = pick["article_id"].values
        rows = [aids[i:i + cols] for i in range(0, len(aids), cols)]
        titles = [[f"{a}\n{items.set_index('article_id').at[a, 'graphical_appearance_name']}" for a in r] for r in rows]
        _grid(zip_path, rows, out_dir / f"label_check_{app}.png", titles)
        for a in aids:
            records.append({"appearance": app, "article_id": int(a), "label_correct": ""})
    pd.DataFrame(records).to_csv(out_dir / "label_check.csv", index=False)
    return out_dir
