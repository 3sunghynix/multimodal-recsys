"""UME 학습 · 추천 검증 (합성 데이터, CPU). torch가 없으면 건너뛴다."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from recsys import ume  # noqa: E402


def test_sequences_exclude_same_time_and_negatives():
    h = pd.DataFrame({"user_id": [1, 1, 1, 1, 2, 2], "item_id": [10, 11, 12, 13, 20, 21],
                      "feedback": [1, 1, -1, 0.5, 1, 1],
                      "ts": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-02", "2023-01-05", "2023-01-05"])})
    s = ume.build_sequences(h, max_hist=50)
    # 고객 1의 긍정: 10(1일) · 11(2일) · 13(2일, 0.5). 12는 부정이라 제외
    assert s.items.tolist() == [10, 11, 13, 20, 21]
    # 11 · 13은 같은 날이라 서로의 이력이 아니다 → 둘 다 이력 = [10]
    assert [s.items[a:b].tolist() for a, b in zip(s.start, s.end)] == [[], [10], [10], [], []]
    assert s.samples().tolist() == [1, 2] and s.weight[2] == 0.5
    st, en = s.last_history(np.array([2, 1, 9]), max_hist=2)
    assert [s.items[a:b].tolist() for a, b in zip(st, en)] == [[20, 21], [11, 13], []]


def test_bag_mean_matches_manual():
    X = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    flat = torch.tensor([0, 1, 2, 3, 4, 5])
    idx, off = ume._bag(flat, torch.tensor([0, 2]), torch.tensor([2, 3]), "cpu")
    got = torch.nn.functional.embedding_bag(idx, X, off, mode="mean")
    assert torch.allclose(got, torch.stack([X[0:2].mean(0), X[2:5].mean(0)]))


def toy_root(tmp_path, n_users=300, seed=0):
    """3개 묶음(브랜드 = 묶음)의 상품 60개. 고객은 한 묶음을 좋아하고 그 묶음 상품을 산다."""
    rng = np.random.default_rng(seed)
    n_items, d = 60, 16
    cluster = np.arange(n_items) % 3
    centers = rng.normal(size=(3, d)) * 3
    emb = {m: (centers[cluster] + rng.normal(size=(n_items, d))).astype(np.float32) for m in ("text", "image")}
    rows = []
    for u in range(n_users):
        c = u % 3
        mine = np.flatnonzero(cluster == c)
        days = pd.date_range("2023-01-01", periods=10)
        for t in range(10):
            split = "train" if t < 8 else ("valid" if t == 8 else "test")
            rows.append((u, int(rng.choice(mine)), 1.0, days[t], split))
    root = tmp_path / "root"
    (root / "processed" / "amazon_v2").mkdir(parents=True)
    (root / "embeddings" / "v2").mkdir(parents=True)
    pd.DataFrame(rows, columns=["user_id", "item_id", "feedback", "ts", "split"]).to_parquet(root / "processed/amazon_v2/events.parquet")
    pd.DataFrame({"item_id": np.arange(n_items), "brand_norm": [f"b{c}" for c in cluster]}).to_parquet(root / "processed/amazon_v2/items.parquet")
    pd.DataFrame({"row": np.arange(n_items), "source": "amazon", "item_id": np.arange(n_items)}).to_parquet(root / "embeddings/v2/catalog.parquet")
    for m in ("text", "image"):
        np.save(root / f"embeddings/v2/{m}_pca256.npy", emb[m] / np.linalg.norm(emb[m], axis=1, keepdims=True))
    return root


def test_run_learns_and_artifacts_serve_in_numpy(tmp_path):
    root = toy_root(tmp_path)
    out = tmp_path / "out"
    res = ume.run(root, "amazon", "valid", out=out, epochs=3, device="cpu", ks=(5, 20), log=lambda *a: None)
    m = res["models"]
    # 고객 취향(묶음)이 뚜렷하므로 개인화가 인기보다 훨씬 좋아야 한다
    assert m["ume"]["recall@20"] > m["popularity"]["recall@20"] + 0.1
    assert m["ume"]["brand_alignment@5"] > 0.9
    assert {"text", "image", "ume+pop2@5", "pop19+ume1", "ume+brand1.0"} <= set(m)
    assert json.loads((out / "amazon_valid_pca.json").read_text(encoding="utf-8"))["eval"]["users"] == 300

    # 서빙: 저장된 가중치로 numpy 계산 = torch 계산
    w = dict(np.load(out / "amazon_valid_pca" / "text" / "weights.npz"))
    X = ume.item_features(root, "amazon", 60)["text"]

    def mlp(side, x):
        h = np.maximum(x @ w[f"{side}.0.weight"].T + w[f"{side}.0.bias"], 0)
        v = h @ w[f"{side}.3.weight"].T + w[f"{side}.3.bias"]
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    events = pd.read_parquet(root / "processed/amazon_v2/events.parquet")
    hist = events[events.split == "train"]
    seq = ume.build_sequences(hist, 50)
    users = np.array([0, 1, 2])
    p = {**ume.DEFAULT, "epochs": 3}
    model = {s: ume._mlp(16, p) for s in ("user", "item")}
    for s in ("user", "item"):
        model[s].load_state_dict({k.split(".", 1)[1]: torch.from_numpy(v) for k, v in w.items() if k.startswith(s + ".")})
        model[s].eval()
    U, V = ume.tower_vectors(model, X, seq, users, np.arange(60), 50, "cpu")
    st, en = seq.last_history(users, 50)
    U_np = mlp("user", np.stack([X[seq.items[a:b]].mean(0) for a, b in zip(st, en)]))
    assert np.allclose(U, U_np, atol=1e-5) and np.allclose(V, mlp("item", X), atol=1e-5)


def test_mix_popular():
    rec = np.array([[1, 2, 3, 4]])
    pop = np.array([2, 9, 8, 7])
    assert ume.mix_popular(rec, pop, 2, 4).tolist() == [[1, 2, 9, 8]]      # 앞 2개 rec, 나머지는 rec에 없는 인기


def test_popularity_bias_dominates_when_large():
    h = pd.DataFrame({"user_id": [1, 1, 2, 3, 3, 3], "item_id": [5, 6, 6, 6, 7, 5], "feedback": 1.0,
                      "ts": pd.to_datetime(["2023-01-01"] * 6)})
    catalog = np.array([5, 6, 7, 8])
    z = ume.popularity_score(h, catalog, None)
    assert z.argmax() == 1 and z[3] == z.min() and abs(z.mean()) < 1e-6        # 6이 가장 인기, 8은 0회
    rng = np.random.default_rng(0)
    U = {"text": rng.normal(size=(3, 4)).astype(np.float32), "image": rng.normal(size=(3, 4)).astype(np.float32)}
    V = {"text": rng.normal(size=(4, 4)).astype(np.float32), "image": rng.normal(size=(4, 4)).astype(np.float32)}
    rec = ume.recommend(U, V, catalog, 2, "cpu", item_bias=100 * z)
    assert (rec[:, 0] == 6).all() and (rec[:, 1] == 5).all()


def test_paired_bootstrap():
    from recsys.eval.stats import paired_bootstrap
    rng = np.random.default_rng(0)
    a = rng.random(2000)
    r = paired_bootstrap(a, a + 0.1)
    assert abs(r["diff"] - 0.1) < 1e-9 and r["ci95"][0] > 0.09                  # 확실한 차이 → 0 밖
    r = paired_bootstrap(a, a + rng.normal(0, 0.5, 2000))
    assert r["ci95"][0] < 0 < r["ci95"][1]                                        # 잡음뿐 → 0 포함


def test_preferred_brand_ignores_negative_reviews():
    h = pd.DataFrame({"user_id": [1, 1, 1, 2], "item_id": [0, 1, 2, 2], "feedback": [-1.0, -1.0, 1.0, 0.5]})
    brand_of = np.array([0, 0, 1])                  # 상품 0 · 1 = 브랜드 0 (부정 2건), 상품 2 = 브랜드 1 (긍정 1건)
    assert ume.preferred_brand(h, brand_of, np.array([1, 2, 3])).tolist() == [1, 1, -1]


def test_outputs_separated_and_test_needs_allowlist(tmp_path):
    root = toy_root(tmp_path, n_users=60)
    out = tmp_path / "out"
    kw = dict(out=out, epochs=1, device="cpu", ks=(5, 20), log=lambda *a: None)
    ume.run(root, "amazon", "valid", **kw)
    ume.run(root, "amazon", "valid", feedback_weight=False, **kw)            # 다른 폴더 → 덮어쓰지 않음
    cfg = lambda d: json.loads((out / d / "text" / "config.json").read_text(encoding="utf-8"))["feedback_weight"]  # noqa: E731
    assert cfg("amazon_valid_pca") is True and cfg("amazon_valid_pca_nofb") is False
    with pytest.raises(FileExistsError):
        ume.run(root, "amazon", "valid", **kw)
    with pytest.raises(AssertionError):
        ume.run(root, "amazon", "test", **kw)                                 # test는 허용 목록 필수
    r = ume.run(root, "amazon", "test", gammas=(2.0,), betas=(2.0,), only=["ume+popscore2.0+brand2.0"], **kw)
    assert set(r["models"]) == {"popularity", "ume+popscore2.0+brand2.0"} and set(r["vs_popularity"]) == {"ume+popscore2.0+brand2.0"}
