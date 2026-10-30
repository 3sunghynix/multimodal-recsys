"""검색 결합 규칙 검증 (합성 데이터: 서로 떨어진 두 데이터셋)."""
import numpy as np
import pytest

from recsys import search as S


def two_sources(n_a=800, n_b=200, d=16, seed=0):
    """데이터셋 A · B가 서로 다른 방향에 몰려 있는 상품 임베딩 (실제 진단과 같은 상황)."""
    rng = np.random.default_rng(seed)
    ca, cb = np.zeros(d), np.zeros(d)
    ca[0], cb[1] = 3, 3
    a = S.l2(ca + rng.normal(size=(n_a, d)))
    b = S.l2(cb + rng.normal(size=(n_b, d)))
    return np.vstack([a, b]).astype(np.float32), np.array(["a"] * n_a + ["b"] * n_b)


def test_top_k_and_standardize():
    s = np.array([[0.1, 0.9, 0.5, 0.3]])
    assert S.top_k(s, 2).tolist() == [[1, 2]]
    z = S.standardize(np.array([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]]), np.array(list("aaabbb")))
    assert np.allclose(z[0, :3], z[0, 3:], atol=1e-4)     # 그룹마다 같은 모양 → 같은 표준 점수


def test_per_source_merge_balances_sources():
    emb, g = two_sources()
    q = S.l2(np.array([[1.0, 0.3] + [0] * 14]))          # A 쪽으로 기운 질의
    idx_g, _ = S.search({"text": q}, {"text": emb}, {"text": 1.0}, g, k=20, merge="global")
    idx_z, _ = S.search({"text": q}, {"text": emb}, {"text": 1.0}, g, k=20, merge="per_source_z")
    assert (g[idx_g[0]] == "b").sum() == 0                # 원래 점수: A가 독차지
    assert (g[idx_z[0]] == "b").sum() >= 3                # 데이터셋별 표준화: B도 나옴


def test_weights_follow_priority():
    rng = np.random.default_rng(1)
    t, i = S.l2(rng.normal(size=(50, 8))), S.l2(rng.normal(size=(50, 8)))
    q = {"text": t[[3]], "image": i[[7]]}                 # 텍스트로는 상품 3, 이미지로는 상품 7이 정답
    items = {"text": t, "image": i}
    top = lambda w: S.search(q, items, w, None, k=1, merge="global")[0][0, 0]
    assert top({"text": 1.0}) == 3 and top({"image": 1.0}) == 7
    assert top({"text": 0.9, "image": 0.1}) == 3 and top({"text": 0.1, "image": 0.9}) == 7
    with pytest.raises(AssertionError):
        S.combine({"text": t @ t.T}, {"text": 0.0}, None)


def test_project_matches_pca_transform():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 12))
    mean, comp = rng.normal(size=12).astype(np.float32), np.linalg.qr(rng.normal(size=(12, 4)))[0].T
    v = S.project(x, mean, comp)
    ref = (S.l2(x) - mean) @ comp.T
    assert np.allclose(v, ref / np.linalg.norm(ref, axis=1, keepdims=True), atol=1e-5)
    assert np.allclose(np.linalg.norm(S.project(x, mean, comp, center=False), axis=1), 1, atol=1e-5)


def test_per_source_rank_follows_catalog_share():
    emb, g = two_sources()                                # A 80% · B 20%
    q = S.l2(np.array([[1.0, 0.3] + [0] * 14]))
    idx, _ = S.search({"text": q}, {"text": emb}, {"text": 1.0}, g, k=20, merge="per_source_rank")
    assert (g[idx[0]] == "b").sum() == 4                  # 20 × 20% = 4


def test_translate_ko():
    from recsys.translate import translate_ko
    assert translate_ko("검은색 원피스")[0] == "black dress"
    assert translate_ko("청자켓")[0] == "denim jacket"
    assert translate_ko("하이웨이스트청바지")[0] == "high waist jeans"         # 붙여 쓴 복합어
    assert translate_ko("여름에 입을 원피스", mt=lambda ws: ["wear"] * len(ws)) == ("summer wear dress", ["입을"])
    assert translate_ko("앙고라 자켓") == ("앙고라 jacket", ["앙고라"])         # mt 없으면 원문 유지
    assert translate_ko("Nike 운동화")[0] == "Nike sneakers"


def test_backend_copies_match():
    """백엔드(Docker 빌드 범위가 backend/)는 같은 코드를 복사해 쓴다. 실험과 서비스가 달라지지 않게 확인."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for a, b in (("recsys/search.py", "backend/app/services/search_core.py"),
                 ("recsys/translate.py", "backend/app/services/translate_ko.py")):
        assert (root / a).read_bytes() == (root / b).read_bytes(), f"{b}를 {a}와 같게 복사하세요"


def test_min_rel_keeps_irrelevant_source_out():
    rng = np.random.default_rng(3)
    d = 16
    q = S.l2(np.eye(d)[[0]])
    good_a = S.l2(np.eye(d)[0] + rng.normal(scale=0.1, size=(40, d)))       # A: 질의와 맞는 상품
    b_far = S.l2(np.eye(d)[5] + rng.normal(scale=0.1, size=(10, d)))        # B: 전부 다른 종류 (러닝화 검색의 양말)
    b_mixed = np.vstack([b_far[:8], S.l2(np.eye(d)[0] + rng.normal(scale=0.1, size=(2, d)))])   # B에 맞는 상품 2개
    g = np.array(["a"] * 40 + ["b"] * 10)
    for items, expect_b in ((np.vstack([good_a, b_far]), 0), (np.vstack([good_a, b_mixed]), 2)):
        idx, _ = S.search({"text": q}, {"text": items}, {"text": 1.0}, g, k=10, merge="per_source_rank", min_rel=0.7)
        assert (g[idx[0]] == "b").sum() == expect_b
        idx0, _ = S.search({"text": q}, {"text": items}, {"text": 1.0}, g, k=10, merge="per_source_rank")
        assert (g[idx0[0]] == "b").sum() == 2                              # 기준 없이는 B 칸(10 × 20%)을 항상 채움
