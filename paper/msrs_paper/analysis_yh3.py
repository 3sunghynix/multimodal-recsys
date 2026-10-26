"""5차 사전 등록(configs/prereg_yh_where.yaml, 김유환) 분석 (s2 분할).

확증 (본페로니 3개, 98.33%):
  H7: 상대 내용 기여도 LAION-2B CLIP B/32 − OpenAI CLIP B/32 (구간 all)
  H8: Marqo 절대 내용 기여도 sparse − established
  H9: R_all(B2 200 + 텍스트 + Marqo 모델 C2) − R_all(B2 200 + 텍스트만 모델 C0)
탐색: 인코더 6종 × 구간 표(텍스트만 · 같은 벡터 대조군 대비 R 포함), SigLIP − ResNet, new − established(이미 일부 봄),
      결합 n 스윕 · 구간별 · 이미지 드롭아웃 모델, B1 · B2 단독.
인코더 기여도는 analysis_v2.build_context(split="s2")를, 결합 후보는 같은 부트스트랩 추출(metrics.bootstrap_weights, 같은 seed)을 쓴다.

결과물 (드라이브 results_yh3/): confirmatory.json, encoders_yh3.csv, hybrid_yh3.csv, contrasts_yh3.csv
"""
import numpy as np
import pandas as pd

from . import analysis_v2, metrics
from .config import load_yaml
from .data_prep import load_split
from .utils import now, write_json

SPLIT = "s2"
SEGMENTS = ["all", "established", "sparse", "new_item"]
ENCODER_CELLS = ["noid_resnet50", "noid_clipb32", "noid_laion_b32", "noid_siglip_b16", "noid_marqo"]
HYBRID_MODELS = {"text_only": ("YH3_ENC", "C0"), "text_marqo": ("YH2_BASE", "C2"), "text_marqo_imgdrop30": ("YH2_VD", "YH_VD3R")}


def results_dir(paths):
    d = paths.root / "results_yh3"
    (d / "cache").mkdir(parents=True, exist_ok=True)
    return d


def combine(base, model, n_base, k):
    """고객마다 base 앞 n_base개 + model 후보를 순서대로 중복 없이 채워 k개."""
    if n_base >= k:
        return base[:, :k]
    if n_base == 0:
        return model[:, :k]
    head = base[:, :n_base].astype(np.int64)
    n = int(max(base.max(), model.max())) + 1
    row = np.arange(len(base), dtype=np.int64)[:, None]
    dup = np.isin(row * n + model.astype(np.int64), (row * n + head).ravel())
    order = np.argsort(dup, axis=1, kind="stable")        # 중복 아닌 것 먼저, 원래 순서 유지
    tail = np.take_along_axis(model, order, axis=1)[:, :k - n_base]
    return np.concatenate([base[:, :n_base], tail], axis=1)


def series_stats(named_hits, rows, masks, U, n_boot, seed, segs):
    """(이름, 구매별 적중) → {(이름, 구간): (R 점추정, 부트스트랩 벡터)}. build_context와 같은 계산 · 같은 추출."""
    A, B, keys = [], [], []
    for name, hit in named_hits:
        for seg in segs:
            m = masks[seg]
            n_u = np.bincount(rows[m], minlength=U)
            h_u = np.bincount(rows[m], weights=hit[m], minlength=U)
            has = n_u > 0
            A.append((np.divide(h_u, n_u, out=np.zeros(U), where=has) * has).astype(np.float32))
            B.append(has.astype(np.float32))
            keys.append((name, seg))
    A, B = np.stack(A), np.stack(B)
    point = A.sum(1, dtype=np.float64) / B.sum(1, dtype=np.float64)
    boot = np.empty((len(A), n_boot))
    col, W = 0, []
    for w in metrics.bootstrap_weights(U, n_boot, seed):
        W.append(w.astype(np.float32))
        if len(W) == 100:
            Wm = np.stack(W, axis=1)
            boot[:, col:col + 100] = (A @ Wm) / (B @ Wm)
            col += 100
            W = []
    if W:
        Wm = np.stack(W, axis=1)
        boot[:, col:col + len(W)] = (A @ Wm) / (B @ Wm)
    return {key: (point[i], boot[i]) for i, key in enumerate(keys)}


def hybrid_analysis(cfg, paths, prereg, masks, log=print):
    k = cfg["eval"]["k"]
    _, _, targets = load_split(paths, SPLIT)
    b2_dir = paths.run_dir("YH3_BASE", SPLIT, "B2_seed0")
    b1_dir = paths.run_dir("YH3_BASE", SPLIT, "B1_seed0")
    users = np.load(b2_dir / "users.npy")
    rows = metrics.target_rows(users, targets)
    items = targets["item_idx"].values
    b2 = np.load(b2_dir / "candidates.npy")
    hits = [("B2", metrics.hits_at_k(b2, rows, items, k)),
            ("B1", metrics.hits_at_k(np.load(b1_dir / "candidates.npy"), rows, items, k))]
    sweep = prereg["hybrid"]["sweep_exploratory"]
    seeds = prereg["seeds"]
    for label, (group, cond) in HYBRID_MODELS.items():
        for s in seeds:
            d = paths.run_dir(group, SPLIT, f"{cond}_seed{s}")
            if not (d / "run_info.json").exists():
                log(f"[결합] 실행 없음, 건너뜀: {d}")
                continue
            if not np.array_equal(np.load(d / "users.npy"), users):
                raise ValueError(f"{d}: 테스트 고객 목록이 B2와 다릅니다")
            model = np.load(d / "candidates.npy")
            for n in sweep:
                if n >= k:
                    continue
                hits.append((f"{label}|{s}|{n}", metrics.hits_at_k(combine(b2, model, n, k), rows, items, k)))
        log(f"[결합] {label} 완료")
    n_boot = prereg["measure"]["bootstrap"]["n"]
    st = series_stats(hits, rows, masks, len(users), n_boot, prereg["measure"]["bootstrap"]["seed"], SEGMENTS)

    def level(label, n, seg):
        if n >= k or label in ("B1", "B2"):
            return st[("B2" if label != "B1" else "B1", seg)]
        got = [st[(f"{label}|{s}|{n}", seg)] for s in seeds if (f"{label}|{s}|{n}", seg) in st]
        return float(np.mean([g[0] for g in got])), np.mean([g[1] for g in got], axis=0)

    rows_out = []
    for label in HYBRID_MODELS:
        for n in sweep:
            for seg in SEGMENTS:
                p, b = level(label, n, seg)
                pb, bb = level("B2", k, seg)
                lo, hi = analysis_v2._q(b - bb, 0.95)
                rows_out.append({"model": label, "n_baseline": n, "segment": seg, "R": p, "R_minus_B2": p - pb,
                                 "R_minus_B2_ci_low": lo, "R_minus_B2_ci_high": hi})
    for seg in SEGMENTS:
        for name in ("B1", "B2"):
            p, _ = st[(name, seg)]
            rows_out.append({"model": name, "n_baseline": k, "segment": seg, "R": p})
    return level, pd.DataFrame(rows_out)


def run(cfg, exps, paths, log=print, ctx=None):
    prereg = load_yaml("prereg_yh_where")
    assert prereg["status"] == "final"
    out = results_dir(paths)
    conf = prereg["confirmatory"]
    lvl = conf["ci_level"]
    expected = prereg["seeds"]

    ctx = ctx or analysis_v2.build_context(cfg, exps, paths, log, cells_key="yh3_cells", out=out, split=SPLIT)
    contribution, level = ctx["contribution"], ctx["level"]

    def contrast(a, b, lv):
        (pa, ba, sa), (pb, bb, sb) = a, b
        lo, hi = analysis_v2._q(ba - bb, lv)
        complete = sa == expected and sb == expected
        return {"a": pa, "b": pb, "estimate": pa - pb, "ci_low": lo, "ci_high": hi, "ci_level": lv,
                "complete": complete, "supported": bool(complete and lo > 0)}

    confirmatory = {"correction": conf["correction"], "split": SPLIT, "created_at": now()}
    confirmatory["H7"] = {"name": conf["H7"]["name"], "comparison": conf["H7"]["comparison"],
                          **contrast(contribution("noid_laion_b32", "all", "content", "relative"),
                                     contribution("noid_clipb32", "all", "content", "relative"), lvl)}
    confirmatory["H8"] = {"name": conf["H8"]["name"], "comparison": conf["H8"]["comparison"],
                          **contrast(contribution("noid_marqo", "sparse", "content", "absolute"),
                                     contribution("noid_marqo", "established", "content", "absolute"), lvl)}

    h_level, hybrid = hybrid_analysis(cfg, paths, prereg, ctx["masks"], log)
    n0 = prereg["hybrid"]["n_baseline"]
    pa, ba = h_level("text_marqo", n0, "all")
    pb, bb = h_level("text_only", n0, "all")
    lo, hi = analysis_v2._q(ba - bb, lvl)
    confirmatory["H9"] = {"name": conf["H9"]["name"], "comparison": conf["H9"]["comparison"],
                          "a": pa, "b": pb, "estimate": pa - pb, "ci_low": lo, "ci_high": hi, "ci_level": lvl,
                          "supported": bool(lo > 0), "R_B2_alone": float(h_level("B2", 10 ** 9, "all")[0])}
    write_json(out / "confirmatory.json", confirmatory)
    for h in ("H7", "H8", "H9"):
        c = confirmatory[h]
        log(f"{h} {c['name']}: {c['a']:.4f} − {c['b']:.4f} = {c['estimate']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
            f"({lvl:.2%}) 지지={c['supported']}")

    # 인코더 표 (탐색)
    rows = []
    for cell in ENCODER_CELLS:
        for seg in SEGMENTS:
            pr, br, seeds = level(cell, "real", seg)
            if not seeds:
                continue
            row = {"cell": cell, "segment": seg, "R_real": pr}
            for ref in ("text_only", "const_ctrl"):
                p0, b0, s0 = level(ref, "real", seg)
                if s0:
                    row[f"R_minus_{ref}"] = pr - p0
                    row[f"R_minus_{ref}_ci_low"], row[f"R_minus_{ref}_ci_high"] = analysis_v2._q(br - b0, 0.95)
            for kind in ("relative", "absolute"):
                pc, bc, _ = contribution(cell, seg, "content", kind)
                row[f"content_{kind}"] = pc
                row[f"content_{kind}_ci_low"], row[f"content_{kind}_ci_high"] = analysis_v2._q(bc, 0.95)
            rows.append(row)
    encoders = pd.DataFrame(rows)
    encoders.to_csv(out / "encoders_yh3.csv", index=False)
    hybrid.to_csv(out / "hybrid_yh3.csv", index=False)

    con = []
    for desc, a, b in [
        ("SigLIP B/16 − ResNet-50 (상대, all)", contribution("noid_siglip_b16", "all", "content", "relative"),
         contribution("noid_resnet50", "all", "content", "relative")),
        ("Marqo new_item − established (절대, 이미 일부 봄)", contribution("noid_marqo", "new_item", "content", "absolute"),
         contribution("noid_marqo", "established", "content", "absolute")),
        ("이미지 드롭아웃 0.3 − Marqo (R, all)", level("imgdrop30", "real", "all"), level("noid_marqo", "real", "all")),
    ]:
        con.append({"description": desc, **contrast(a, b, 0.95), "status": "탐색적"})
    for seg in SEGMENTS:
        p1, b1 = h_level("text_marqo", n0, seg)
        p2, b2_ = h_level("B2", 10 ** 9, seg)
        lo, hi = analysis_v2._q(b1 - b2_, 0.95)
        con.append({"description": f"결합(B2 {n0} + 텍스트 + Marqo) − B2 단독 ({seg})", "a": p1, "b": p2,
                    "estimate": p1 - p2, "ci_low": lo, "ci_high": hi, "ci_level": 0.95, "status": "탐색적"})
        p3, b3 = h_level("text_only", n0, seg)
        lo, hi = analysis_v2._q(b1 - b3, 0.95)
        con.append({"description": f"결합에서 이미지 효과: 텍스트 + Marqo − 텍스트만 ({seg})", "a": p1, "b": p3,
                    "estimate": p1 - p3, "ci_low": lo, "ci_high": hi, "ci_level": 0.95, "status": "탐색적"})
    contrasts = pd.DataFrame(con)
    contrasts.to_csv(out / "contrasts_yh3.csv", index=False)
    for r in con:
        log(f"[탐색] {r['description']}: {r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    return confirmatory, encoders, hybrid, contrasts
