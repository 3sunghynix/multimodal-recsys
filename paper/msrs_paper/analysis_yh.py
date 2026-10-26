"""3차 사전 등록(configs/prereg_yh_fusion.yaml, 김유환) 분석: 텍스트 드롭아웃 · 결합 구조.

확증 (본페로니 2개, 97.5%):
  H3: R_all(텍스트 드롭아웃 p*) − R_all(C2)    p*는 검증 주 튜닝 1등 (튜닝 결과에서 자동 선택)
  H4: R_all(gate) − R_all(C2)
탐색: 조건 × 구간별 R · 내용 기여도, 기준(이어 붙이기) 대비 차이, 게이트 α 분포.
채점 · 부트스트랩은 analysis_v2.build_context를 그대로 쓴다 (같은 구간 정의, 같은 추출, seed별 계산 후 평균).

결과물 (드라이브 results_yh/):
  tuning_td.csv, confirmatory.json, table_yh.csv, gate_alpha_summary.csv, gate_alpha_corr.json, fig_yh.png
"""
import numpy as np
import pandas as pd

from . import analysis_v2
from .config import load_yaml
from .data_prep import load_items
from .utils import now, read_json, write_json

TD_CELLS = {0.1: ("td10", "YH_TD1R"), 0.3: ("td30", "YH_TD3R"), 0.5: ("td50", "YH_TD5R")}
BASE = "base_concat"
SEGMENTS = ["all", "established", "sparse", "new_item"]


def results_dir(paths):
    d = paths.root / "results_yh"
    (d / "cache").mkdir(parents=True, exist_ok=True)
    return d


def select_text_dropout(paths):
    """검증 주 튜닝 결과에서 p* 선택 (사전 등록 규칙: 텍스트 + Marqo 진짜, seed 42, 검증 Recall@300 1등)."""
    rows = []
    for p, (cell, cond) in TD_CELLS.items():
        f = paths.outputs / "tuning" / "main" / f"{cond}_seed42_h2" / "run_info.json"
        if not f.exists():
            raise FileNotFoundError(f"튜닝 결과 없음: {f} — 08 노트북 튜닝 셀을 먼저 실행하세요")
        rows.append({"text_dropout": p, "cell": cell, "valid_recall@300": read_json(f)["valid_recall@300"]})
    ref = paths.outputs / "tuning" / "main" / "C2_seed42_h2" / "run_info.json"
    if ref.exists():
        rows.append({"text_dropout": 0.0, "cell": BASE, "valid_recall@300": read_json(ref)["valid_recall@300"],
                     "note": "참고값 (선택 후보 아님)"})
    table = pd.DataFrame(rows).sort_values("valid_recall@300", ascending=False)
    best = table[table["cell"] != BASE].iloc[0]
    return float(best["text_dropout"]), best["cell"], table


def visual_distinctiveness(paths, items):
    """같은 product_type_name 평균 이미지 임베딩(Marqo)과의 코사인 거리."""
    emb = np.load(paths.embeddings / "img_marqo.npy")
    df = pd.DataFrame({"t": items["product_type_name"].fillna("NA").values})
    out = np.zeros(len(items), dtype=np.float32)
    for _, idx in df.groupby("t").indices.items():
        c = emb[idx].mean(0)
        c = c / (np.linalg.norm(c) + 1e-12)
        out[idx] = 1 - emb[idx] @ c
    return out


def gate_analysis(ctx, paths, out, log=print):
    runs = ctx["cells"]["gate"].get("real", [])
    alphas = [np.load(d / "gate_alpha.npy") for _, _, _, d in runs if (d / "gate_alpha.npy").exists()]
    if not alphas:
        log("게이트 α 파일 없음 — 건너뜀")
        return None, None
    alpha = np.mean(alphas, axis=0)
    items = load_items(paths).sort_values("idx").reset_index(drop=True)
    sales = analysis_v2.item_sales_counts(paths, len(items))
    bins = [-1, 0, 2, 9, 49, np.inf]
    labels = ["0", "1-2", "3-9", "10-49", "50+"]
    df = pd.DataFrame({"alpha": alpha, "sales_bucket": pd.cut(sales, bins=bins, labels=labels).astype(str),
                       "pgroup": items["pgroup"].fillna("excluded").values,
                       "appearance": items["appearance"].fillna("excluded").values,
                       "distinct": visual_distinctiveness(paths, items)})
    df["distinct_tercile"] = pd.qcut(df["distinct"], 3, labels=["low", "mid", "high"]).astype(str)
    rows = []
    for key in ["sales_bucket", "pgroup", "appearance", "distinct_tercile"]:
        g = df.groupby(key)["alpha"].agg(["mean", "std", "count"]).reset_index().rename(columns={key: "value"})
        g.insert(0, "by", key)
        rows.append(g)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(out / "gate_alpha_summary.csv", index=False)
    corr = {"n_seeds": len(alphas), "alpha_mean": float(alpha.mean()), "alpha_std": float(alpha.std()),
            "spearman_alpha_log_sales": float(pd.Series(alpha).corr(pd.Series(np.log1p(sales)), method="spearman")),
            "spearman_alpha_distinct": float(df["alpha"].corr(df["distinct"], method="spearman")),
            "status": "탐색적 (사전 등록의 exploratory 항목)"}
    write_json(out / "gate_alpha_corr.json", corr)
    log(f"게이트 α 평균 {corr['alpha_mean']:.3f}, 판매 횟수와 순위상관 {corr['spearman_alpha_log_sales']:+.3f}, "
        f"시각 특이성과 {corr['spearman_alpha_distinct']:+.3f}")
    return summary, corr


def figure(table, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table[table["segment"] == "all"].set_index("cell")
    order = [c for c in [BASE, "td10", "td30", "td50", "vd10", "vd30", "vd50", "late", "gate"] if c in t.index]
    t = t.loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].bar(order, t["R_real"], yerr=[t["R_real"] - t["R_real_ci_low"], t["R_real_ci_high"] - t["R_real"]], color="#4C72B0")
    ax[0].axhline(t.loc[BASE, "R_real"], color="gray", lw=0.8, ls="--")
    ax[0].set_ylim(t["R_real_ci_low"].min() * 0.97, t["R_real_ci_high"].max() * 1.02)
    ax[0].set_title("R (Recall@300, all)")
    ax[1].bar(order, t["content_relative"], yerr=[t["content_relative"] - t["content_relative_ci_low"],
                                                  t["content_relative_ci_high"] - t["content_relative"]], color="#DD8452")
    ax[1].set_title("relative content contribution (all)")
    for a in ax:
        a.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(cfg, exps, paths, log=print, ctx=None):
    prereg = load_yaml("prereg_yh_fusion")
    assert prereg["status"] == "final"
    out = results_dir(paths)
    p_star, td_cell, tuning = select_text_dropout(paths)
    tuning.to_csv(out / "tuning_td.csv", index=False)
    log(f"텍스트 드롭아웃 p* = {p_star} ({td_cell})")

    ctx = ctx or analysis_v2.build_context(cfg, exps, paths, log, cells_key="yh_cells", out=out)
    level, contribution, cells = ctx["level"], ctx["contribution"], ctx["cells"]
    expected = prereg["seeds"]
    conf = prereg["confirmatory"]

    def test(cell, lvl):
        pa, ba, sa = level(cell, "real", "all")
        pb, bb, sb = level(BASE, "real", "all")
        lo, hi = analysis_v2._q(ba - bb, lvl)
        complete = sa == expected and sb == expected
        return {"estimate": pa - pb, "ci_low": lo, "ci_high": hi, "ci_level": lvl,
                "a": {"cell": cell, "R": pa, "seeds": sa}, "b": {"cell": BASE, "R": pb, "seeds": sb},
                "status": "ok" if complete else "seeds_incomplete", "supported": bool(complete and lo > 0)}

    confirmatory = {
        "H3": {"name": conf["H3"]["name"], "comparison": conf["H3"]["comparison"], "text_dropout": p_star,
               **test(td_cell, conf["ci_level"])},
        "H4": {"name": conf["H4"]["name"], "comparison": conf["H4"]["comparison"], **test("gate", conf["ci_level"])},
        "correction": conf["correction"], "created_at": now(),
    }
    write_json(out / "confirmatory.json", confirmatory)
    for h in ("H3", "H4"):
        c = confirmatory[h]
        log(f"{h} {c['name']}: {c['estimate']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
            f"({c['ci_level']:.1%}) 지지={c['supported']} 상태={c['status']}")

    rows = []
    for cell in cells:
        for seg in SEGMENTS:
            pr, br, seeds = level(cell, "real", seg)
            if not seeds:
                continue
            pg, bg, _ = level(cell, "shuffle_global", seg)
            row = {"cell": cell, "segment": seg, "seeds": ",".join(map(str, seeds)), "R_real": pr, "R_global": pg}
            row["R_real_ci_low"], row["R_real_ci_high"] = analysis_v2._q(br, 0.95)
            p0, b0, _ = level(BASE, "real", seg)
            row["R_minus_base"] = pr - p0
            row["R_minus_base_ci_low"], row["R_minus_base_ci_high"] = analysis_v2._q(br - b0, 0.95)
            for kind in ("relative", "absolute"):
                pc, bc, _ = contribution(cell, seg, "content", kind)
                pc0, bc0, _ = contribution(BASE, seg, "content", kind)
                row[f"content_{kind}"] = pc
                row[f"content_{kind}_ci_low"], row[f"content_{kind}_ci_high"] = analysis_v2._q(bc, 0.95)
                row[f"content_{kind}_minus_base"] = pc - pc0
                lo, hi = analysis_v2._q(bc - bc0, 0.95)
                row[f"content_{kind}_minus_base_ci_low"], row[f"content_{kind}_minus_base_ci_high"] = lo, hi
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "table_yh.csv", index=False)
    figure(table, out / "fig_yh.png")
    gate_analysis(ctx, paths, out, log)
    return confirmatory, table, tuning
