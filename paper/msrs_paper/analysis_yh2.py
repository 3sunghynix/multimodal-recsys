"""4차 사전 등록(configs/prereg_yh_imgdrop.yaml, 김유환) 분석: 이미지 드롭아웃 재검증과 원인 분리 (s2 분할).

확증 (본페로니 2개, 97.5%):
  H5: R_all(이미지 드롭아웃 p*) − R_all(C2)            p*: s2 검증 주 튜닝 1등
  H6: R_all(배율 보정 이미지 드롭아웃 p*_rs) − R_all(C2)  p*_rs: 같은 규칙으로 따로 선택
탐색: 원소 단위 대 블록, 텍스트 + 이미지 드롭아웃, 추천 때 배율, H3 재현, 대조군 종류(0-2), 진짜 · 섞은 이미지 분해.
채점 · 부트스트랩은 analysis_v2.build_context(split="s2")를 그대로 쓴다.

결과물 (드라이브 results_yh2/): tuning.csv, confirmatory.json, table_yh2.csv, contrasts_yh2.csv
"""
import pandas as pd

from . import analysis_v2
from .config import load_yaml
from .utils import now, read_json, write_json

SPLIT = "s2"
P_TAG = {0.1: "10", 0.3: "30", 0.5: "50"}
TUNE = {"vd": {0.1: "YH_VD1R", 0.3: "YH_VD3R", 0.5: "YH_VD5R"},
        "vs": {0.1: "YH_VS1R", 0.3: "YH_VS3R", 0.5: "YH_VS5R"}}
SEGMENTS = ["all", "established", "sparse", "new_item"]


def results_dir(paths):
    d = paths.root / "results_yh2"
    (d / "cache").mkdir(parents=True, exist_ok=True)
    return d


def select(paths):
    """s2 검증 주 튜닝 결과에서 vd · vs 각각 p 선택."""
    rows, best = [], {}
    for kind, conds in TUNE.items():
        for p, cond in conds.items():
            f = paths.outputs / "tuning" / SPLIT / f"{cond}_seed42_h2" / "run_info.json"
            if not f.exists():
                raise FileNotFoundError(f"튜닝 결과 없음: {f} — 09 노트북 튜닝 셀을 먼저 실행하세요")
            rows.append({"kind": kind, "p": p, "cond": cond, "valid_recall@300": read_json(f)["valid_recall@300"]})
    ref = paths.outputs / "tuning" / SPLIT / "C2_seed42_h2" / "run_info.json"
    if ref.exists():
        rows.append({"kind": "base", "p": 0.0, "cond": "C2", "valid_recall@300": read_json(ref)["valid_recall@300"]})
    table = pd.DataFrame(rows).sort_values(["kind", "valid_recall@300"], ascending=[True, False])
    for kind in TUNE:
        t = table[table["kind"] == kind]
        best[kind] = float(t.iloc[0]["p"])
    return best, table


def run(cfg, exps, paths, log=print, ctx=None):
    prereg = load_yaml("prereg_yh_imgdrop")
    assert prereg["status"] == "final"
    out = results_dir(paths)
    best, tuning = select(paths)
    tuning.to_csv(out / "tuning.csv", index=False)
    vd_cell, vs_cell = f"vd{P_TAG[best['vd']]}", f"vs{P_TAG[best['vs']]}"
    log(f"이미지 드롭아웃 p* = {best['vd']} ({vd_cell}), 배율 보정 p*_rs = {best['vs']} ({vs_cell})")

    ctx = ctx or analysis_v2.build_context(cfg, exps, paths, log, cells_key="yh2_cells", out=out, split=SPLIT)
    level, contribution, cells = ctx["level"], ctx["contribution"], ctx["cells"]
    expected = prereg["seeds"]
    conf = prereg["confirmatory"]

    def diff(a, b, seg="all", lvl=0.95, role_a="real", role_b="real"):
        pa, ba, sa = level(a, role_a, seg)
        pb, bb, sb = level(b, role_b, seg)
        lo, hi = analysis_v2._q(ba - bb, lvl)
        return {"a": a, "b": b, "segment": seg, "R_a": pa, "R_b": pb, "estimate": pa - pb, "ci_low": lo, "ci_high": hi,
                "ci_level": lvl, "seeds_a": sa, "seeds_b": sb, "complete": sa == expected and sb == expected}

    confirmatory = {"correction": conf["correction"], "created_at": now(), "split": SPLIT,
                    "selected": {"image_dropout": best["vd"], "image_dropout_rescaled": best["vs"]}}
    for h, cell in (("H5", vd_cell), ("H6", vs_cell)):
        d = diff(cell, "base", lvl=conf["ci_level"])
        d["supported"] = bool(d["complete"] and d["ci_low"] > 0)
        confirmatory[h] = {"name": conf[h]["name"], "comparison": conf[h]["comparison"], **d}
        log(f"{h} {conf[h]['name']}: {d['estimate']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] "
            f"({conf['ci_level']:.1%}) 지지={d['supported']} seed 완전={d['complete']}")
    write_json(out / "confirmatory.json", confirmatory)

    # 조건 × 구간 표 (R, 기준 대비, 내용 기여도, 진짜 · 섞은 이미지 분해)
    rows = []
    for cell in cells:
        for seg in SEGMENTS:
            pr, br, seeds = level(cell, "real", seg)
            if not seeds:
                continue
            p0, b0, _ = level("base", "real", seg)
            row = {"cell": cell, "segment": seg, "seeds": ",".join(map(str, seeds)), "R_real": pr, "R_minus_base": pr - p0}
            row["R_minus_base_ci_low"], row["R_minus_base_ci_high"] = analysis_v2._q(br - b0, 0.95)
            if cells[cell].get("shuffle_global"):
                pg, bg, _ = level(cell, "shuffle_global", seg)
                g0, gb0, _ = level("base", "shuffle_global", seg)
                row["R_global"], row["R_global_minus_base"] = pg, pg - g0
                for kind in ("relative", "absolute"):
                    pc, bc, _ = contribution(cell, seg, "content", kind)
                    row[f"content_{kind}"] = pc
                    row[f"content_{kind}_ci_low"], row[f"content_{kind}_ci_high"] = analysis_v2._q(bc, 0.95)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "table_yh2.csv", index=False)

    # 탐색 대비 (95%, 보정 없음)
    tag = P_TAG[best["vd"]]
    plan = [
        ("블록 대 원소 단위 (p*)", f"vd{tag}", f"ve{tag}"),
        ("텍스트 0.1 + 이미지 p* − 이미지 p*", f"tv{tag}", f"vd{tag}"),
        ("배율 보정 p*_rs − 보정 없음 p*", vs_cell, vd_cell),
        ("추천 때 이미지 2배 − 기준", "ts2", "base"),
        ("H3 재현: 텍스트 드롭아웃 0.1 − 기준", "td10", "base"),
        ("0-2 섞은 이미지: td 0.5 − td 0", "ctrl_shuf5", "ctrl_shuf0"),
        ("0-2 무작위 벡터: td 0.5 − td 0", "ctrl_rand5", "ctrl_rand0"),
        ("0-2 같은 벡터: td 0.5 − td 0", "ctrl_const5", "ctrl_const0"),
        ("0-2 무작위 벡터 − 섞은 이미지 (td 0)", "ctrl_rand0", "ctrl_shuf0"),
    ]
    con = []
    for desc, a, b in plan:
        for seg in ("all", "established", "new_item"):
            d = diff(a, b, seg)
            con.append({"description": desc, **{k: v for k, v in d.items() if k not in ("seeds_a", "seeds_b")},
                        "excludes_zero": bool(d["ci_low"] > 0 or d["ci_high"] < 0), "status": "탐색적"})
    contrasts = pd.DataFrame(con)
    contrasts.to_csv(out / "contrasts_yh2.csv", index=False)
    for r in con:
        if r["segment"] == "all":
            log(f"[탐색] {r['description']}: {r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    return confirmatory, table, contrasts, tuning
