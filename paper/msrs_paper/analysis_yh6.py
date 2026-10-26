"""8차 사전 등록(configs/prereg_yh_ume.yaml, 김유환) 분석: 따로 학습 후 점수 합산(UME) 확증 — s2 확증, main 재현.

확증 (s2, 본페로니 3개, 98.33%):
  H15: R(UME: C0 + C1) − R(C2 두 개 앙상블)
  H16: R(UME: C0 + C1) − R(C1 두 개 앙상블)
  H17: R(B2 200 + UME) − R(B2 200 + C2)
앙상블: 모델별 고객 × 전체 상품 점수를 고객별 z 표준화해 가중합한 상위 k개. 세 짝(seed 또는 seed 쌍)의 평균.
모든 계열을 같은 부트스트랩 추출로 계산한다 (analysis_yh3.series_stats).

결과물 (드라이브 results_yh6/): confirmatory.json, table_yh6.csv, contrasts_yh6.csv, determinism_yh6.csv
"""
import numpy as np
import pandas as pd

from . import analysis_v2, metrics
from .analysis_v2 import item_sales_counts, segment_masks
from .analysis_yh3 import combine, series_stats
from .config import load_yaml
from .data_prep import load_split
from .utils import now, write_json

SEGMENTS = ["all", "established", "sparse", "new_item"]
GROUP = {"s2": "YH6_S2", "main": "YH6_MAIN"}
PAIRS = [(42, 43), (43, 44), (44, 42)]
OLD_RUNS = {  # 결정성 확인용 기존 실행 (분할, 조건) → (묶음, seed 목록)
    ("s2", "C0"): ("YH3_ENC", (42, 43, 44)), ("s2", "C1"): ("YH4", (42, 43, 44)), ("s2", "C2"): ("YH2_BASE", (42, 43, 44)),
    ("s2", "C5"): ("YH4", (42, 43, 44)), ("s2", "YH4_SI"): ("YH4", (42, 43, 44)),
    ("main", "C0"): ("A", (42, 43, 44)), ("main", "C1"): ("A", (42,)), ("main", "C2"): ("A", (42, 43, 44)),
}


def results_dir(paths):
    d = paths.root / "results_yh6"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensemble_candidates(dirs, weights, k, device=None, chunk=2048):
    """여러 모델의 점수를 고객별 z 표준화해 가중합한 상위 k개. dirs: 벡터가 저장된 실행 폴더 목록."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    load = lambda d, n: torch.from_numpy(np.load(d / n).astype(np.float32)).to(device)
    UV = [(load(d, "user_vectors.npy"), load(d, "item_vectors.npy")) for d in dirs]
    z = lambda x: (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)
    out = []
    with torch.no_grad():
        for s in range(0, len(UV[0][0]), chunk):
            score = sum(w * z(U[s:s + chunk] @ V.T) for (U, V), w in zip(UV, weights))
            out.append(score.topk(k, dim=1).indices.int().cpu().numpy())
    return np.concatenate(out)


def analyze_split(cfg, paths, prereg, split, log):
    k = cfg["eval"]["k"]
    seeds = prereg["seeds"]
    g = GROUP[split]
    run = lambda c, s: paths.run_dir(g, split, f"{c}_seed{s}")
    _, _, targets = load_split(paths, split)
    users = np.load(run("C0", 42) / "users.npy")
    rows = metrics.target_rows(users, targets)
    items = targets["item_idx"].values
    n_items = len(pd.read_parquet(paths.processed / "items.parquet", columns=["idx"]))
    masks = segment_masks(targets, item_sales_counts(paths, n_items, split), load_yaml("prereg_v2"))
    b2 = np.load(paths.outputs / prereg["hybrid"]["b2_runs"][split].split("outputs/")[1] / "candidates.npy")
    b2_users = np.load(paths.outputs / prereg["hybrid"]["b2_runs"][split].split("outputs/")[1] / "users.npy")
    assert np.array_equal(b2_users, users), "B2와 모델의 테스트 고객 목록이 다릅니다"

    # 결정성 확인
    det = []
    for (sp, c), (og, oseeds) in OLD_RUNS.items():
        if sp != split:
            continue
        for s in oseeds:
            old = paths.run_dir(og, split, f"{c}_seed{s}") / "candidates.npy"
            new = run(c, s) / "candidates.npy"
            if old.exists() and new.exists():
                det.append({"split": split, "cond": c, "seed": s, "identical": bool(np.array_equal(np.load(old), np.load(new)))})

    # 계열: 이름 → 세 짝의 후보
    series = {}
    single = ["C0", "C1", "C2"] + (["YH4_SI", "C5"] if split == "s2" else [])
    for c in single:
        series[c] = [np.load(run(c, s) / "candidates.npy") for s in seeds]
    E = lambda conds_per_rep, w=None: [ensemble_candidates([run(c, s) for c, s in rep], w or [1.0] * len(rep), k)
                                       for rep in conds_per_rep]
    series["UME"] = E([[("C0", s), ("C1", s)] for s in seeds])
    series["C2+C2"] = E([[("C2", a), ("C2", b)] for a, b in PAIRS])
    series["C1+C1"] = E([[("C1", a), ("C1", b)] for a, b in PAIRS])
    series["C0+C0"] = E([[("C0", a), ("C0", b)] for a, b in PAIRS])
    series["UME_w0.5"] = E([[("C0", s), ("C1", s)] for s in seeds], [1.0, 0.5])
    series["UME_w2"] = E([[("C0", s), ("C1", s)] for s in seeds], [1.0, 2.0])
    if split == "s2":
        series["UME_SigLIP"] = E([[("C0", s), ("YH4_SI", s)] for s in seeds])
        series["UME_ResNet"] = E([[("C0", s), ("C5", s)] for s in seeds])
        series["C0+C1+C5"] = E([[("C0", s), ("C1", s), ("C5", s)] for s in seeds])
        series["C2x3"] = E([[("C2", 42), ("C2", 43), ("C2", 44)]] * 3)
    log(f"[{split}] 앙상블 계산 완료")
    for n in prereg_hybrid_n():
        series[f"B2{n}+UME"] = [combine(b2, c, n, k) for c in series["UME"]]
        series[f"B2{n}+C2"] = [combine(b2, c, n, k) for c in series["C2"]]
    series["B2"] = [b2[:, :k]] * 3

    named = [(f"{name}|{i}", metrics.hits_at_k(c, rows, items, k)) for name, reps in series.items() for i, c in enumerate(reps)]
    log(f"[{split}] 채점 계열 {len(named)}개, 부트스트랩 시작")
    st = series_stats(named, rows, masks, len(users), prereg["measure"]["bootstrap"]["n"],
                      prereg["measure"]["bootstrap"]["seed"], SEGMENTS)

    def lv(name, seg="all"):
        got = [st[(f"{name}|{i}", seg)] for i in range(3)]
        return float(np.mean([g[0] for g in got])), np.mean([g[1] for g in got], axis=0)

    def diff(a, b, seg="all", lvl=0.95):
        pa, ba = lv(a, seg)
        pb, bb = lv(b, seg)
        lo, hi = analysis_v2._q(ba - bb, lvl)
        return {"split": split, "a": a, "b": b, "segment": seg, "R_a": pa, "R_b": pb, "estimate": pa - pb,
                "ci_low": lo, "ci_high": hi, "ci_level": lvl}

    table = pd.DataFrame([{"split": split, "series": name, "segment": seg, "R": lv(name, seg)[0],
                           "R_minus_text_only": lv(name, seg)[0] - lv("C0", seg)[0]}
                          for name in series for seg in SEGMENTS])
    return diff, table, pd.DataFrame(det)


def prereg_hybrid_n():
    return [100, 200, 250, 280]


def run(cfg, exps, paths, log=print):
    prereg = load_yaml("prereg_yh_ume")
    assert prereg["status"] == "final"
    out = results_dir(paths)
    conf = prereg["confirmatory"]
    lvl = conf["ci_level"]
    res = {sp: analyze_split(cfg, paths, prereg, sp, log) for sp in ("s2", "main")}

    tests = {"H15": ("UME", "C2+C2"), "H16": ("UME", "C1+C1"), "H17": ("B2200+UME", "B2200+C2")}
    confirmatory = {"correction": conf["correction"], "created_at": now()}
    for h, (a, b) in tests.items():
        d = res["s2"][0](a, b, lvl=lvl)
        d["supported"] = bool(d["ci_low"] > 0)
        confirmatory[h] = {"name": conf[h]["name"], "comparison": conf[h]["comparison"], **d}
        log(f"{h} {conf[h]['name']} (s2): {d['R_a']:.4f} − {d['R_b']:.4f} = {d['estimate']:+.4f} "
            f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] ({lvl:.2%}) 지지={d['supported']}")
    write_json(out / "confirmatory.json", confirmatory)

    plan = [("UME − 함께 학습 C2 (단일)", "UME", "C2"), ("UME − C2 두 개 앙상블", "UME", "C2+C2"),
            ("UME − 이미지만 두 개 앙상블", "UME", "C1+C1"), ("UME − 텍스트만 두 개 앙상블", "UME", "C0+C0"),
            ("B2 200 + UME − B2 200 + C2", "B2200+UME", "B2200+C2"), ("B2 200 + UME − B2 단독", "B2200+UME", "B2"),
            ("B2 280 + UME − B2 단독", "B2280+UME", "B2")]
    s2_only = [("UME(SigLIP) − UME(Marqo)", "UME_SigLIP", "UME"), ("UME(ResNet) − UME(Marqo)", "UME_ResNet", "UME"),
               ("C0 + C1 + C5 − C2 세 개 앙상블", "C0+C1+C5", "C2x3")]
    con = []
    for sp, (diff, _, _) in res.items():
        for desc, a, b in plan + (s2_only if sp == "s2" else []):
            for seg in ("all", "established", "new_item"):
                con.append({"description": desc, **diff(a, b, seg), "status": "탐색적" if not (sp == "s2" and seg == "all" and (a, b) in tests.values()) else "확증과 같음"})
    contrasts = pd.DataFrame(con)
    contrasts.to_csv(out / "contrasts_yh6.csv", index=False)
    table = pd.concat([res[sp][1] for sp in res], ignore_index=True)
    table.to_csv(out / "table_yh6.csv", index=False)
    det = pd.concat([res[sp][2] for sp in res], ignore_index=True)
    det.to_csv(out / "determinism_yh6.csv", index=False)
    log(f"재학습 결정성: {int(det['identical'].sum()) if len(det) else 0} / {len(det)} 실행이 기존과 같은 추천")
    for r in con:
        if r["segment"] == "all":
            log(f"[{r['split']}] {r['description']}: {r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    return confirmatory, table, contrasts, det
