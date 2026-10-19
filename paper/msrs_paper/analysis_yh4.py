"""6차 사전 등록(configs/prereg_yh_balance.yaml, 김유환) 분석: 우세 모달리티 억제 + 반사실 이미지 활용도 (s2).

확증 (본페로니 2개, 97.5%):
  H10: Δ(SigLIP, 이미지 드롭아웃) − Δ(ResNet, 이미지 드롭아웃)
  H11: Δ(ResNet, 텍스트 드롭아웃) − Δ(SigLIP, 텍스트 드롭아웃)
  Δ(인코더, 처방) = R(처방) − R(기준), seed 평균, 구간 all
탐색: 인코더 × 처방 Δ, 우세 모달리티(이미지만 − 텍스트만), 반사실 활용도 U_img · U_txt, 구간별 Δ.
모든 계열(원래 추천 · 평균 대체 추천)을 같은 부트스트랩 추출(metrics.bootstrap_weights, 같은 seed)로 계산한다.

결과물 (드라이브 results_yh4/): confirmatory.json, effects_yh4.csv, utilization_yh4.csv, dominance_yh4.csv
"""
import numpy as np
import pandas as pd

from . import analysis_v2, metrics
from .analysis_yh3 import series_stats
from .analysis_v2 import item_sales_counts, segment_masks
from .config import load_yaml
from .data_prep import load_split
from .utils import now, write_json

SPLIT = "s2"
GROUP = "YH4"
SEGMENTS = ["all", "established", "sparse", "new_item"]
ENC = {  # 인코더: (기준, 이미지 드롭아웃, 텍스트 드롭아웃, 이미지만)
    "marqo": ("C2", "YH_VD3R", "YH_TD3R", "C1"),
    "siglip_b16": ("S2", "YH4_SVD", "YH4_STD", "YH4_SI"),
    "resnet50": ("C6", "YH4_RVD", "YH4_RTD", "C5"),
    "clipb32": ("C10", "YH4_CVD", "YH4_CTD", "C9"),
}
KINDS = ("base", "imgdrop", "txtdrop", "imgonly")
TEXT_ONLY = ("YH3_ENC", "C0")


def results_dir(paths):
    d = paths.root / "results_yh4"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(cfg, exps, paths, log=print):
    prereg = load_yaml("prereg_yh_balance")
    assert prereg["status"] == "final"
    out = results_dir(paths)
    k = cfg["eval"]["k"]
    seeds = prereg["seeds"]
    _, _, targets = load_split(paths, SPLIT)
    users = np.load(paths.run_dir(GROUP, SPLIT, "C2_seed42") / "users.npy")
    rows = metrics.target_rows(users, targets)
    items = targets["item_idx"].values
    n_items = len(pd.read_parquet(paths.processed / "items.parquet", columns=["idx"]))
    masks = segment_masks(targets, item_sales_counts(paths, n_items, SPLIT), analysis_v2.load_yaml("prereg_v2"))

    # (계열 이름, 적중) 목록: 원래 추천 + 평균 대체 추천
    named, missing = [], []
    def add(name, run_dir, fname="candidates.npy"):
        f = run_dir / fname
        if not f.exists():
            missing.append(str(f))
            return
        u = np.load(run_dir / "users.npy")
        if not np.array_equal(u, users):
            raise ValueError(f"{run_dir}: 테스트 고객 목록이 다릅니다")
        named.append((name, metrics.hits_at_k(np.load(f), rows, items, k)))

    for enc, conds in ENC.items():
        for kind, cond in zip(KINDS, conds):
            for s in seeds:
                d = paths.run_dir(GROUP, SPLIT, f"{cond}_seed{s}")
                add(f"{enc}|{kind}|full|{s}", d)
                if kind != "imgonly":
                    add(f"{enc}|{kind}|cf_img|{s}", d, "candidates_cf_img.npy")
                    add(f"{enc}|{kind}|cf_txt|{s}", d, "candidates_cf_txt.npy")
    for s in seeds:
        add(f"text_only|base|full|{s}", paths.run_dir(TEXT_ONLY[0], SPLIT, f"{TEXT_ONLY[1]}_seed{s}"))
    if missing:
        log(f"없는 파일 {len(missing)}개 (예: {missing[:3]})")
    log(f"채점 계열 {len(named)}개 × 구간 {len(SEGMENTS)}개, 부트스트랩 시작")
    st = series_stats(named, rows, masks, len(users), prereg["measure"]["bootstrap"]["n"],
                      prereg["measure"]["bootstrap"]["seed"], SEGMENTS)

    def lv(enc, kind, view="full", seg="all"):
        got = [st[(f"{enc}|{kind}|{view}|{s}", seg)] for s in seeds if (f"{enc}|{kind}|{view}|{s}", seg) in st]
        if not got:
            return float("nan"), None, 0
        return float(np.mean([g[0] for g in got])), np.mean([g[1] for g in got], axis=0), len(got)

    def delta(enc, kind, seg="all"):
        p1, b1, n1 = lv(enc, kind, seg=seg)
        p0, b0, n0 = lv(enc, "base", seg=seg)
        return p1 - p0, (None if b1 is None or b0 is None else b1 - b0), min(n1, n0)

    conf = prereg["confirmatory"]
    lvl = conf["ci_level"]
    confirmatory = {"correction": conf["correction"], "split": SPLIT, "created_at": now()}
    for h, (a, b, kind) in {"H10": ("siglip_b16", "resnet50", "imgdrop"), "H11": ("resnet50", "siglip_b16", "txtdrop")}.items():
        pa, ba, na = delta(a, kind)
        pb, bb, nb = delta(b, kind)
        lo, hi = analysis_v2._q(ba - bb, lvl)
        complete = na == len(seeds) and nb == len(seeds)
        confirmatory[h] = {"name": conf[h]["name"], "comparison": conf[h]["comparison"],
                           "delta_a": pa, "delta_b": pb, "estimate": pa - pb, "ci_low": lo, "ci_high": hi,
                           "ci_level": lvl, "complete": complete, "supported": bool(complete and lo > 0)}
        log(f"{h} {conf[h]['name']}: {pa:+.4f} − {pb:+.4f} = {pa - pb:+.4f} [{lo:+.4f}, {hi:+.4f}] "
            f"({lvl:.1%}) 지지={confirmatory[h]['supported']}")
    write_json(out / "confirmatory.json", confirmatory)

    # 탐색: 인코더 × 처방 Δ (구간별)
    eff = []
    for enc in ENC:
        for kind in ("imgdrop", "txtdrop"):
            for seg in SEGMENTS:
                p, b, n = delta(enc, kind, seg)
                lo, hi = analysis_v2._q(b, 0.95) if b is not None else (float("nan"), float("nan"))
                eff.append({"encoder": enc, "treatment": kind, "segment": seg, "R_base": lv(enc, "base", seg=seg)[0],
                            "R_treat": lv(enc, kind, seg=seg)[0], "delta": p, "ci_low": lo, "ci_high": hi, "seeds": n})
    effects = pd.DataFrame(eff)
    effects.to_csv(out / "effects_yh4.csv", index=False)

    # 탐색: 우세 모달리티 (이미지만 − 텍스트만)
    pt, bt, _ = lv("text_only", "base")
    dom = []
    for enc in ENC:
        pi, bi, n = lv(enc, "imgonly")
        lo, hi = analysis_v2._q(bi - bt, 0.95) if bi is not None and bt is not None else (float("nan"), float("nan"))
        dom.append({"encoder": enc, "R_image_only": pi, "R_text_only": pt, "image_minus_text": pi - pt,
                    "ci_low": lo, "ci_high": hi, "seeds": n})
    dominance = pd.DataFrame(dom)
    dominance.to_csv(out / "dominance_yh4.csv", index=False)

    # 탐색: 반사실 활용도 U = 1 − R(평균 대체) / R(원래)
    ut = []
    for enc in ENC:
        for kind in ("base", "imgdrop", "txtdrop"):
            pf, bf, _ = lv(enc, kind)
            for view, name in (("cf_img", "U_img"), ("cf_txt", "U_txt")):
                pc, bc, n = lv(enc, kind, view)
                u = 1 - pc / pf
                lo, hi = analysis_v2._q(1 - bc / bf, 0.95) if bc is not None else (float("nan"), float("nan"))
                ut.append({"encoder": enc, "model": kind, "measure": name, "R_full": pf, "R_counterfactual": pc,
                           "utilization": u, "ci_low": lo, "ci_high": hi, "seeds": n})
    util = pd.DataFrame(ut)
    util.to_csv(out / "utilization_yh4.csv", index=False)
    for r in eff:
        if r["segment"] == "all":
            log(f"[탐색] {r['encoder']:11s} {r['treatment']}: Δ {r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    return confirmatory, effects, dominance, util
