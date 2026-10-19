"""7차 사전 등록(configs/prereg_yh_image_use.yaml, 김유환) 분석: 텍스트만 대비 이미지를 함께 쓰는 방식 비교 (s4).

확증 (본페로니 3개, 98.33%):
  H12: 인코더 4종 평균 [R(이미지 드롭아웃 0.3) − R(기준)]
  H13: R(Marqo 이미지 잡음 σ*) − R(C2)     σ*: s4 검증 주 튜닝 1등
  H14: R(UME: 텍스트만 C0 + 이미지만 C1, w = 1) − R(C2)
탐색: 인코더별 Δ, 잡음 σ별 · 잡음 − 드롭아웃, UME w · ResNet, 외우기 정도(학습 주 R − 테스트 주 R), 구간별, 텍스트만 대비 표.
모든 계열을 같은 부트스트랩 추출(metrics.bootstrap_weights, 같은 seed)로 계산한다 (analysis_yh3.series_stats).

결과물 (드라이브 results_yh5/): confirmatory.json, methods_yh5.csv, contrasts_yh5.csv, memorization_yh5.csv, tuning_yh5.csv
"""
import numpy as np
import pandas as pd

from . import analysis_v2, metrics
from .analysis_v2 import item_sales_counts, segment_masks
from .analysis_yh3 import series_stats
from .config import load_yaml
from .data_prep import load_split
from .utils import now, read_json, write_json

SPLIT = "s4"
SEGMENTS = ["all", "established", "sparse", "new_item"]
ENC = {"marqo": ("C2", "YH_VD3R"), "siglip_b16": ("S2", "YH4_SVD"), "resnet50": ("C6", "YH4_RVD"), "clipb32": ("C10", "YH4_CVD")}
NOISE = {0.03: "YH5_N1", 0.06: "YH5_N2", 0.12: "YH5_N3"}


def results_dir(paths):
    d = paths.root / "results_yh5"
    d.mkdir(parents=True, exist_ok=True)
    return d


def select_noise(paths):
    rows = []
    for sigma, cond in NOISE.items():
        f = paths.outputs / "tuning" / SPLIT / f"{cond}_seed42_h2" / "run_info.json"
        if not f.exists():
            raise FileNotFoundError(f"튜닝 결과 없음: {f} — 12 노트북 튜닝 셀을 먼저 실행하세요")
        rows.append({"sigma": sigma, "cond": cond, "valid_recall@300": read_json(f)["valid_recall@300"]})
    ref = paths.outputs / "tuning" / SPLIT / "C2_seed42_h2" / "run_info.json"
    if ref.exists():
        rows.append({"sigma": 0.0, "cond": "C2 (참고)", "valid_recall@300": read_json(ref)["valid_recall@300"]})
    t = pd.DataFrame(rows).sort_values("valid_recall@300", ascending=False)
    best = t[t["cond"].isin(NOISE.values())].iloc[0]
    return float(best["sigma"]), best["cond"], t


def ume_candidates(dir_t, dir_i, w, k, device=None, chunk=2048):
    """따로 학습한 두 모델의 점수를 고객별로 표준화해 더한 상위 k개."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Ut = torch.from_numpy(np.load(dir_t / "user_vectors.npy").astype(np.float32)).to(device)
    Vt = torch.from_numpy(np.load(dir_t / "item_vectors.npy").astype(np.float32)).to(device)
    Ui = torch.from_numpy(np.load(dir_i / "user_vectors.npy").astype(np.float32)).to(device)
    Vi = torch.from_numpy(np.load(dir_i / "item_vectors.npy").astype(np.float32)).to(device)
    out = []
    with torch.no_grad():
        for s in range(0, len(Ut), chunk):
            st, si = Ut[s:s + chunk] @ Vt.T, Ui[s:s + chunk] @ Vi.T
            z = lambda x: (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)
            out.append((z(st) + w * z(si)).topk(k, dim=1).indices.int().cpu().numpy())
    return np.concatenate(out)


def run(cfg, exps, paths, log=print):
    prereg = load_yaml("prereg_yh_image_use")
    assert prereg["status"] == "final"
    out = results_dir(paths)
    k, seeds = cfg["eval"]["k"], prereg["seeds"]
    sigma, noise_cond, tuning = select_noise(paths)
    tuning.to_csv(out / "tuning_yh5.csv", index=False)
    log(f"이미지 잡음 σ* = {sigma} ({noise_cond})")

    _, _, targets = load_split(paths, SPLIT)
    users = np.load(paths.run_dir("YH5_ENC", SPLIT, "C2_seed42") / "users.npy")
    rows = metrics.target_rows(users, targets)
    items = targets["item_idx"].values
    n_items = len(pd.read_parquet(paths.processed / "items.parquet", columns=["idx"]))
    masks = segment_masks(targets, item_sales_counts(paths, n_items, SPLIT), load_yaml("prereg_v2"))

    named = []
    def add(name, cands, run_dir=None):
        if run_dir is not None and not np.array_equal(np.load(run_dir / "users.npy"), users):
            raise ValueError(f"{run_dir}: 테스트 고객 목록이 다릅니다")
        named.append((name, metrics.hits_at_k(cands, rows, items, k)))

    runs = {}   # (조건 이름, seed) → 실행 폴더
    for group, conds in (("YH5_ENC", [c for pair in ENC.values() for c in pair]),
                         ("YH5_NOISE", list(NOISE.values())), ("YH5_UME", ["C0", "C1", "C5"])):
        for c in conds:
            for s in seeds:
                d = paths.run_dir(group, SPLIT, f"{c}_seed{s}")
                if (d / "run_info.json").exists():
                    runs[(c, s)] = d
                    add(f"{c}|{s}", np.load(d / "candidates.npy"), d)
    for s in seeds:
        for img, tag in (("C1", "ume_marqo"), ("C5", "ume_resnet")):
            if ("C0", s) in runs and (img, s) in runs:
                for w in (1.0, 0.5, 2.0):
                    add(f"{tag}_w{w}|{s}", ume_candidates(runs[("C0", s)], runs[(img, s)], w, k))
    log(f"채점 계열 {len(named)}개, 부트스트랩 시작")
    st = series_stats(named, rows, masks, len(users), prereg["measure"]["bootstrap"]["n"],
                      prereg["measure"]["bootstrap"]["seed"], SEGMENTS)

    def lv(name, seg="all"):
        got = [st[(f"{name}|{s}", seg)] for s in seeds if (f"{name}|{s}", seg) in st]
        if not got:
            return float("nan"), None, 0
        return float(np.mean([g[0] for g in got])), np.mean([g[1] for g in got], axis=0), len(got)

    def diff(a, b, seg="all", lvl=0.95):
        pa, ba, na = lv(a, seg)
        pb, bb, nb = lv(b, seg)
        lo, hi = analysis_v2._q(ba - bb, lvl) if ba is not None and bb is not None else (float("nan"), float("nan"))
        return {"a": a, "b": b, "segment": seg, "R_a": pa, "R_b": pb, "estimate": pa - pb, "ci_low": lo, "ci_high": hi,
                "ci_level": lvl, "complete": na == len(seeds) and nb == len(seeds)}

    conf = prereg["confirmatory"]
    lvl = conf["ci_level"]
    confirmatory = {"correction": conf["correction"], "split": SPLIT, "image_noise_sigma": sigma, "created_at": now()}
    # H12: 인코더 4종 평균 Δ
    ds, bs, ok = [], [], True
    for enc, (base, vd) in ENC.items():
        p1, b1, n1 = lv(vd)
        p0, b0, n0 = lv(base)
        ok = ok and n1 == len(seeds) and n0 == len(seeds)
        ds.append(p1 - p0)
        bs.append(b1 - b0)
    lo, hi = analysis_v2._q(np.mean(bs, axis=0), lvl)
    confirmatory["H12"] = {"name": conf["H12"]["name"], "comparison": conf["H12"]["comparison"],
                           "per_encoder": dict(zip(ENC, ds)), "estimate": float(np.mean(ds)), "ci_low": lo, "ci_high": hi,
                           "ci_level": lvl, "complete": ok, "supported": bool(ok and lo > 0)}
    for h, (a, b) in {"H13": (noise_cond, "C2"), "H14": ("ume_marqo_w1.0", "C2")}.items():
        d = diff(a, b, lvl=lvl)
        d["supported"] = bool(d["complete"] and d["ci_low"] > 0)
        confirmatory[h] = {"name": conf[h]["name"], "comparison": conf[h]["comparison"], **d}
    write_json(out / "confirmatory.json", confirmatory)
    for h in ("H12", "H13", "H14"):
        c = confirmatory[h]
        log(f"{h} {c['name']}: {c['estimate']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] ({lvl:.2%}) 지지={c['supported']}")

    # 텍스트만 대비 표 (방법별)
    methods = [("텍스트만", "C0"), ("이미지만 (Marqo)", "C1"), ("텍스트 + Marqo 이어 붙이기", "C2"),
               ("+ 이미지 드롭아웃 0.3", "YH_VD3R"), (f"+ 이미지 잡음 σ* = {sigma}", noise_cond),
               ("따로 학습 후 점수 합산 (UME, w = 1)", "ume_marqo_w1.0")] + \
              [(f"이미지 잡음 σ = {sg}", c) for sg, c in NOISE.items()] + \
              [("UME w = 0.5", "ume_marqo_w0.5"), ("UME w = 2", "ume_marqo_w2.0"),
               ("ResNet 이어 붙이기", "C6"), ("ResNet + 이미지 드롭아웃", "YH4_RVD"), ("UME ResNet (C0 + C5)", "ume_resnet_w1.0"),
               ("SigLIP 이어 붙이기", "S2"), ("SigLIP + 이미지 드롭아웃", "YH4_SVD"),
               ("OpenAI CLIP 이어 붙이기", "C10"), ("OpenAI CLIP + 이미지 드롭아웃", "YH4_CVD")]
    mt = []
    for label, name in methods:
        for seg in SEGMENTS:
            d = diff(name, "C0", seg)
            mt.append({"method": label, "series": name, "segment": seg, "R": d["R_a"], "R_minus_text_only": d["estimate"],
                       "ci_low": d["ci_low"], "ci_high": d["ci_high"]})
    methods_df = pd.DataFrame(mt)
    methods_df.to_csv(out / "methods_yh5.csv", index=False)

    # 탐색 대비
    plan = [(f"{e}: 이미지 드롭아웃 − 기준", vd, base) for e, (base, vd) in ENC.items()] + [
        ("Marqo: 이미지 잡음 σ* − 이미지 드롭아웃 0.3", noise_cond, "YH_VD3R"),
        ("UME(Marqo) − 이미지 드롭아웃 0.3", "ume_marqo_w1.0", "YH_VD3R"),
        ("UME(ResNet) − ResNet 이어 붙이기", "ume_resnet_w1.0", "C6"),
    ]
    con = [{"description": desc, **diff(a, b, seg), "status": "탐색적"} for desc, a, b in plan for seg in ("all", "established", "new_item")]
    contrasts = pd.DataFrame(con)
    contrasts.to_csv(out / "contrasts_yh5.csv", index=False)

    # 외우기 정도: 학습 주 R − 테스트 주 R (run_info, seed 평균 · 표준편차)
    mem = []
    for (c, s), d in runs.items():
        info = read_json(d / "run_info.json")
        if "insample_recall@300" in info:
            mem.append({"cond": c, "seed": s, "insample": info["insample_recall@300"], "test": info["test_recall@300"]})
    mem = pd.DataFrame(mem)
    if len(mem):
        mem["gap"] = mem["insample"] - mem["test"]
        mem = mem.groupby("cond")[["insample", "test", "gap"]].agg(["mean", "std"]).round(4)
        mem.columns = [f"{a}_{b}" for a, b in mem.columns]
        mem = mem.reset_index()
    mem.to_csv(out / "memorization_yh5.csv", index=False)
    for r in con:
        if r["segment"] == "all":
            log(f"[탐색] {r['description']}: {r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    return confirmatory, methods_df, contrasts, mem, tuning
