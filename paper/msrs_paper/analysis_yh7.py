"""9차 사전 등록(configs/prereg_yh_slots.yaml, 김유환) 분석: B2 + UME 자리 (s4 확증, s2 · main 탐색). 학습 없음.

확증 (s4, 본페로니 2개, 97.5%):
  H18: R(B2 280 + UME 20) − R(B2)
  H19: R(B2 280 + 희소 · 신규(학습 판매 10회 미만) 전용 UME 20) − R(B2)

결과물 (드라이브 results_yh7/): confirmatory.json, slots_yh7.csv, contrasts_yh7.csv
"""
import numpy as np
import pandas as pd

from . import analysis_v2, metrics
from .analysis_v2 import item_sales_counts, segment_masks
from .analysis_yh3 import combine, series_stats
from .analysis_yh6 import ensemble_candidates
from .config import load_yaml
from .data_prep import load_split
from .utils import now, write_json

SEGMENTS = ["all", "established", "sparse", "new_item"]
UME_RUNS = {"s4": "YH5_UME", "s2": "YH6_S2", "main": "YH6_MAIN"}
C2_RUNS = {"s4": "YH5_ENC", "s2": "YH6_S2", "main": "YH6_MAIN"}
B2_RUNS = {"s4": "YH7_BASE", "s2": "YH3_BASE", "main": "C"}
SLOTS = (10, 20, 50)


def results_dir(paths):
    d = paths.root / "results_yh7"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cold_slots(base, model, n_slot, k, cold):
    """base 앞 k − n_slot개 + model 후보 중 cold(상품별 bool)이고 base 앞부분에 없는 것 n_slot개. 모자라면 model 일반 후보로."""
    n_head = k - n_slot
    out = np.empty((len(base), k), dtype=np.int32)
    short = 0
    for i in range(len(base)):
        head = base[i, :n_head]
        hs = set(head.tolist())
        pick = [x for x in model[i] if cold[x] and x not in hs][:n_slot]
        if len(pick) < n_slot:
            short += 1
            ps = set(pick)
            pick += [x for x in model[i] if x not in hs and x not in ps][:n_slot - len(pick)]
        out[i, :n_head] = head
        out[i, n_head:] = pick
    return out, short / len(base)


def analyze_split(cfg, paths, prereg, split, log):
    k = cfg["eval"]["k"]
    seeds = prereg["seeds"]
    _, _, targets = load_split(paths, split)
    b2_dir = paths.run_dir(B2_RUNS[split], split, "B2_seed0")
    users = np.load(b2_dir / "users.npy")
    b2 = np.load(b2_dir / "candidates.npy")
    rows = metrics.target_rows(users, targets)
    items = targets["item_idx"].values
    n_items = len(pd.read_parquet(paths.processed / "items.parquet", columns=["idx"]))
    counts = item_sales_counts(paths, n_items, split)
    cold = counts < 10
    masks = segment_masks(targets, counts, load_yaml("prereg_v2"))
    g = UME_RUNS[split]
    ume = []
    for s in seeds:
        d0, d1 = paths.run_dir(g, split, f"C0_seed{s}"), paths.run_dir(g, split, f"C1_seed{s}")
        assert np.array_equal(np.load(d0 / "users.npy"), users), "UME와 B2의 고객 목록이 다릅니다"
        ume.append(ensemble_candidates([d0, d1], [1.0, 1.0], k))
    c2 = [np.load(paths.run_dir(C2_RUNS[split], split, f"C2_seed{s}") / "candidates.npy") for s in seeds]
    series = {"B2": [b2[:, :k]] * 3}
    shorts = {}
    for n in SLOTS:
        series[f"B2+UME{n}"] = [combine(b2, u, k - n, k) for u in ume]
        res = [cold_slots(b2, u, n, k, cold) for u in ume]
        series[f"B2+coldUME{n}"] = [r[0] for r in res]
        shorts[n] = float(np.mean([r[1] for r in res]))
        series[f"B2+C2_{n}"] = [combine(b2, c, k - n, k) for c in c2]
        series[f"B2+coldC2_{n}"] = [cold_slots(b2, c, n, k, cold)[0] for c in c2]
    log(f"[{split}] 결합 계산 완료, 전용 자리가 모자란 고객 비율: {shorts}")
    named = [(f"{name}|{i}", metrics.hits_at_k(c, rows, items, k)) for name, reps in series.items() for i, c in enumerate(reps)]
    st = series_stats(named, rows, masks, len(users), prereg["measure"]["bootstrap"]["n"],
                      prereg["measure"]["bootstrap"]["seed"], SEGMENTS)

    def lv(name, seg):
        got = [st[(f"{name}|{i}", seg)] for i in range(3)]
        return float(np.mean([x[0] for x in got])), np.mean([x[1] for x in got], axis=0)

    def diff(a, b, seg="all", lvl=0.95):
        pa, ba = lv(a, seg)
        pb, bb = lv(b, seg)
        lo, hi = analysis_v2._q(ba - bb, lvl)
        return {"split": split, "a": a, "b": b, "segment": seg, "R_a": pa, "R_b": pb, "estimate": pa - pb,
                "ci_low": lo, "ci_high": hi, "ci_level": lvl}

    table = pd.DataFrame([{"split": split, "series": name, "segment": seg, "R": lv(name, seg)[0],
                           "R_minus_B2": lv(name, seg)[0] - lv("B2", seg)[0]} for name in series for seg in SEGMENTS])
    return diff, table, shorts


def run(cfg, exps, paths, log=print):
    prereg = load_yaml("prereg_yh_slots")
    assert prereg["status"] == "final"
    out = results_dir(paths)
    conf = prereg["confirmatory"]
    lvl = conf["ci_level"]
    res = {sp: analyze_split(cfg, paths, prereg, sp, log) for sp in ("s4", "s2", "main")}
    confirmatory = {"correction": conf["correction"], "created_at": now(), "cold_short_share_s4": res["s4"][2]}
    for h, a in {"H18": "B2+UME20", "H19": "B2+coldUME20"}.items():
        d = res["s4"][0](a, "B2", lvl=lvl)
        d["supported"] = bool(d["ci_low"] > 0)
        confirmatory[h] = {"name": conf[h]["name"], "comparison": conf[h]["comparison"], **d}
        log(f"{h} {conf[h]['name']} (s4): {d['R_a']:.4f} − {d['R_b']:.4f} = {d['estimate']:+.4f} "
            f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] ({lvl:.1%}) 지지={d['supported']}")
    write_json(out / "confirmatory.json", confirmatory)
    con = []
    for sp, (diff, _, _) in res.items():
        for n in SLOTS:
            for desc, a, b in [(f"B2 + UME {n} − B2", f"B2+UME{n}", "B2"), (f"B2 + 전용 UME {n} − B2", f"B2+coldUME{n}", "B2"),
                               (f"전용 − 일반 UME {n}", f"B2+coldUME{n}", f"B2+UME{n}"),
                               (f"B2 + 전용 UME {n} − B2 + 전용 C2 {n}", f"B2+coldUME{n}", f"B2+coldC2_{n}")]:
                for seg in SEGMENTS:
                    con.append({"description": desc, **diff(a, b, seg)})
    contrasts = pd.DataFrame(con)
    contrasts.to_csv(out / "contrasts_yh7.csv", index=False)
    table = pd.concat([res[sp][1] for sp in res], ignore_index=True)
    table.to_csv(out / "slots_yh7.csv", index=False)
    return confirmatory, table, contrasts
