"""세 명이 드라이브에 올린 결과를 한 번에 채점하고 합친다.

결과물 (results/):
- table1.csv: 조건별 Recall@300 (구간별), NDCG@12, product_code 수준 Recall. seed 평균·표준편차
- contributions.csv: 인코더 묶음별 기여도(img/content/fine) × 구간 × 상품군, 짝지은 부트스트랩 95% 구간
- overall_contributions.csv: 기여도 자체(고객별 평균 Recall@300 차이) × 인코더 묶음 × 구간, 부트스트랩 95% 구간
- main_test.json: 사전 등록한 주 검정 결과
- cell_counts.csv: 셀별 테스트 구매 수
- fig1.png: 상품군별 (패턴 − 무지) 기여도 차이, 인코더별
- fig2.png: 인코더별 · 구간별 이미지 기여도 (img, content)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .data_prep import APPEARANCE_ORDER, load_items, load_split
from .utils import now, read_json, write_json

MEASURE_CONTROL = {"img": "text_only", "content": "shuffle_global", "fine": "shuffle_type"}


def discover_runs(paths):
    runs = []
    for info_path in sorted(paths.outputs.glob("*/*/*/run_info.json")):
        group = info_path.parts[-4]
        if group == "tuning":
            continue
        info = read_json(info_path)
        runs.append({"group": group, "split": info["split"], "cond": info["cond"], "seed": info["seed"],
                     "dir": info_path.parent})
    return pd.DataFrame(runs)


def cell_ids(targets, pgroups):
    pg = targets["pgroup"].map({g: i for i, g in enumerate(pgroups)})
    ap = targets["appearance"].map({a: i for i, a in enumerate(APPEARANCE_ORDER)})
    ok = pg.notna() & ap.notna()
    return np.where(ok, pg.fillna(0) * len(APPEARANCE_ORDER) + ap.fillna(0), -1).astype(np.int64)


def score_runs(cfg, plan, paths, runs, log=print):
    """모든 실행을 같은 코드로 채점해서 고객별 집계를 캐시에 저장한다."""
    k, ndcg_k = cfg["eval"]["k"], cfg["eval"]["ndcg_k"]
    pgroups = list(plan["product_groups"])
    n_cells = len(pgroups) * len(APPEARANCE_ORDER)
    items = load_items(paths)
    codes = items.sort_values("idx")["product_code"].values.astype(np.int64)
    cache = paths.results / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tallies, users_of = {}, {}
    for split, sruns in runs.groupby("split"):
        _, _, targets = load_split(paths, split)
        cid = cell_ids(targets, pgroups)
        for r in sruns.itertuples():
            key = (r.group, r.split, r.cond, r.seed)
            f = cache / f"{r.group}_{r.split}_{r.cond}_seed{r.seed}.npz"
            if f.exists() and f.stat().st_mtime >= (Path(r.dir) / "run_info.json").stat().st_mtime:
                z = np.load(f)
                users = z["users"]
                t = {s: {m: z[f"{s}__{m}"] for m in ("hits", "n", "ndcg", "cell_hits", "cell_n")}
                     for s in metrics.SEGMENTS}
                t["code"] = {"hits": z["code__hits"], "n": z["code__n"]}
            else:
                cands, users = np.load(Path(r.dir) / "candidates.npy"), np.load(Path(r.dir) / "users.npy")
                t = metrics.user_tallies(cands, users, targets, k, ndcg_k, cid, n_cells)
                t["code"] = metrics.code_level_tallies(cands, users, targets, codes, k)
                flat = {f"{s}__{m}": v for s, d in t.items() for m, v in d.items()}
                np.savez_compressed(f, users=users, **flat)
                log(f"채점 {key}")
            for s in metrics.SEGMENTS:     # 부트스트랩 행렬곱을 BLAS로 빠르게
                t[s]["cell_hits"] = t[s]["cell_hits"].astype(np.float32)
                t[s]["cell_n"] = t[s]["cell_n"].astype(np.float32)
            if split in users_of and not np.array_equal(users_of[split], users):
                raise ValueError(f"{key}: 테스트 고객 목록이 다른 실행과 다릅니다 (다른 분할 파일을 썼을 가능성)")
            users_of[split] = users
            tallies[key] = t
    return tallies, pgroups


def table1(runs, tallies):
    rows = []
    for r in runs.itertuples():
        t = tallies[(r.group, r.split, r.cond, r.seed)]
        row = {"group": r.group, "split": r.split, "cond": r.cond, "seed": r.seed}
        for s in metrics.SEGMENTS:
            row[f"recall@300_{s}"] = metrics.user_mean(t[s]["hits"], t[s]["n"])
        has = t["all"]["n"] > 0
        row["ndcg@12_all"] = float(t["all"]["ndcg"][has].mean())
        row["recall@300_code"] = metrics.user_mean(t["code"]["hits"], t["code"]["n"])
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    value_cols = [c for c in per_seed.columns if c.startswith(("recall", "ndcg"))]
    agg = per_seed.groupby(["group", "split", "cond"])[value_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = per_seed.groupby(["group", "split", "cond"]).size()
    return per_seed, agg.reset_index()


def _pattern_minus_solid(delta_cells, n_cells, strata, n_app):
    """상품군별 (패턴 − 무지) 차이와, 구매 수로 가중 평균한 전체 차이."""
    si, pi = APPEARANCE_ORDER.index("solid"), APPEARANCE_ORDER.index("pattern")
    d_g, w_g = {}, {}
    for g_idx, g in strata:
        cs, cp = g_idx * n_app + si, g_idx * n_app + pi
        d_g[g] = delta_cells[cp] - delta_cells[cs]
        w_g[g] = n_cells[cp] + n_cells[cs]
    total_w = sum(w_g.values())
    d_g["weighted"] = sum(w_g[g] * d_g[g] for g in w_g) / total_w if total_w > 0 else np.nan
    return d_g


def contributions(cfg, exps, plan, runs, tallies, pgroups, log=print):
    """묶음(group, split) × 인코더 × 기여도 종류 × 구간별로 두 가지를 짝지은 부트스트랩 구간과 함께 계산한다.

    - 층화: 상품군별 (패턴 − 무지) 기여도 차이 (구매 건별 비율 기준) → contributions.csv
    - 전체: 기여도 자체 = 고객별 평균 Recall@300(real) − (control) → overall_contributions.csv
    seed가 여러 개면 seed별로 계산해 평균한다 (모든 seed가 같은 부트스트랩 추출을 공유).
    """
    n_app = len(APPEARANCE_ORDER)
    min_n = plan["min_cell_purchases"]
    specs = []   # (group, split, family, measure, seeds)
    for (group, split), g_runs in runs.groupby(["group", "split"]):
        have = {(c, s) for c, s in zip(g_runs["cond"], g_runs["seed"])}
        for fam, conds in exps["families"].items():
            for measure, ctrl in MEASURE_CONTROL.items():
                if "real" not in conds or ctrl not in conds:
                    continue
                seeds = sorted(s for s in g_runs["seed"].unique()
                               if (conds["real"], s) in have and (conds[ctrl], s) in have)
                if seeds:
                    specs.append((group, split, fam, measure, tuple(seeds)))
    if not specs:
        return pd.DataFrame(), pd.DataFrame()

    def run_stats(key, seg, w, memo):
        if (key, seg) not in memo:
            t = tallies[key][seg]
            if w is None:
                memo[(key, seg)] = (t["cell_hits"].sum(0, dtype=np.float64), t["cell_n"].sum(0, dtype=np.float64),
                                    metrics.user_mean(t["hits"], t["n"]))
            else:
                memo[(key, seg)] = (w @ t["cell_hits"], w @ t["cell_n"], metrics.user_mean(t["hits"], t["n"], w))
        return memo[(key, seg)]

    def estimate(spec, seg, w, memo):
        group, split, fam, measure, seeds = spec
        conds = exps["families"][fam]
        cell_d, overall = [], []
        for s in seeds:
            h_real, n, r_real = run_stats((group, split, conds["real"], s), seg, w, memo)
            h_ctrl, _, r_ctrl = run_stats((group, split, conds[MEASURE_CONTROL[measure]], s), seg, w, memo)
            cell_d.append(np.divide(h_real - h_ctrl, n, out=np.full(len(n), np.nan), where=n > 0))
            overall.append(r_real - r_ctrl)
        return np.mean(cell_d, axis=0), n, float(np.mean(overall))

    # 층화 대상 상품군: 패턴·무지 셀 모두 최소 구매 수 이상 (부트스트랩 전에 고정)
    si, pi = APPEARANCE_ORDER.index("solid"), APPEARANCE_ORDER.index("pattern")
    memo0 = {}
    strata_of, point, point_all, n_of = {}, {}, {}, {}
    for spec in specs:
        for seg in metrics.SEGMENTS:
            d_cells, n, ov = estimate(spec, seg, None, memo0)
            key = (spec, seg)
            strata_of[key] = [(i, g) for i, g in enumerate(pgroups)
                              if n[i * n_app + pi] >= min_n and n[i * n_app + si] >= min_n]
            point[key] = _pattern_minus_solid(d_cells, n, strata_of[key], n_app)
            point_all[key], n_of[key] = ov, n
    boot = {key: {g: [] for g in point[key]} for key in point}
    boot_all = {key: [] for key in point}

    n_boot = cfg["eval"]["bootstrap"]
    by_split_users = {s: len(tallies[next(k for k in tallies if k[1] == s)]["all"]["n"]) for s in runs["split"].unique()}
    for split, n_users in by_split_users.items():
        keys = [key for key in point if key[0][1] == split]
        log(f"부트스트랩 {split}: {n_boot}회 × 조합 {len(keys)}개")
        t0 = time.time()
        for it, w in enumerate(metrics.bootstrap_weights(n_users, n_boot, cfg["eval"]["bootstrap_seed"])):
            w = w.astype(np.float32)
            memo = {}
            for key in keys:
                d_cells, n, ov = estimate(key[0], key[1], w, memo)
                for g, v in _pattern_minus_solid(d_cells, n, strata_of[key], n_app).items():
                    boot[key][g].append(v)
                boot_all[key].append(ov)
            if (it + 1) % 100 == 0:
                log(f"  {split} {it + 1}/{n_boot} ({time.time() - t0:.0f}초)")

    rows, rows_all = [], []
    for (spec, seg), d in point.items():
        group, split, fam, measure, seeds = spec
        base = {"group": group, "split": split, "family": fam, "measure": measure, "segment": seg,
                "seeds": ",".join(map(str, seeds))}
        n_cells = n_of[(spec, seg)]
        for g, v in d.items():
            lo, hi = metrics.ci(boot[(spec, seg)][g])
            row = {**base, "stratum": g, "estimate": v, "ci_low": lo, "ci_high": hi}
            if g != "weighted":
                i = pgroups.index(g)
                row["n_solid"], row["n_pattern"] = int(n_cells[i * n_app + si]), int(n_cells[i * n_app + pi])
            rows.append(row)
        lo, hi = metrics.ci(boot_all[(spec, seg)])
        rows_all.append({**base, "estimate": point_all[(spec, seg)], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows), pd.DataFrame(rows_all)


def main_test(plan, contrib):
    mt = plan["main_test"]
    sel = contrib[(contrib["group"] == mt["group"]) & (contrib["split"] == mt["split"])
                  & (contrib["family"] == mt["family"]) & (contrib["measure"] == mt["measure"])
                  & (contrib["segment"] == mt["segment"]) & (contrib["stratum"] == "weighted")]
    if sel.empty:
        return {"status": "not_available", "spec": mt}
    r = sel.iloc[0]
    expected = ",".join(map(str, mt["seeds"]))
    return {"status": "ok" if r["seeds"] == expected else f"seeds_incomplete ({r['seeds']} / {expected})",
            "estimate": float(r["estimate"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
            "supports_hypothesis": bool(r["ci_low"] > 0), "spec": mt, "created_at": now()}


def cell_counts(paths, pgroups):
    rows = []
    for split in ("main", "alt"):
        if not (paths.split_dir(split) / "test_targets.parquet").exists():
            continue
        _, _, t = load_split(paths, split)
        for seg in metrics.SEGMENTS:
            m = metrics.segment_mask(t, seg)
            c = t[m].groupby(["pgroup", "appearance"]).size().rename("n").reset_index()
            c["split"], c["segment"] = split, seg
            rows.append(c)
    return pd.concat(rows, ignore_index=True)


def figure1(plan, contrib, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mt = plan["main_test"]
    d = contrib[(contrib["split"] == mt["split"]) & (contrib["measure"] == mt["measure"])
                & (contrib["segment"] == mt["segment"])]
    if d.empty:
        return None
    strata = [g for g in list(plan["product_groups"]) + ["weighted"] if g in set(d["stratum"])]
    fams = list(dict.fromkeys(d["family"]))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    width = 0.8 / len(fams)
    for j, fam in enumerate(fams):
        f = d[d["family"] == fam].set_index("stratum").reindex(strata)
        x = np.arange(len(strata)) + j * width
        err = [f["estimate"] - f["ci_low"], f["ci_high"] - f["estimate"]]
        ax.bar(x, f["estimate"], width, yerr=err, capsize=2, label=fam)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(strata)) + 0.4 - width / 2)
    ax.set_xticklabels(strata)
    ax.set_ylabel("pattern − solid\n(Recall@300 contribution)")
    ax.set_title(f"{mt['measure']} contribution, {mt['segment']} purchases")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def figure2(overall, out_path, split="main"):
    """인코더 묶음별 이미지 기여도. 왼쪽 img(텍스트만 대비), 오른쪽 content(전체 섞기 대비)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = overall[overall["split"] == split].drop_duplicates(["family", "measure", "segment"])
    if d.empty:
        return None
    segs = ["all", "non_repurchase", "new_item"]
    fams = list(dict.fromkeys(d["family"]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, measure in zip(axes, ["img", "content"]):
        width = 0.8 / max(len(fams), 1)
        for j, fam in enumerate(fams):
            f = d[(d["family"] == fam) & (d["measure"] == measure)].set_index("segment").reindex(segs)
            x = np.arange(len(segs)) + j * width
            err = [f["estimate"] - f["ci_low"], f["ci_high"] - f["estimate"]]
            ax.bar(x, f["estimate"], width, yerr=err, capsize=2, label=fam)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(segs)) + 0.4 - width / 2)
        ax.set_xticklabels(segs)
        ax.set_title({"img": "image − text only", "content": "image − shuffled image"}[measure])
    axes[0].set_ylabel("Δ Recall@300")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_all(cfg, exps, plan, paths, log=print):
    runs = discover_runs(paths)
    if runs.empty:
        raise RuntimeError("outputs/ 에 완료된 실행이 없습니다.")
    log(f"완료된 실행 {len(runs)}개")
    tallies, pgroups = score_runs(cfg, plan, paths, runs, log)
    per_seed, t1 = table1(runs, tallies)
    per_seed.to_csv(paths.results / "table1_per_seed.csv", index=False)
    t1.to_csv(paths.results / "table1.csv", index=False)
    cell_counts(paths, pgroups).to_csv(paths.results / "cell_counts.csv", index=False)
    contrib, overall = contributions(cfg, exps, plan, runs, tallies, pgroups, log)
    contrib.to_csv(paths.results / "contributions.csv", index=False)
    overall.to_csv(paths.results / "overall_contributions.csv", index=False)
    mt = main_test(plan, contrib) if not contrib.empty else {"status": "not_available"}
    write_json(paths.results / "main_test.json", mt)
    if not contrib.empty:
        figure1(plan, contrib, paths.results / "fig1.png")
        figure2(overall, paths.results / "fig2.png")
    return t1, contrib, overall, mt
