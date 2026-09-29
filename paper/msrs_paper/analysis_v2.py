"""2차 사전 등록(configs/prereg_v2.yaml) 분석.

측정값: 이미지 내용 기여도
  상대 = [R(real) − R(control)] / R(control), 절대 = R(real) − R(control)
  R = 고객별 평균 Recall@300. 구간별 R은 분자·분모 모두 그 구간 구매로만 계산하고, 그 구간 구매가 없는 고객은 제외.
  control: shuffle_global (content, 확증 검정에 쓰는 값) 또는 shuffle_type (fine, 탐색적)
seed별로 계산해 평균하고, 모든 seed · 조건이 같은 고객 부트스트랩 추출을 공유한다 (짝지은 부트스트랩).

결과물 (드라이브 results_v2/):
  confirmatory.json, replication.json, table_v2.csv, segment_counts_v2.csv, fig_v2.png
"""
import numpy as np
import pandas as pd

from . import metrics
from .config import load_yaml
from .data_prep import load_split
from .utils import now, read_json, write_json

SPLIT = "main"
CONTROL = {"content": "shuffle_global", "fine": "shuffle_type"}


def results_dir(paths):
    d = paths.root / "results_v2"
    (d / "cache").mkdir(parents=True, exist_ok=True)
    return d


def item_sales_counts(paths, n_items):
    """학습에 실제로 쓰는 데이터(history, 테스트 주 이전)에서 센 상품별 판매 횟수 (사전 등록 명확화)."""
    stats = read_json(paths.split_dir(SPLIT) / "stats.json")
    history, _, _ = load_split(paths, SPLIT)
    h = history[history["t_dat"] < pd.Timestamp(stats["test_start"])]
    return np.bincount(h["item_idx"].values, minlength=n_items)


def segment_masks(targets, counts, prereg):
    c = counts[targets["item_idx"].values]
    masks = {
        "all": np.ones(len(targets), dtype=bool),
        "established": c >= 10,
        "sparse": (c >= 1) & (c <= 9),
        "new_item": targets["is_new_item"].values.astype(bool),
    }
    for t in prereg["segments"]["established_sensitivity_thresholds"]:
        masks[f"established_{t}"] = c >= t
    return masks


def selected_id(cfg):
    return cfg.get("id_model", {}).get("selected")


def resolve_cells(exps, cfg, paths):
    """v2_cells → {cell: {role: [(group, cond, seed, run_dir), ...]}}. 없는 실행은 빠진다.
    G 묶음이 없고 선택 설정이 i2면, i2 민감도 셀은 F 묶음 결과를 쓴다 (같은 설정)."""
    seeds = load_yaml("prereg_v2")["measure"]["seeds"]
    g_to_f = {"J0": "I0", "J2": "I2", "J3": "I3", "J4": "I4", "JS2": "IS2", "JS3": "IS3", "JS4": "IS4"}
    out = {}
    for cell, roles in exps["v2_cells"].items():
        out[cell] = {}
        for role, (group, cond) in roles.items():
            found = []
            for s in seeds:
                d = paths.run_dir(group, SPLIT, f"{cond}_seed{s}")
                if not (d / "run_info.json").exists() and group == "G" and selected_id(cfg) == "i2":
                    d = paths.run_dir("F", SPLIT, f"{g_to_f[cond]}_seed{s}")
                if (d / "run_info.json").exists():
                    found.append((group, cond, s, d))
            out[cell][role] = found
    return out


def hit_vector(run_dir, users_ref, rows, items, k, cache_dir):
    """정답 표 각 구매가 그 고객의 상위 k개 후보에 들어갔는지 (bool). 결과는 캐시에 저장."""
    f = cache_dir / (str(run_dir).replace("\\", "/").split("/outputs/")[-1].replace("/", "__") + ".npy")
    if f.exists() and f.stat().st_mtime >= (run_dir / "run_info.json").stat().st_mtime:
        return np.load(f)
    cands, users = np.load(run_dir / "candidates.npy"), np.load(run_dir / "users.npy")
    if not np.array_equal(users, users_ref):
        raise ValueError(f"{run_dir}: 테스트 고객 목록이 다릅니다")
    hit = metrics.hits_at_k(cands, rows, items, k)
    np.save(f, hit)
    return hit


def _q(x, level):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def run(cfg, exps, paths, log=print):
    prereg = load_yaml("prereg_v2")
    out = results_dir(paths)
    k = cfg["eval"]["k"]
    _, _, targets = load_split(paths, SPLIT)
    cells = resolve_cells(exps, cfg, paths)

    # 기준 고객 목록 (모든 실행이 같아야 함)
    any_run = next(r[3] for c in cells.values() for runs in c.values() for r in runs)
    users_ref = np.load(any_run / "users.npy")
    rows = metrics.target_rows(users_ref, targets)
    items = targets["item_idx"].values
    n_items = len(pd.read_parquet(paths.processed / "items.parquet", columns=["idx"]))
    counts = item_sales_counts(paths, n_items)
    masks = segment_masks(targets, counts, prereg)
    U = len(users_ref)

    seg_rows = []
    for seg, m in masks.items():
        seg_rows.append({"segment": seg, "n_purchases": int(m.sum()), "n_customers": int(len(np.unique(rows[m])))})
    pd.DataFrame(seg_rows).to_csv(out / "segment_counts_v2.csv", index=False)
    log(f"구간별 구매 수: { {r['segment']: r['n_purchases'] for r in seg_rows} }")

    # 계열(series) = (실행, 구간). R = (A · w) / (B · w). A = 고객별 Recall × 해당 고객 포함 여부, B = 포함 여부
    run_dirs = sorted({r[3] for c in cells.values() for runs in c.values() for r in runs}, key=str)
    series, A, B = {}, [], []
    for d in run_dirs:
        hit = hit_vector(d, users_ref, rows, items, k, out / "cache")
        for seg, m in masks.items():
            n_u = np.bincount(rows[m], minlength=U)
            h_u = np.bincount(rows[m], weights=hit[m], minlength=U)
            has = n_u > 0
            r_u = np.divide(h_u, n_u, out=np.zeros(U), where=has)
            series[(str(d), seg)] = len(A)
            A.append((r_u * has).astype(np.float32))
            B.append(has.astype(np.float32))
    A, B = np.stack(A), np.stack(B)
    log(f"채점 완료: 실행 {len(run_dirs)}개 × 구간 {len(masks)}개")

    n_boot = prereg["measure"]["bootstrap"]["n"]
    np.seterr(divide="ignore", invalid="ignore")   # 구매가 없는 구간은 NaN으로 남긴다
    R_point = A.sum(1, dtype=np.float64) / B.sum(1, dtype=np.float64)
    R_boot = np.empty((len(A), n_boot), dtype=np.float64)
    W = []
    col = 0
    for w in metrics.bootstrap_weights(U, n_boot, prereg["measure"]["bootstrap"]["seed"]):
        W.append(w.astype(np.float32))
        if len(W) == 100:
            Wm = np.stack(W, axis=1)
            R_boot[:, col:col + 100] = (A @ Wm) / (B @ Wm)
            col += 100
            W = []
            log(f"  부트스트랩 {col}/{n_boot}")
    if W:
        Wm = np.stack(W, axis=1)
        R_boot[:, col:col + len(W)] = (A @ Wm) / (B @ Wm)

    def contribution(cell, seg, measure, kind):
        """seed 평균 기여도: (점추정, 부트스트랩 벡터, 사용 seed). kind: relative | absolute."""
        roles = cells[cell]
        real = {s: str(d) for _, _, s, d in roles.get("real", [])}
        ctrl = {s: str(d) for _, _, s, d in roles.get(CONTROL[measure], [])}
        seeds = sorted(set(real) & set(ctrl))
        if not seeds:
            return float("nan"), np.full(n_boot, np.nan), []
        pts, boots = [], []
        for s in seeds:
            ir, ic = series[(real[s], seg)], series[(ctrl[s], seg)]
            for R, acc in ((R_point, pts), (R_boot, boots)):
                diff = R[ir] - R[ic]
                acc.append(diff / R[ic] if kind == "relative" else diff)
        return float(np.mean(pts)), np.mean(boots, axis=0), seeds

    conf = prereg["confirmatory"]
    level = conf["ci_level"]
    expected_seeds = prereg["measure"]["seeds"]

    def test(cell_a, cell_b, seg, lvl):
        pa, ba, sa = contribution(cell_a, seg, "content", "relative")
        pb, bb, sb = contribution(cell_b, seg, "content", "relative")
        lo, hi = _q(ba - bb, lvl)
        complete = sa == expected_seeds and sb == expected_seeds
        return {"estimate": pa - pb, "ci_low": lo, "ci_high": hi, "ci_level": lvl,
                "a": {"cell": cell_a, "relative_contribution": pa, "seeds": sa},
                "b": {"cell": cell_b, "relative_contribution": pb, "seeds": sb},
                "status": "ok" if complete else "seeds_incomplete", "supported": bool(complete and lo > 0)}

    confirmatory = {
        "H1": {"name": conf["H1"]["name"], "comparison": conf["H1"]["comparison"],
               **test("noid_marqo", "noid_siglip_b16", "all", level)},
        "H2": {"name": conf["H2"]["name"], "comparison": conf["H2"]["comparison"],
               "id_setting": selected_id(cfg), **test("noid_marqo", "id_marqo", "established", level)},
        "correction": conf["correction"], "created_at": now(),
    }
    replication = {"comparison": conf["replication_pair"],
                   **test("noid_fashionclip2", "noid_laion_b32", "all", 0.95)}
    replication["same_direction_as_H1"] = bool(np.sign(replication["estimate"]) == np.sign(confirmatory["H1"]["estimate"]))
    write_json(out / "confirmatory.json", confirmatory)
    write_json(out / "replication.json", replication)

    table = []
    for cell in cells:
        for seg in masks:
            for measure in ("content", "fine"):
                row = {"cell": cell, "segment": seg, "measure": measure}
                for kind in ("relative", "absolute"):
                    p, b, seeds = contribution(cell, seg, measure, kind)
                    lo, hi = _q(b, 0.95)
                    row.update({f"{kind}": p, f"{kind}_ci_low": lo, f"{kind}_ci_high": hi})
                row["seeds"] = ",".join(map(str, seeds))
                if seeds:
                    table.append(row)
    table = pd.DataFrame(table)
    table.to_csv(out / "table_v2.csv", index=False)
    figure(table, out / "fig_v2.png")
    log(f"H1: {confirmatory['H1']['estimate']:+.4f} [{confirmatory['H1']['ci_low']:+.4f}, "
        f"{confirmatory['H1']['ci_high']:+.4f}] 지지={confirmatory['H1']['supported']}")
    log(f"H2: {confirmatory['H2']['estimate']:+.4f} [{confirmatory['H2']['ci_low']:+.4f}, "
        f"{confirmatory['H2']['ci_high']:+.4f}] 지지={confirmatory['H2']['supported']}")
    return confirmatory, replication, table


def figure(table, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [("noid_siglip_b16", "SigLIP B/16\n(generic)"), ("noid_marqo", "Marqo\n(fashion)"),
             ("noid_laion_b32", "LAION CLIP B/32\n(generic)"), ("noid_fashionclip2", "FashionCLIP 2.0\n(fashion)")]
    d = table[(table["segment"] == "all") & (table["measure"] == "content")].set_index("cell")
    present = [(c, lab) for c, lab in order if c in d.index]
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.arange(len(present))
    est = [d.at[c, "relative"] for c, _ in present]
    err = [[d.at[c, "relative"] - d.at[c, "relative_ci_low"] for c, _ in present],
           [d.at[c, "relative_ci_high"] - d.at[c, "relative"] for c, _ in present]]
    colors = ["#9aa5b1" if "generic" in lab else "#2f6fb0" for _, lab in present]
    ax.bar(x, est, yerr=err, capsize=3, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in present], fontsize=8)
    ax.set_ylabel("relative image contribution\n(vs shuffled image)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
