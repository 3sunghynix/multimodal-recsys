"""10차 사전 등록(configs/prereg_yh_tuned.yaml, 김유환): 규제를 튜닝한 함께 학습 대비 UME, 상위 12개 지표 — 새 테스트 주 s6.

순서: tune() 튜닝 54회(조건 3 × 격자 18) → 설정 선택(results_yh8/tuning.json) → train() 학습 18회 + B2 → run() 분석.

확증 (s6, 구간 all, 본페로니 3개, 98.33%):
  H20: Recall@300(UME*: 튜닝 C0 + 튜닝 C1) − Recall@300(튜닝 C2)
  H21: Recall@300(UME*) − Recall@300(튜닝 C2 두 개 앙상블)
  H22: MAP@12(UME*) − MAP@12(튜닝 C2)
계열 이름: 뒤에 * 가 붙으면 튜닝 설정(s6만), 없으면 기본 설정. UME = C0 + C1 점수를 고객별 z 표준화해 1:1 합.
탐색: 상위 12개 지표 전체, 기본 대 튜닝, 점수 결합 방식(z · 원점수 · min-max · RRF), 적중 분해, main · s2 · s4의 상위 12개 지표,
      학습 주 재채점(외우기 정도), s6의 B2 280 + 후보 20.

결과물 (드라이브 results_yh8/): tuning.csv, tuning.json, confirmatory.json, table_yh8.csv, contrasts_yh8.csv,
                               hits_yh8.csv, memorization_yh8.csv
"""
import itertools

import numpy as np
import pandas as pd

from . import analysis_v2, baselines, metrics, two_tower
from .analysis_v2 import item_sales_counts, segment_masks
from .analysis_yh3 import combine
from .config import load_yaml
from .data_prep import load_split
from .utils import now, read_json, write_json

SPLIT = "s6"
CONDS = ["C0", "C1", "C2"]
SEEDS = [42, 43, 44]
PAIRS = [(42, 43), (43, 44), (44, 42)]
SEGMENTS = ["all", "established", "sparse", "new_item"]
METRICS = ["recall300", "map12", "recall12", "ndcg12"]
SEG_METRICS = ["recall300", "recall12"]          # 구간별로는 적중 기반 지표만
FUSIONS = ["z", "raw", "minmax", "rrf"]
RRF_K = 60
OTHER = {   # 탐색: 이미 있는 주의 기본 설정 실행 (조건 → 묶음). s4의 C2는 벡터가 없어 C2 두 개 앙상블은 계산하지 않는다
    "main": {"C0": "YH6_MAIN", "C1": "YH6_MAIN", "C2": "YH6_MAIN"},
    "s2": {"C0": "YH6_S2", "C1": "YH6_S2", "C2": "YH6_S2"},
    "s4": {"C0": "YH5_UME", "C1": "YH5_UME", "C2": "YH5_ENC"},
}


def results_dir(paths):
    d = paths.root / "results_yh8"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prereg():
    prereg = load_yaml("prereg_yh_tuned")
    assert prereg["status"] == "final", "prereg_yh_tuned.yaml 확정 · 커밋이 필요합니다"
    return prereg


def grid(prereg):
    """튜닝 격자: 이름 → 학습 설정 덮어쓰기. 이름 예: d1_w0_e8 (dropout 0.1, weight decay 0, 8 에폭)."""
    g = prereg["tuning"]["grid"]
    out = {}
    for d, w, e in itertools.product(g["dropout"], g["weight_decay"], g["epochs"]):
        wn = "0" if w == 0 else f"{w:.0e}".replace("e-0", "e-")
        out[f"d{round(d * 10)}_w{wn}_e{e}"] = {"dropout": float(d), "weight_decay": float(w), "epochs": int(e)}
    assert prereg["tuning"]["default"] in out
    return out


def select(table, prereg):
    """조건마다 검증 Recall@300 최고 설정. 같은 값이면 기본 설정, 그다음 격자 순서."""
    order = {hp: i for i, hp in enumerate(grid(prereg))}
    default = prereg["tuning"]["default"]
    sel = {}
    for cond, t in table.groupby("cond"):
        t = t.assign(_def=(t["hp"] != default).astype(int), _ord=t["hp"].map(order))
        sel[cond] = t.sort_values(["valid_recall@300", "_def", "_ord"], ascending=[False, True, True]).iloc[0]["hp"]
    return sel


def tune(cfg, exps, paths, log=print):
    """s6 검증 주, seed 42, 조건 3 × 격자 18 = 54회. 이미 있는 실행은 건너뛴다. 고른 설정을 tuning.json에 고정한다."""
    prereg = _prereg()
    G = grid(prereg)
    rows = []
    for cond in prereg["tuning"]["conds"]:
        for hp, ov in G.items():
            info = two_tower.run(cfg, exps, paths, cond, 42, SPLIT, "tuning", mode="tune", hp_name=hp, overrides=ov, log=log)
            rows.append({"cond": cond, "hp": hp, **ov, "valid_recall@300": info["valid_recall@300"],
                         "train_seconds": info["train_seconds"]})
    table = pd.DataFrame(rows)
    sel = select(table, prereg)
    out = results_dir(paths)
    table.to_csv(out / "tuning.csv", index=False)
    f = out / "tuning.json"
    if f.exists():
        old = read_json(f)["selected"]
        assert old == sel, f"이미 고정된 선택 {old}와 다릅니다 — 튜닝 결과가 바뀌었는지 확인하세요"
    else:
        write_json(f, {"selected": sel, "settings": {c: G[h] for c, h in sel.items()}, "created_at": now()})
    for c, h in sel.items():
        best = table[(table["cond"] == c) & (table["hp"] == h)]["valid_recall@300"].iloc[0]
        base = table[(table["cond"] == c) & (table["hp"] == prereg["tuning"]["default"])]["valid_recall@300"].iloc[0]
        log(f"[튜닝] {c}: {h} 검증 R {best:.4f} (기본 {base:.4f}, {best - base:+.4f})")
    return table, sel


def train(cfg, exps, paths, log=print):
    """기본 · 튜닝 설정 × C0 · C1 · C2 × seed 3 = 18회 (벡터 저장, 학습 주 재채점) + B2."""
    prereg = _prereg()
    G = grid(prereg)
    sel = read_json(results_dir(paths) / "tuning.json")["selected"]
    for seed in SEEDS:
        for cond in CONDS:
            two_tower.run(cfg, exps, paths, cond, seed, SPLIT, "YH8_DEF", log=log, save_vectors=True, insample=True)
            two_tower.run(cfg, exps, paths, cond, seed, SPLIT, "YH8_TUNED", log=log, save_vectors=True, insample=True,
                          overrides=G[sel[cond]])
    baselines.run(cfg, paths, "B2", SPLIT, group="YH8_BASE", log=log)


# ------------------------------------------------------------------ 채점
def fuse(dirs, k, method="z", device=None, chunk=1024):
    """여러 모델 점수의 1:1 결합 상위 k개. method: z(고객별 표준화) · raw(정규화 내적 그대로) · minmax(고객별) · rrf(순위 역수, k = 60)."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    load = lambda d, n: torch.from_numpy(np.load(d / n).astype(np.float32)).to(device)
    UV = [(load(d, "user_vectors.npy"), load(d, "item_vectors.npy")) for d in dirs]

    def norm(x):
        if method == "z":
            return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)
        if method == "raw":
            return x
        if method == "minmax":
            lo, hi = x.min(1, keepdim=True).values, x.max(1, keepdim=True).values
            return (x - lo) / (hi - lo + 1e-8)
        if method == "rrf":
            rank = torch.argsort(torch.argsort(x, dim=1, descending=True), dim=1)   # 0부터
            return 1.0 / (RRF_K + rank.float() + 1.0)
        raise ValueError(method)

    out = []
    with torch.no_grad():
        for s in range(0, len(UV[0][0]), chunk):
            score = sum(norm(U[s:s + chunk] @ V.T) for U, V in UV)
            out.append(score.topk(k, dim=1).indices.int().cpu().numpy())
    return np.concatenate(out)


_IDEAL = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(12) + 2.0))])   # 이상 DCG, 인덱스 = min(구매 수, 12)


def user_metrics(cands, rows, items, U):
    """고객별 Recall@300 · Recall@12 · NDCG@12 · AP@12와 구매가 있는 고객 표시. rows · items는 같은 구간의 구매."""
    n_u = np.bincount(rows, minlength=U)
    has = n_u > 0
    div = lambda a, b: np.divide(a, b, out=np.zeros(U), where=b > 0)
    hit300 = metrics.hits_at_k(cands, rows, items, 300)
    rank = metrics.hit_ranks(cands, rows, items, 12)
    hit12 = rank >= 0
    m12 = np.minimum(n_u, 12)
    gain = np.where(hit12, 1.0 / np.log2(np.maximum(rank, 0) + 2.0), 0.0)
    # AP@12: 맞힌 구매를 고객 · 순위 순으로 놓고, j번째 적중의 정밀도 = j / (순위 + 1)
    r, k = rows[hit12], rank[hit12]
    o = np.lexsort((k, r))
    r, k = r[o], k[o]
    j = pd.Series(np.ones(len(r))).groupby(r).cumsum().values if len(r) else np.zeros(0)
    ap = div(np.bincount(r, weights=j / (k + 1.0), minlength=U), m12)
    return {"recall300": div(np.bincount(rows, weights=hit300, minlength=U), n_u),
            "recall12": div(np.bincount(rows, weights=hit12, minlength=U), n_u),
            "ndcg12": div(np.bincount(rows, weights=gain, minlength=U), _IDEAL[m12]),
            "map12": ap, "has": has}


def boot_stats(named, U, n_boot, seed):
    """(키, 고객별 값, 구매 있음) → {키: (점추정, 부트스트랩 벡터)}. 모든 키가 같은 고객 추출을 공유한다 (짝지은 부트스트랩)."""
    A = np.stack([(v * h).astype(np.float32) for _, v, h in named])
    B = np.stack([h.astype(np.float32) for _, _, h in named])
    point = A.sum(1, dtype=np.float64) / B.sum(1, dtype=np.float64)
    boot = np.empty((len(A), n_boot))
    col, W = 0, []
    for w in metrics.bootstrap_weights(U, n_boot, seed):
        W.append(w.astype(np.float32))
        if len(W) == 100 or col + len(W) == n_boot:
            Wm = np.stack(W, axis=1)
            boot[:, col:col + len(W)] = (A @ Wm) / (B @ Wm)
            col += len(W)
            W = []
    return {key: (point[i], boot[i]) for i, (key, _, _) in enumerate(named)}


def hit_decomposition(series, rows, items, split, primary):
    """구매 단위 적중 겹침. primary: 계열 이름 뒤에 붙는 표시('*' 또는 ''). seed별로 계산해 평균."""
    out = []
    for K in (300, 12):
        H = lambda name, i: metrics.hits_at_k(series[name][i], rows, items, K)
        acc = []
        for i in range(len(SEEDS)):
            T, I, Um, J = (H(n + primary, i) for n in ("C0", "C1", "UME", "C2"))
            T2, I2 = H("C0" + primary, (i + 1) % 3), H("C1" + primary, (i + 1) % 3)
            jac = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
            cats = {"both": T & I, "text_only": T & ~I, "image_only": ~T & I, "neither": ~T & ~I}
            r = {f"share_{c}": m.mean() for c, m in cats.items()}
            r.update({f"ume_hit_given_{c}": Um[m].mean() if m.any() else np.nan for c, m in cats.items()})
            r.update({f"joint_hit_given_{c}": J[m].mean() if m.any() else np.nan for c, m in cats.items()})
            r.update({"ume_extra": (Um & ~T & ~I).mean(), "ume_lost": (~Um & (T | I)).mean(),
                      "joint_extra": (J & ~T & ~I).mean(), "joint_lost": (~J & (T | I)).mean(),
                      "jaccard_text_image": jac(T, I), "jaccard_text_text_seed": jac(T, T2),
                      "jaccard_image_image_seed": jac(I, I2), "hit_text": T.mean(), "hit_image": I.mean(),
                      "hit_ume": Um.mean(), "hit_joint": J.mean()})
            acc.append(r)
        out.append({"split": split, "K": K, "setting": "tuned" if primary else "default",
                    **pd.DataFrame(acc).mean().to_dict()})
    return out


def _runs(paths, split, setting):
    if split == SPLIT:
        g = "YH8_TUNED" if setting == "tuned" else "YH8_DEF"
        return {c: {s: paths.run_dir(g, split, f"{c}_seed{s}") for s in SEEDS} for c in CONDS}
    return {c: {s: paths.run_dir(OTHER[split][c], split, f"{c}_seed{s}") for s in SEEDS} for c in CONDS}


def analyze_split(cfg, paths, prereg, split, log):
    k = cfg["eval"]["k"]
    _, _, targets = load_split(paths, split)
    settings = ["tuned", "default"] if split == SPLIT else ["default"]
    R = {st: _runs(paths, split, st) for st in settings}
    users = np.load(R[settings[0]]["C0"][42] / "users.npy")
    for st in settings:
        for c in CONDS:
            for s in SEEDS:
                assert np.array_equal(np.load(R[st][c][s] / "users.npy"), users), f"{R[st][c][s]}: 테스트 고객 목록이 다릅니다"
    rows = metrics.target_rows(users, targets)
    items = targets["item_idx"].values
    n_items = len(pd.read_parquet(paths.processed / "items.parquet", columns=["idx"]))
    masks = segment_masks(targets, item_sales_counts(paths, n_items, split), load_yaml("prereg_v2"))

    series = {}    # 이름 → 세 짝의 후보
    primary = "*" if split == SPLIT else ""
    for st in settings:
        suf = "*" if st == "tuned" else ""
        vec = lambda c: all((R[st][c][s] / "user_vectors.npy").exists() for s in SEEDS)
        for c in CONDS:
            series[c + suf] = [np.load(R[st][c][s] / "candidates.npy") for s in SEEDS]
        for meth in (FUSIONS if suf == primary else ["z"]):
            name = "UME" + suf + ("" if meth == "z" else f"_{meth}")
            series[name] = [fuse([R[st]["C0"][s], R[st]["C1"][s]], k, meth) for s in SEEDS]
        if vec("C2"):
            series["C2" + suf + "x2"] = [fuse([R[st]["C2"][a], R[st]["C2"][b]], k) for a, b in PAIRS]
        log(f"[{split}] {st} 결합 계산 완료")
    if split == SPLIT:
        bdir = paths.run_dir("YH8_BASE", split, "B2_seed0")
        assert np.array_equal(np.load(bdir / "users.npy"), users), "B2와 모델의 테스트 고객 목록이 다릅니다"
        b2 = np.load(bdir / "candidates.npy")
        series["B2"] = [b2[:, :k]] * 3
        series["B2_280+UME*"] = [combine(b2, c, 280, k) for c in series["UME*"]]
        series["B2_280+C2*"] = [combine(b2, c, 280, k) for c in series["C2*"]]

    U = len(users)
    named = []
    for name, reps in series.items():
        for i, c in enumerate(reps):
            for seg in SEGMENTS:
                m = masks[seg]
                um = user_metrics(c, rows[m], items[m], U)
                for met in (METRICS if seg == "all" else SEG_METRICS):
                    named.append(((name, i, met, seg), um[met], um["has"]))
    log(f"[{split}] 채점 계열 {len(series)}개 × 지표, 부트스트랩 시작")
    st = boot_stats(named, U, prereg["measure"]["bootstrap"]["n"], prereg["measure"]["bootstrap"]["seed"])

    def lv(name, met="recall300", seg="all"):
        got = [st[(name, i, met, seg)] for i in range(3)]
        return float(np.mean([g[0] for g in got])), np.mean([g[1] for g in got], axis=0)

    def diff(a, b, met="recall300", seg="all", lvl=0.95):
        pa, ba = lv(a, met, seg)
        pb, bb = lv(b, met, seg)
        lo, hi = analysis_v2._q(ba - bb, lvl)
        return {"split": split, "a": a, "b": b, "metric": met, "segment": seg, "value_a": pa, "value_b": pb,
                "estimate": pa - pb, "ci_low": lo, "ci_high": hi, "ci_level": lvl}

    base = lambda name: "C0*" if "*" in name else "C0"     # 같은 설정의 텍스트만 대비
    table = pd.DataFrame([{"split": split, "series": name, "metric": met, "segment": seg, "value": lv(name, met, seg)[0],
                           "minus_text_only": lv(name, met, seg)[0] - lv(base(name), met, seg)[0]}
                          for name in series if not name.startswith("B2")
                          for seg in SEGMENTS for met in (METRICS if seg == "all" else SEG_METRICS)]
                         + [{"split": split, "series": name, "metric": "recall300", "segment": "all",
                             "value": lv(name)[0], "minus_text_only": np.nan} for name in series if name.startswith("B2")])
    hits = hit_decomposition(series, rows, items, split, primary)
    memo = []
    for stt in settings:
        for c in CONDS:
            for s in SEEDS:
                info = read_json(R[stt][c][s] / "run_info.json")
                if "insample_recall@300" in info:
                    memo.append({"split": split, "setting": stt, "cond": c, "seed": s, "params": info["params"],
                                 "insample_R": info["insample_recall@300"], "test_R": info["test_recall@300"],
                                 "gap": info["insample_recall@300"] - info["test_recall@300"]})
    return {"diff": diff, "lv": lv, "series": list(series), "table": table, "hits": hits, "memo": memo}


def _available(paths, split):
    return all((r / "candidates.npy").exists() for c in CONDS for r in _runs(paths, split, "default")[c].values()) \
        and all((_runs(paths, split, "default")[c][s] / "user_vectors.npy").exists() for c in ("C0", "C1") for s in SEEDS)


def run(cfg, exps, paths, log=print):
    prereg = _prereg()
    out = results_dir(paths)
    conf = prereg["confirmatory"]
    lvl = conf["ci_level"]
    res = {SPLIT: analyze_split(cfg, paths, prereg, SPLIT, log)}
    for sp in ("main", "s2", "s4"):
        if _available(paths, sp):
            res[sp] = analyze_split(cfg, paths, prereg, sp, log)
        else:
            log(f"[{sp}] 저장된 실행 · 벡터가 없어 건너뜀 (탐색 항목)")

    tests = {"H20": ("UME*", "C2*", "recall300"), "H21": ("UME*", "C2*x2", "recall300"), "H22": ("UME*", "C2*", "map12")}
    confirmatory = {"correction": conf["correction"], "split": SPLIT,
                    "selected": read_json(out / "tuning.json")["selected"], "created_at": now()}
    for h, (a, b, met) in tests.items():
        d = res[SPLIT]["diff"](a, b, met, lvl=lvl)
        d["supported"] = bool(d["ci_low"] > 0)
        confirmatory[h] = {"name": conf[h]["name"], "comparison": conf[h]["comparison"], **d}
        log(f"{h} {conf[h]['name']} ({SPLIT}, {met}): {d['value_a']:.4f} − {d['value_b']:.4f} = {d['estimate']:+.4f} "
            f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] ({lvl:.2%}) 지지={d['supported']}")
    write_json(out / "confirmatory.json", confirmatory)

    con = []
    add = lambda sp, desc, a, b, met="recall300", seg="all": con.append(
        {"description": desc, **res[sp]["diff"](a, b, met, seg),
         "status": "확증과 같음" if sp == SPLIT and seg == "all" and (a, b, met) in tests.values() else "탐색적"})
    for sp, r in res.items():
        p = "*" if sp == SPLIT else ""
        for met in METRICS:
            add(sp, "UME − 함께 학습", "UME" + p, "C2" + p, met)
            if "C2" + p + "x2" in r["series"]:
                add(sp, "UME − 함께 학습 두 개 앙상블", "UME" + p, "C2" + p + "x2", met)
            for meth in FUSIONS[1:]:
                add(sp, f"결합 {meth} − z", f"UME{p}_{meth}", "UME" + p, met)
        for seg in SEGMENTS[1:]:
            for met in SEG_METRICS:
                add(sp, "UME − 함께 학습 (구간)", "UME" + p, "C2" + p, met, seg)
        if sp == SPLIT:
            for met in METRICS:
                add(sp, "기본 UME − 튜닝 C2", "UME", "C2*", met)
                add(sp, "기본 UME − 기본 C2", "UME", "C2", met)
                add(sp, "튜닝 UME − 기본 UME", "UME*", "UME", met)
                for c in CONDS:
                    add(sp, f"튜닝 {c} − 기본 {c}", c + "*", c, met)
            add(sp, "B2 280 + UME* 20 − B2", "B2_280+UME*", "B2")
            add(sp, "B2 280 + UME* 20 − B2 280 + C2* 20", "B2_280+UME*", "B2_280+C2*")
    contrasts = pd.DataFrame(con)
    contrasts.to_csv(out / "contrasts_yh8.csv", index=False)
    table = pd.concat([r["table"] for r in res.values()], ignore_index=True)
    table.to_csv(out / "table_yh8.csv", index=False)
    hits = pd.DataFrame([h for r in res.values() for h in r["hits"]])
    hits.to_csv(out / "hits_yh8.csv", index=False)
    memo = pd.DataFrame([m for r in res.values() for m in r["memo"]])
    memo.to_csv(out / "memorization_yh8.csv", index=False)
    for r in con:
        if r["segment"] == "all" and r["metric"] in ("recall300", "map12"):
            log(f"[{r['split']}] {r['description']} ({r['metric']}): {r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    return confirmatory, table, contrasts, hits, memo
