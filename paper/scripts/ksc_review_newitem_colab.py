# KSC2026 심사 의견 대응: 신규 상품 구간에서 UME vs 이른 결합을 네 분할(main · s2 · s4 · s6)에서 같은 비교 대상으로 확인
# Colab에서 셀 단위로 붙여 넣어 실행한다. 런타임 유형: GPU (셀 3만 GPU가 필요, 셀 2는 CPU로도 됨)
# 재학습 없음. 드라이브에 있는 후보(candidates.npy) · 벡터(user/item_vectors.npy)만 다시 채점한다.

# %% ===== 셀 1: 드라이브 연결 + 코드 =====
from google.colab import drive
drive.mount("/content/drive")

REPO_URL = "https://github.com/yuhwani/multimodal-recsys.git"   # 논문 실험을 돌린 레포 (다르면 이 줄만 수정)
BRANCH = "main"
!rm -rf /content/repo && git clone -q -b {BRANCH} {REPO_URL} /content/repo
!pip -q install -r /content/repo/paper/requirements.txt

import sys
sys.path.insert(0, "/content/repo/paper")
from msrs_paper.config import load_all, Paths
cfg, exps, plan = load_all()
paths = Paths(cfg["drive_root"])          # 기본: /content/drive/MyDrive/msrs_paper
print(paths.root, "존재:", paths.root.exists())


# %% ===== 셀 2: 이미 저장된 결과 확인 (analysis_yh8이 네 분할 구간별 결과를 저장해 둠) =====
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)

t8 = paths.root / "results_yh8" / "table_yh8.csv"
c8 = paths.root / "results_yh8" / "contrasts_yh8.csv"
if t8.exists():
    t = pd.read_csv(t8)
    t = t[(t["metric"] == "recall300") & t["series"].isin(["C0", "C2", "C2x2", "UME", "C0*", "C2*", "C2*x2", "UME*"])]
    print("== Recall@300 (구간별) ==")
    print(t.pivot_table(index=["split", "series"], columns="segment", values="value").round(4))
    print("\n== 텍스트 단일 대비 (구간별) ==")
    print(t.pivot_table(index=["split", "series"], columns="segment", values="minus_text_only").round(4))
else:
    print("table_yh8.csv 없음 → 셀 3으로 다시 계산")
if c8.exists():
    c = pd.read_csv(c8)
    c = c[(c["description"] == "UME − 함께 학습 (구간)") & (c["metric"] == "recall300")]
    print("\n== UME − 이른 결합 (구간, 95% CI) ==")
    print(c[["split", "a", "b", "segment", "value_a", "value_b", "estimate", "ci_low", "ci_high"]].round(4).to_string(index=False))

# 5.1.4절 확인용: B2 단독 Recall@300 (논문의 's2 · main 0.257')
t6 = paths.root / "results_yh6" / "table_yh6.csv"
if t6.exists():
    b = pd.read_csv(t6)
    print("\n== B2 단독 (all) ==")
    print(b[(b["series"] == "B2") & (b["segment"] == "all")][["split", "R"]].round(4).to_string(index=False))


# %% ===== 셀 3: 네 분할 다시 채점 — 같은 비교 대상 + 신뢰구간 (GPU 권장, 약 10 ~ 20분) =====
import numpy as np
from msrs_paper import metrics, analysis_v2
from msrs_paper.analysis_yh8 import _runs, fuse, SEEDS, PAIRS
from msrs_paper.analysis_v2 import item_sales_counts, segment_masks
from msrs_paper.analysis_yh3 import series_stats
from msrs_paper.config import load_yaml
from msrs_paper.data_prep import load_split
from msrs_paper.utils import now, write_json

K = cfg["eval"]["k"]                       # 300
SEGS = ["all", "established", "sparse", "new_item"]
N_BOOT, BOOT_SEED = cfg["eval"]["bootstrap"], cfg["eval"]["bootstrap_seed"]   # 1000, 42 (논문과 동일)
prereg_v2 = load_yaml("prereg_v2")
n_items = len(pd.read_parquet(paths.processed / "items.parquet", columns=["idx"]))


def run_split(split, setting="default"):
    """기본 설정(표 1과 같음)의 C0 · C1 · C2 실행으로 C2, C2×2(벡터가 있을 때), UME를 구간별로 채점."""
    R = _runs(paths, split, setting)
    _, _, targets = load_split(paths, split)
    users = np.load(R["C0"][42] / "users.npy")
    for cnd in ("C0", "C1", "C2"):
        for s in SEEDS:
            assert np.array_equal(np.load(R[cnd][s] / "users.npy"), users), f"{R[cnd][s]}: 테스트 고객 목록이 다름"
    rows = metrics.target_rows(users, targets)
    items = targets["item_idx"].values
    masks = segment_masks(targets, item_sales_counts(paths, n_items, split), prereg_v2)

    series = {cnd: [np.load(R[cnd][s] / "candidates.npy") for s in SEEDS] for cnd in ("C0", "C1", "C2")}
    series["UME"] = [fuse([R["C0"][s], R["C1"][s]], K) for s in SEEDS]
    if all((R["C2"][s] / "user_vectors.npy").exists() for s in SEEDS):
        series["C2x2"] = [fuse([R["C2"][a], R["C2"][b]], K) for a, b in PAIRS]
    else:
        print(f"[{split}] C2 벡터 없음 → C2×2 생략 (s4는 원래 없음)")

    named = [(f"{n}|{i}", metrics.hits_at_k(cand, rows, items, K)) for n, reps in series.items() for i, cand in enumerate(reps)]
    st = series_stats(named, rows, masks, len(users), N_BOOT, BOOT_SEED, SEGS)

    def lv(n, seg):
        got = [st[(f"{n}|{i}", seg)] for i in range(3)]
        return float(np.mean([g[0] for g in got])), np.mean([g[1] for g in got], axis=0)

    out = []
    for seg in SEGS:
        r = {"split": split, "setting": setting, "segment": seg, "n_purchases": int(masks[seg].sum())}
        for n in series:
            r[f"R_{n}"] = lv(n, seg)[0]
        for a, b in [("UME", "C2"), ("UME", "C2x2"), ("C2", "C0"), ("C2x2", "C0"), ("UME", "C0")]:
            if a in series and b in series:
                (pa, ba), (pb, bb) = lv(a, seg), lv(b, seg)
                lo, hi = analysis_v2._q(ba - bb, 0.95)
                r[f"{a}-{b}"] = pa - pb
                r[f"{a}-{b}_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"
        out.append(r)
    print(f"[{split}/{setting}] 완료")
    return out


res = []
for sp in ("main", "s2", "s4", "s6"):
    res += run_split(sp, "default")
res += run_split("s6", "tuned")            # 참고: 5.1.2절의 튜닝 설정

df = pd.DataFrame(res)
outdir = paths.root / "results_newitem"
outdir.mkdir(exist_ok=True)
df.to_csv(outdir / "newitem_by_split.csv", index=False)
write_json(outdir / "run_info.json", {"created_at": now(), "k": K, "bootstrap": N_BOOT, "seed": BOOT_SEED})

cols = ["split", "setting", "n_purchases", "R_C0", "R_C2", "R_C2x2", "R_UME",
        "C2-C0", "C2x2-C0", "UME-C0", "UME-C2", "UME-C2_ci", "UME-C2x2", "UME-C2x2_ci"]
for seg in ("new_item", "established", "all"):
    print(f"\n========== 구간: {seg} ==========")
    print(df[df["segment"] == seg].reindex(columns=cols).round(4).to_string(index=False))
print(f"\n저장: {outdir / 'newitem_by_split.csv'}")
