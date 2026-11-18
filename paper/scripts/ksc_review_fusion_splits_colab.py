# KSC2026 심사 의견 2 대응: "공동 학습 후 합산(late)"과 "학습형 게이트(gate)"를 main 외 세 분할(s2 · s4 · s6)에서도 학습해 재현 확인
# Colab에서 셀 단위로 붙여 넣어 실행한다. 런타임 유형: GPU (셀 2가 학습)
# main은 이미 YH_FU 묶음에 있으므로 다시 학습하지 않는다. 새 학습 18회 = {late, gate} × {s2, s4, s6} × seed {42, 43, 44}
# 설정은 main과 같다: 기본 학습 설정(h2), 튜닝 없음. 중간에 끊겨도 다시 실행하면 끝난 실행은 건너뛴다.

# %% ===== 셀 1: 드라이브 연결 + 코드 =====
import sys
if any(m.startswith("msrs_paper") for m in sys.modules):
    raise RuntimeError("이전에 불러온 코드가 메모리에 남아 있습니다. '런타임 → 세션 다시 시작' 후 이 셀부터 다시 실행하세요.")
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

REPO_URL = "https://github.com/yuhwani/multimodal-recsys.git"
BRANCH = "main"
!rm -rf /content/repo && git clone -q -b {BRANCH} {REPO_URL} /content/repo
!pip -q install -r /content/repo/paper/requirements.txt
!cd /content/repo && git log -1 --format="코드 커밋: %h %s (%cd)"

sys.path.insert(0, "/content/repo/paper")
from msrs_paper.config import load_all, Paths
cfg, exps, plan = load_all()
paths = Paths(cfg["drive_root"]).make_dirs()
print("드라이브 폴더:", paths.root)
!nvidia-smi --query-gpu=name,memory.total --format=csv

from msrs_paper.utils import verify_manifest, now
m = verify_manifest(paths)
print(f"준비물 {len(m['files'])}개 일치 — 준비물 커밋 {m['git_commit']}")
for c in ("YH_LR", "YH_GR"):
    print(c, exps["conditions"][c])

log_file = paths.logs / "yh9_fusion_splits.log"
def log(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{now()} {msg}\n")


# %% ===== 셀 2: 학습 18회 (GPU) =====
import time
from msrs_paper import two_tower

GROUP = "YH9_FU"                       # 새 묶음 이름 (드라이브 outputs/YH9_FU/<분할>/ 에 저장)
NEW_SPLITS = ["s2", "s4", "s6"]
FU_CONDS = ["YH_LR", "YH_GR"]          # late = 공동 학습 후 합산, gate = 학습형 게이트
SEEDS = [42, 43, 44]

t0 = time.time()
for sp in NEW_SPLITS:
    for s in SEEDS:
        for c in FU_CONDS:
            info = two_tower.run(cfg, exps, paths, c, s, sp, GROUP, log=log)   # mode="final", 기본 설정 = main YH_FU와 동일
            log(f"  {sp} {c} seed{s}: {info['train_seconds']}초 · 누적 {(time.time() - t0) / 60:.1f}분")
log(f"===== {GROUP} 학습 완료 ({(time.time() - t0) / 60:.1f}분) =====")


# %% ===== 셀 3: 채점 — 네 분할에서 이른 결합(C2) · UME와 비교, 구간별 + 95% CI + 게이트 α =====
import numpy as np, pandas as pd
from msrs_paper import metrics, analysis_v2
from msrs_paper.analysis_yh8 import _runs, fuse, SEEDS as S8
from msrs_paper.analysis_v2 import item_sales_counts, segment_masks
from msrs_paper.analysis_yh3 import series_stats
from msrs_paper.config import load_yaml
from msrs_paper.data_prep import load_split
from msrs_paper.utils import write_json

K = cfg["eval"]["k"]
SEGS = ["all", "established", "sparse", "new_item"]
N_BOOT, BOOT_SEED = cfg["eval"]["bootstrap"], cfg["eval"]["bootstrap_seed"]   # 1000, 42
prereg_v2 = load_yaml("prereg_v2")
n_items = len(pd.read_parquet(paths.processed / "items.parquet", columns=["idx"]))
FU_GROUP = {"main": "YH_FU", "s2": GROUP, "s4": GROUP, "s6": GROUP}
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)


def score_split(split):
    R = _runs(paths, split, "default")                   # C0 · C1 · C2 기본 설정 (표 1과 같은 실행)
    for c in FU_CONDS:
        R[c] = {s: paths.run_dir(FU_GROUP[split], split, f"{c}_seed{s}") for s in SEEDS}
    _, _, targets = load_split(paths, split)
    users = np.load(R["C2"][42] / "users.npy")
    for c in ("C0", "C1", "C2", *FU_CONDS):
        for s in SEEDS:
            assert (R[c][s] / "run_info.json").exists(), f"{R[c][s]} 없음 (셀 2 미완료?)"
            assert np.array_equal(np.load(R[c][s] / "users.npy"), users), f"{R[c][s]}: 테스트 고객 목록이 다름"
    rows = metrics.target_rows(users, targets)
    items = targets["item_idx"].values
    masks = segment_masks(targets, item_sales_counts(paths, n_items, split), prereg_v2)

    series = {c: [np.load(R[c][s] / "candidates.npy") for s in SEEDS] for c in ("C0", "C2", *FU_CONDS)}
    series["UME"] = [fuse([R["C0"][s], R["C1"][s]], K) for s in SEEDS]
    named = [(f"{n}|{i}", metrics.hits_at_k(cand, rows, items, K)) for n, reps in series.items() for i, cand in enumerate(reps)]
    st = series_stats(named, rows, masks, len(users), N_BOOT, BOOT_SEED, SEGS)

    def lv(n, seg):
        got = [st[(f"{n}|{i}", seg)] for i in range(3)]
        return float(np.mean([g[0] for g in got])), np.mean([g[1] for g in got], axis=0)

    alpha = np.concatenate([np.load(R["YH_GR"][s] / "gate_alpha.npy").ravel() for s in SEEDS
                            if (R["YH_GR"][s] / "gate_alpha.npy").exists()] or [np.array([np.nan])])
    out = []
    for seg in SEGS:
        r = {"split": split, "segment": seg, "n_purchases": int(masks[seg].sum())}
        for n in series:
            r[f"R_{n}"] = lv(n, seg)[0]
        for a, b in [("YH_LR", "C2"), ("YH_GR", "C2"), ("UME", "YH_LR"), ("UME", "YH_GR"), ("UME", "C2")]:
            (pa, ba), (pb, bb) = lv(a, seg), lv(b, seg)
            lo, hi = analysis_v2._q(ba - bb, 0.95)
            r[f"{a}-{b}"] = pa - pb
            r[f"{a}-{b}_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"
        r["gate_alpha_mean"], r["gate_alpha_sd"] = float(np.nanmean(alpha)), float(np.nanstd(alpha))
        out.append(r)
    print(f"[{split}] 완료")
    return out


res = []
for sp in ("main", "s2", "s4", "s6"):
    res += score_split(sp)
df = pd.DataFrame(res)
outdir = paths.root / "results_fusion_splits"
outdir.mkdir(exist_ok=True)
df.to_csv(outdir / "fusion_variants_by_split.csv", index=False)
write_json(outdir / "run_info.json", {"created_at": now(), "k": K, "bootstrap": N_BOOT, "seed": BOOT_SEED,
                                      "groups": FU_GROUP, "note": "YH_LR = 공동 학습 후 합산(late), YH_GR = 학습형 게이트(gate), C2 = 이른 결합 기본 설정"})

cols = ["split", "R_C0", "R_C2", "R_YH_LR", "R_YH_GR", "R_UME",
        "YH_LR-C2", "YH_LR-C2_ci", "YH_GR-C2", "YH_GR-C2_ci", "UME-YH_LR", "UME-YH_LR_ci", "UME-YH_GR", "UME-YH_GR_ci"]
for seg in SEGS:
    print(f"\n========== 구간: {seg} ==========")
    print(df[df["segment"] == seg].reindex(columns=cols).round(4).to_string(index=False))
print("\n== 게이트 α (이미지 가중치, 상품 전체 · seed 3개) ==")
print(df[df["segment"] == "all"][["split", "gate_alpha_mean", "gate_alpha_sd"]].round(3).to_string(index=False))
print(f"\n저장: {outdir / 'fusion_variants_by_split.csv'}")
