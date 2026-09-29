"""1차 사전 등록 주 검정 민감도: 라벨 검사에서 확인된 오라벨 상품을 테스트 정답에서 제외해도 결과가 유지되는지.

주 검정 사양(configs/analysis_plan.yaml main_test):
  family=marqo (real=C2, shuffle_type=C4), split=main, segment=non_repurchase, measure=fine,
  contrast=[pattern, solid], strata=product_group, weight=purchases, seeds=42·43·44

이 스크립트는 재학습을 하지 않고, 이미 저장된 후보(outputs/A/main/C{2,4}_seed{s}/candidates.npy)를
그대로 쓰되 테스트 정답에서 오라벨 상품의 구매를 뺀 뒤 채점만 다시 한다.

사용:
  python -m paper.scripts.sensitivity_labels \\
    --label-csv /content/drive/MyDrive/msrs_paper/results/label_check/label_check.csv \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --out /content/drive/MyDrive/msrs_paper/results/sensitivity_labels.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_module():
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent.parent))
    from msrs_paper import metrics
    from msrs_paper.analysis import cell_ids
    from msrs_paper.data_prep import APPEARANCE_ORDER
    return metrics, cell_ids, APPEARANCE_ORDER


def _weighted_pattern_minus_solid(h_real, h_ctrl, n_agg, strata, n_app, appearance_order):
    """가중치가 반영된 cell 집계(h_real, h_ctrl, n_agg: (n_cells,))로 상품군 가중 (패턴 − 무지)."""
    si = appearance_order.index("solid")
    pi = appearance_order.index("pattern")
    num, den = 0.0, 0.0
    for g_idx, _ in strata:
        cs, cp = g_idx * n_app + si, g_idx * n_app + pi
        n_s, n_p = n_agg[cs], n_agg[cp]
        if n_s <= 0 or n_p <= 0:
            continue
        d_s = (h_real[cs] - h_ctrl[cs]) / n_s
        d_p = (h_real[cp] - h_ctrl[cp]) / n_p
        w = n_s + n_p
        num += w * (d_p - d_s)
        den += w
    return float(num / den) if den > 0 else float("nan")


def _load_run(paths_outputs, group, split, cond, seed):
    d = paths_outputs / group / split / f"{cond}_seed{seed}"
    if not (d / "candidates.npy").exists():
        raise FileNotFoundError(f"실행 결과 없음: {d}")
    return np.load(d / "candidates.npy"), np.load(d / "users.npy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label-csv", required=True, help="채운 label_check.csv")
    p.add_argument("--drive-root", required=True, help="msrs_paper 폴더 (processed/, outputs/, results/)")
    p.add_argument("--out", required=True, help="결과 json 경로")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=42)
    args = p.parse_args()

    metrics, cell_ids, APPEARANCE_ORDER = _load_module()
    root = Path(args.drive_root)
    outputs = root / "outputs"
    processed = root / "processed"

    # 1) 오라벨 목록
    lab = pd.read_csv(args.label_csv)
    lab["label_correct_num"] = pd.to_numeric(lab["label_correct"], errors="coerce")
    mislabeled_aids = set(lab.loc[lab["label_correct_num"] == 0, "article_id"].astype(int).tolist())
    print(f"오라벨 상품: {len(mislabeled_aids)}개")
    print(f"  {sorted(mislabeled_aids)}")

    # 2) article_id → item_idx
    items = pd.read_parquet(processed / "items.parquet")
    idx_map = items.set_index("article_id")["idx"]
    missing_aids = mislabeled_aids - set(idx_map.index)
    if missing_aids:
        print(f"주의: items.parquet에 없는 article_id {len(missing_aids)}개 (표본에는 있지만 최종 상품 집합에서 빠짐)")
    excluded_idxs = set(idx_map.loc[list(mislabeled_aids - missing_aids)].astype(int).tolist())
    print(f"제외할 item_idx: {len(excluded_idxs)}개")

    # 3) 테스트 정답 로드 + 필터
    tgt_path = processed / "main" / "test_targets.parquet"
    targets_all = pd.read_parquet(tgt_path)
    keep = ~targets_all["item_idx"].isin(excluded_idxs)
    targets = targets_all[keep].reset_index(drop=True)
    n_removed = len(targets_all) - len(targets)
    print(f"테스트 구매: 원본 {len(targets_all):,}건 → 필터 {len(targets):,}건 (제거 {n_removed:,})")

    # 4) 사전 등록 설정 (스크립트와 같은 레포의 configs/ 사용)
    plan_path = Path(__file__).resolve().parent.parent / "configs" / "analysis_plan.yaml"
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    pgroups = list(plan["product_groups"])
    n_app = len(APPEARANCE_ORDER)
    min_n = plan["min_cell_purchases"]
    seeds = plan["main_test"]["seeds"]
    print(f"pgroups={pgroups}, min_cell_purchases={min_n}, seeds={seeds}")

    # 5) segment=non_repurchase 마스크 (필터된 targets에)
    seg_mask = ~targets["is_repurchase"].values
    targets_seg = targets[seg_mask].reset_index(drop=True)
    cid_seg = cell_ids(targets_seg, pgroups)
    print(f"non_repurchase 구매: {len(targets_seg):,}건 (셀 배정된 것 {(cid_seg >= 0).sum():,}건)")

    # 6) seed별로 채점 (real=C2, ctrl=C4). user_tallies는 셀별 (U, C) hits/n을 반환
    n_cells = len(pgroups) * n_app
    k = 300
    ndcg_k = 12
    per_seed_cells = []  # (h_real, h_ctrl, n_agg) 각 (n_cells,)
    users_ref = None
    U = None
    for s in seeds:
        cands_r, users_r = _load_run(outputs, "A", "main", "C2", s)
        cands_c, users_c = _load_run(outputs, "A", "main", "C4", s)
        if not np.array_equal(users_r, users_c):
            raise ValueError(f"seed {s}: C2와 C4의 users가 다릅니다")
        if users_ref is None:
            users_ref = users_r
            U = len(users_ref)
        elif not np.array_equal(users_r, users_ref):
            raise ValueError(f"seed {s}: users가 다른 seed와 다릅니다")
        # 필터된 targets_seg 전체를 넘기되 segment는 이미 마스킹됨 → user_tallies의 segment_mask는 다시 non_repurchase를 걸러도 결과 동일
        t_r = metrics.user_tallies(cands_r, users_ref, targets_seg, k, ndcg_k, cid_seg, n_cells)
        t_c = metrics.user_tallies(cands_c, users_ref, targets_seg, k, ndcg_k, cid_seg, n_cells)
        # non_repurchase 하위 딕셔너리 (targets_seg가 이미 non_repurchase만이므로 seg="all"과 "non_repurchase" 결과가 같음)
        seg_key = "non_repurchase"
        ch_r = t_r[seg_key]["cell_hits"].astype(np.float64)   # (U, C)
        ch_c = t_c[seg_key]["cell_hits"].astype(np.float64)
        cn = t_r[seg_key]["cell_n"].astype(np.float64)         # real·ctrl 동일
        per_seed_cells.append((ch_r, ch_c, cn))
        print(f"  seed {s}: real cell_hits {ch_r.sum():.0f} · ctrl cell_hits {ch_c.sum():.0f} · n {cn.sum():.0f}")

    # 7) 층화 대상: 부트스트랩 전에 확정 (원본 사양 그대로). 셀별 총 구매 수 기준
    n_total_cells = np.mean([c[2].sum(0) for c in per_seed_cells], axis=0)  # (n_cells,)
    si_i, pi_i = APPEARANCE_ORDER.index("solid"), APPEARANCE_ORDER.index("pattern")
    strata = [(i, g) for i, g in enumerate(pgroups)
              if n_total_cells[i * n_app + pi_i] >= min_n and n_total_cells[i * n_app + si_i] >= min_n]
    print(f"층화 대상 상품군: {[g for _, g in strata]}")
    for i, g in strata:
        print(f"  {g}: solid n={int(n_total_cells[i*n_app+si_i])}, pattern n={int(n_total_cells[i*n_app+pi_i])}")

    # 8) 점추정 (seed별 계산 후 평균)
    def point_seed(ch_r, ch_c, cn):
        h_r = ch_r.sum(0)
        h_c = ch_c.sum(0)
        n_agg = cn.sum(0)
        return _weighted_pattern_minus_solid(h_r, h_c, n_agg, strata, n_app, APPEARANCE_ORDER)

    point = float(np.mean([point_seed(*c) for c in per_seed_cells]))
    print(f"\n점추정 (filtered, seed 평균): {point:+.6f}")

    # 9) 부트스트랩 (고객 단위, 짝지은: 모든 seed·조건이 같은 w 공유)
    boots = []
    for it, w in enumerate(metrics.bootstrap_weights(U, args.n_boot, args.boot_seed)):
        w = w.astype(np.float64)
        per_seed_val = []
        for ch_r, ch_c, cn in per_seed_cells:
            h_r = w @ ch_r
            h_c = w @ ch_c
            n_agg = w @ cn
            per_seed_val.append(_weighted_pattern_minus_solid(h_r, h_c, n_agg, strata, n_app, APPEARANCE_ORDER))
        boots.append(float(np.mean(per_seed_val)))
        if (it + 1) % 200 == 0:
            print(f"  부트스트랩 {it + 1}/{args.n_boot}")
    lo, hi = metrics.ci(boots, level=0.95)

    # 10) 원본과 비교
    orig_path = root / "results" / "main_test.json"
    original = None
    if orig_path.exists():
        original = json.loads(orig_path.read_text(encoding="utf-8"))

    result = {
        "kind": "sensitivity_labels",
        "created_by": "paper/scripts/sensitivity_labels.py",
        "excluded_article_ids": sorted(mislabeled_aids),
        "n_excluded_articles": len(mislabeled_aids),
        "n_excluded_articles_present": len(excluded_idxs),
        "n_test_purchases_original": int(len(targets_all)),
        "n_test_purchases_filtered": int(len(targets)),
        "n_test_purchases_removed": int(n_removed),
        "n_non_repurchase_filtered": int(len(targets_seg)),
        "strata_used": [g for _, g in strata],
        "seeds": seeds,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "point_estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "supports_hypothesis": bool(lo > 0),
        "original": {
            "point_estimate": original.get("estimate") if original else None,
            "ci_low": original.get("ci_low") if original else None,
            "ci_high": original.get("ci_high") if original else None,
            "supports_hypothesis": original.get("supports_hypothesis") if original else None,
        } if original else "not_found",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 결과 요약 ===")
    print(f"오라벨 제외 후:  {point:+.4f} [{lo:+.4f}, {hi:+.4f}]  지지={result['supports_hypothesis']}")
    if original:
        print(f"원본(전체 상품): {original['estimate']:+.4f} [{original['ci_low']:+.4f}, {original['ci_high']:+.4f}]  지지={original['supports_hypothesis']}")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
