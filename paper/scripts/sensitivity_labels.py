"""1차 사전 등록 주 검정 민감도: 라벨 노이즈를 제외해도 결과가 유지되는지.

주 검정 사양(configs/analysis_plan.yaml main_test):
  family=marqo (real=C2, shuffle_type=C4), split=main, segment=non_repurchase, measure=fine,
  contrast=[pattern, solid], strata=product_group, weight=purchases, seeds=42·43·44
평가값(k, ndcg_k, bootstrap 반복 수·seed)은 configs/paper.yaml eval을 그대로 읽는다.

두 가지 모드가 있다.
  1) --label-csv 만: 채운 label_check.csv에서 label_correct=0인 상품(확인된 오라벨)만 제외
  2) --exclude-raw-labels "All over pattern,Embroidery" (콤마 구분): 해당 원 라벨 상품 **전부**를
     테스트 정답에서 제외 (오분류가 몰린 라벨을 통째로 빼는 상한 시나리오)
  두 옵션은 함께 쓸 수 있으며, 집합의 합집합으로 제외한다.

이 스크립트는 재학습을 하지 않고, 이미 저장된 후보(outputs/A/main/C{2,4}_seed{s}/candidates.npy)를
그대로 쓰되 테스트 정답에서 지정된 상품의 구매를 뺀 뒤 채점만 다시 한다.

정답·평가 대상 사용자 처리 일관성:
- 제외 대상 상품(item_idx)의 test_targets 구매를 정답 표에서 제거한다.
- users 목록과 candidates.npy는 원본 그대로 사용해 평가 대상 고객 집합을 유지한다.
- 셀별 집계(cell_hits, cell_n)는 필터된 정답에 대해서만 이뤄지므로, 제외 상품 구매는
  해당 셀의 hits와 n에서 자연히 빠진다.
- 어떤 고객이 특정 셀에서 필터로 모든 구매가 빠지면 그 셀의 n_user=0이 되어
  원본 분석(user_mean의 has = n_u > 0)과 동일한 규칙으로 그 셀 기여도에서 제외된다.
- 부트스트랩 가중치 w는 users 전체(46,054명)에 대해 생성되므로 원본 검정과 짝지어진다.
- 층화 대상 상품군은 **원본(필터 전) 정답**의 셀 구매 수로 결정한다 (사전 등록 사양).
- 셀별 델타 계산과 상품군 가중 평균은 analysis._pattern_minus_solid 로직을 그대로 재사용한다.

사용 (cwd=paper/):
  python -m scripts.sensitivity_labels \\
    --label-csv /content/drive/MyDrive/msrs_paper/results/label_check/label_check.csv \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --out /content/drive/MyDrive/msrs_paper/results/sensitivity_labels.json

  # 오분류가 몰린 원 라벨을 통째로 제외:
  python -m scripts.sensitivity_labels \\
    --exclude-raw-labels "All over pattern" \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --out /content/drive/MyDrive/msrs_paper/results/sensitivity_labels_no_allover.json
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
    from msrs_paper.analysis import _pattern_minus_solid, cell_ids
    from msrs_paper.data_prep import APPEARANCE_ORDER
    return metrics, cell_ids, _pattern_minus_solid, APPEARANCE_ORDER


def _weighted_pattern_minus_solid(h_real, h_ctrl, n_agg, strata, n_app, _pattern_minus_solid):
    """analysis._pattern_minus_solid 재사용: 셀별 델타 → 상품군 가중 (패턴 − 무지)."""
    delta = np.divide(h_real - h_ctrl, n_agg,
                      out=np.full_like(n_agg, np.nan, dtype=np.float64), where=n_agg > 0)
    return _pattern_minus_solid(delta, n_agg, strata, n_app)["weighted"]


def _load_run(paths_outputs, group, split, cond, seed):
    d = paths_outputs / group / split / f"{cond}_seed{seed}"
    if not (d / "candidates.npy").exists():
        raise FileNotFoundError(f"실행 결과 없음: {d}")
    return np.load(d / "candidates.npy"), np.load(d / "users.npy")


def _cell_n_from_targets(metrics, cell_ids, targets, pgroups, n_cells):
    """metrics.user_tallies의 non_repurchase 세그먼트에서 cell_n만 얻는 작은 유틸.
    후보 없이 셀 배정만 필요하므로 users를 단일 고객으로 가정한 더미로 계산.
    반환: (n_cells,) — 필터된 targets의 non_repurchase 셀별 구매 수."""
    seg_mask = ~targets["is_repurchase"].values
    ts = targets[seg_mask].reset_index(drop=True)
    cid = cell_ids(ts, pgroups)
    ok = cid >= 0
    return np.bincount(cid[ok], minlength=n_cells).astype(np.float64)


def _print_kv(k, v, log=print):
    log(f"{k}: {v}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label-csv", default=None, help="채운 label_check.csv (label_correct=0인 상품 제외)")
    p.add_argument("--exclude-raw-labels", default=None,
                   help='제외할 원 라벨 목록 (콤마 구분). 예: "All over pattern,Embroidery"')
    p.add_argument("--drive-root", required=True, help="msrs_paper 폴더 (processed/, outputs/, results/)")
    p.add_argument("--out", required=True, help="결과 json 경로")
    args = p.parse_args()

    if not args.label_csv and not args.exclude_raw_labels:
        p.error("--label-csv 또는 --exclude-raw-labels 중 하나는 필요합니다")

    metrics, cell_ids, _pattern_minus_solid_orig, APPEARANCE_ORDER = _load_module()
    root = Path(args.drive_root)
    outputs = root / "outputs"
    processed = root / "processed"

    # 1) 설정 로드 (paper.yaml eval, analysis_plan.yaml main_test)
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    paper_cfg = yaml.safe_load((configs_dir / "paper.yaml").read_text(encoding="utf-8"))
    plan = yaml.safe_load((configs_dir / "analysis_plan.yaml").read_text(encoding="utf-8"))
    k = paper_cfg["eval"]["k"]
    ndcg_k = paper_cfg["eval"]["ndcg_k"]
    n_boot = paper_cfg["eval"]["bootstrap"]
    boot_seed = paper_cfg["eval"]["bootstrap_seed"]
    pgroups = list(plan["product_groups"])
    n_app = len(APPEARANCE_ORDER)
    min_n = plan["min_cell_purchases"]
    seeds = plan["main_test"]["seeds"]
    _print_kv("eval", f"k={k}, ndcg_k={ndcg_k}, n_boot={n_boot}, boot_seed={boot_seed}")
    _print_kv("main_test", f"pgroups={pgroups}, min_cell_purchases={min_n}, seeds={seeds}")

    # 2) 제외 대상 상품 결정
    items = pd.read_parquet(processed / "items.parquet")
    exclude_reasons = {}   # article_id → 사유

    if args.label_csv:
        lab = pd.read_csv(args.label_csv)
        lab["label_correct_num"] = pd.to_numeric(lab["label_correct"], errors="coerce")
        for aid in lab.loc[lab["label_correct_num"] == 0, "article_id"].astype(int):
            exclude_reasons.setdefault(int(aid), []).append("label_correct=0")
        print(f"확인된 오라벨: {sum(1 for r in exclude_reasons.values() if 'label_correct=0' in r)}개")

    exclude_raw_labels = []
    if args.exclude_raw_labels:
        exclude_raw_labels = [s.strip() for s in args.exclude_raw_labels.split(",") if s.strip()]
        mask = items["graphical_appearance_name"].isin(exclude_raw_labels)
        for aid in items.loc[mask, "article_id"].astype(int):
            exclude_reasons.setdefault(int(aid), []).append(f"raw_label∈{exclude_raw_labels}")
        print(f"원 라벨 {exclude_raw_labels} 상품: {int(mask.sum())}개")

    print(f"총 제외 대상 article_id: {len(exclude_reasons)}개")

    idx_map = items.set_index("article_id")["idx"]
    present_aids = [a for a in exclude_reasons if a in idx_map.index]
    excluded_idxs = set(idx_map.loc[present_aids].astype(int).tolist())
    if len(present_aids) < len(exclude_reasons):
        print(f"주의: items.parquet에 없는 article_id {len(exclude_reasons) - len(present_aids)}개")
    print(f"제외할 item_idx: {len(excluded_idxs)}개")

    # 3) 테스트 정답 로드 + 필터
    targets_all = pd.read_parquet(processed / "main" / "test_targets.parquet")
    keep = ~targets_all["item_idx"].isin(excluded_idxs)
    targets = targets_all[keep].reset_index(drop=True)
    n_removed = len(targets_all) - len(targets)
    print(f"테스트 구매: 원본 {len(targets_all):,}건 → 필터 {len(targets):,}건 (제거 {n_removed:,})")

    seg_mask = ~targets["is_repurchase"].values
    targets_seg = targets[seg_mask].reset_index(drop=True)
    cid_seg = cell_ids(targets_seg, pgroups)
    print(f"non_repurchase 구매: {len(targets_seg):,}건 (셀 배정 {(cid_seg >= 0).sum():,}건)")

    # 4) 층화 대상: **원본 정답**의 non_repurchase 셀별 구매 수 기준 (사전 등록 사양)
    n_cells = len(pgroups) * n_app
    n_original_cells = _cell_n_from_targets(metrics, cell_ids, targets_all, pgroups, n_cells)
    si_i, pi_i = APPEARANCE_ORDER.index("solid"), APPEARANCE_ORDER.index("pattern")
    strata = [(i, g) for i, g in enumerate(pgroups)
              if n_original_cells[i * n_app + pi_i] >= min_n
              and n_original_cells[i * n_app + si_i] >= min_n]
    print(f"층화 대상 상품군(원본 기준): {[g for _, g in strata]}")
    for i, g in strata:
        n_s_o = int(n_original_cells[i * n_app + si_i])
        n_p_o = int(n_original_cells[i * n_app + pi_i])
        print(f"  {g}: solid n={n_s_o}, pattern n={n_p_o}")

    # 5) seed별 채점 (real=C2, ctrl=C4)
    per_seed_cells = []   # (h_real, h_ctrl, n_agg) 각 (n_cells,)
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
        t_r = metrics.user_tallies(cands_r, users_ref, targets_seg, k, ndcg_k, cid_seg, n_cells)
        t_c = metrics.user_tallies(cands_c, users_ref, targets_seg, k, ndcg_k, cid_seg, n_cells)
        seg_key = "non_repurchase"    # targets_seg가 이미 non_repurchase만이므로 all==non_repurchase
        ch_r = t_r[seg_key]["cell_hits"].astype(np.float64)
        ch_c = t_c[seg_key]["cell_hits"].astype(np.float64)
        cn = t_r[seg_key]["cell_n"].astype(np.float64)
        per_seed_cells.append((ch_r, ch_c, cn))
        print(f"  seed {s}: real cell_hits {ch_r.sum():.0f} · ctrl {ch_c.sum():.0f} · n {cn.sum():.0f}")

    # 6) 점추정 (seed별 계산 후 평균) — analysis._pattern_minus_solid 재사용
    def seed_val(ch_r, ch_c, cn, w=None):
        if w is None:
            h_r, h_c, n_agg = ch_r.sum(0), ch_c.sum(0), cn.sum(0)
        else:
            h_r, h_c, n_agg = w @ ch_r, w @ ch_c, w @ cn
        return _weighted_pattern_minus_solid(h_r, h_c, n_agg, strata, n_app, _pattern_minus_solid_orig)

    point = float(np.mean([seed_val(*c) for c in per_seed_cells]))
    print(f"\n점추정 (filtered, seed 평균): {point:+.6f}")

    # 7) 부트스트랩 (고객 단위, 짝지은)
    boots = []
    for it, w in enumerate(metrics.bootstrap_weights(U, n_boot, boot_seed)):
        w = w.astype(np.float64)
        vals = [seed_val(*c, w=w) for c in per_seed_cells]
        # NaN(총 가중치 0인 추출) 은 원본과 같이 표본에서 자연 제외
        vals = [v for v in vals if v == v]
        boots.append(float(np.mean(vals)) if vals else float("nan"))
        if (it + 1) % 200 == 0:
            print(f"  부트스트랩 {it + 1}/{n_boot}")
    lo, hi = metrics.ci(boots, level=0.95)

    # 8) 원본 비교
    orig_path = root / "results" / "main_test.json"
    original = json.loads(orig_path.read_text(encoding="utf-8")) if orig_path.exists() else None

    mode_desc = []
    if args.label_csv:
        mode_desc.append("label_correct=0")
    if args.exclude_raw_labels:
        mode_desc.append(f"raw_label∈{exclude_raw_labels}")

    result = {
        "kind": "sensitivity_labels",
        "created_by": "paper/scripts/sensitivity_labels.py",
        "mode": " + ".join(mode_desc),
        "exclude_raw_labels": exclude_raw_labels,
        "excluded_article_ids": sorted(exclude_reasons.keys()),
        "n_excluded_articles": len(exclude_reasons),
        "n_excluded_articles_present": len(excluded_idxs),
        "n_test_purchases_original": int(len(targets_all)),
        "n_test_purchases_filtered": int(len(targets)),
        "n_test_purchases_removed": int(n_removed),
        "n_non_repurchase_filtered": int(len(targets_seg)),
        "strata_used": [g for _, g in strata],
        "strata_basis": "original (unfiltered) targets, per pre-registration",
        "seeds": seeds,
        "n_boot": n_boot,
        "boot_seed": boot_seed,
        "eval_k": k,
        "eval_ndcg_k": ndcg_k,
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
    print(f"모드:            {result['mode']}")
    print(f"제외 상품:       {result['n_excluded_articles']}개 (테스트 구매 {n_removed}건)")
    print(f"필터 후:         {point:+.4f} [{lo:+.4f}, {hi:+.4f}]  지지={result['supports_hypothesis']}")
    if original:
        print(f"원본(전체 상품): {original['estimate']:+.4f} [{original['ci_low']:+.4f}, {original['ci_high']:+.4f}]  지지={original['supports_hypothesis']}")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
