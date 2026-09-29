"""라벨 표본 검사 CLI. 그리드·확인표 생성과 채운 표 집계를 한 파일로.

사용:
  # 1) 표본 40장 × 그룹 격자와 빈 확인표 생성
  python -m paper.scripts.label_check generate \\
    --items /path/to/processed/items.parquet \\
    --images /path/to/images_subset.zip \\
    --out ./label_check

  # 2) label_check.csv의 label_correct 열에 1(맞음)/0(틀림)을 채운 뒤 집계
  python -m paper.scripts.label_check summarize --csv ./label_check/label_check.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    """paper/를 sys.path에 넣고 msrs_paper를 로드. 로컬 · Colab 어디서든 동작."""
    here = Path(__file__).resolve()
    paper_root = here.parent.parent
    sys.path.insert(0, str(paper_root))
    import msrs_paper.inspect_images as ii
    return ii


def cmd_generate(args):
    ii = _load_module()
    items = pd.read_parquet(args.items)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ii.label_sample(items, args.images, out, per_group=args.per_group, seed=args.seed)
    print(f"완료: {out}/label_check_*.png 격자를 보며 label_check.csv의 label_correct에 1/0을 채우세요.")


def cmd_summarize(args):
    ii = _load_module()
    items = pd.read_parquet(args.items) if args.items else None
    out_dir = Path(args.out) if args.out else Path(args.csv).parent
    result = ii.label_agreement_report(args.csv, items=items, out_dir=out_dir)
    print("[그룹별]")
    print(result["by_group"].to_string(index=False))
    print("\n[원 라벨별]")
    print(result["by_raw"].to_string(index=False))
    if result["n_missing"]:
        print(f"\n주의: {result['n_missing']}행이 아직 채워지지 않았습니다 (집계에서 제외).")
    print("\n[논문 한 줄]")
    print(result["sentence"])
    print(f"\n저장: {out_dir}/label_agreement_*.csv, label_agreement_sentence.txt")


def main():
    p = argparse.ArgumentParser(description="라벨 표본 검사 (생성 · 집계)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="그리드 이미지와 빈 확인표 생성")
    g.add_argument("--items", required=True, help="processed/items.parquet 경로")
    g.add_argument("--images", required=True, help="images_subset.zip 경로")
    g.add_argument("--out", required=True, help="출력 폴더")
    g.add_argument("--per-group", type=int, default=40)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("summarize", help="채운 확인표 집계 · 논문 문장 출력")
    s.add_argument("--csv", required=True, help="채운 label_check.csv 경로")
    s.add_argument("--items", default=None, help="옛 스키마 csv면 join용 items.parquet 경로")
    s.add_argument("--out", default=None, help="저장 폴더 (기본: csv와 같은 폴더)")
    s.set_defaults(func=cmd_summarize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
