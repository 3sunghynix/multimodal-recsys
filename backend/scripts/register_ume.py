"""UME 모델 버전 등록 (+ 선택: 바로 운영 모델로 교체).

준비: 드라이브 msrs_project/results/ume/{hm,amazon}_test_pca/ 폴더(각각 text/ · image/ 안에 weights.npz)를
      저장소의 data/models/ume/ 아래에 그대로 복사 → 컨테이너에서는 /data/models/ume/...

실행 (컨테이너):
    docker compose exec api python -m scripts.register_ume --activate
설정은 docs/project/오프라인결과.md 3장 (H&M γ=2 · 최근 7일 인기, Amazon γ=2 · 브랜드 β=2).
"""
import argparse
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import ModelVersion


def params_for(root: Path, gamma: float = 2.0, beta_amazon: float = 2.0) -> dict:
    dirs = {src: str(root / f"{src}_test_pca") for src in ("hm", "amazon")}
    missing = [d for d in dirs.values() if not (Path(d) / "text" / "weights.npz").exists()]
    if missing:
        print(f"[주의] 가중치 없음: {missing}. 해당 데이터셋 고객은 인기 모델로 대체된다")
    return {"artifact_dirs": dirs, "gamma": gamma, "beta": {"amazon": beta_amazon}, "pop_recent_days": {"hm": 7}, "max_hist": 50}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/data/models/ume"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--beta-amazon", type=float, default=2.0)
    ap.add_argument("--activate", action="store_true", help="등록 후 운영 모델로 교체 (기존 active → candidate)")
    a = ap.parse_args()
    init_db()
    with SessionLocal() as db:
        mv = ModelVersion(name="ume", version=a.version, params=params_for(a.root, a.gamma, a.beta_amazon))
        db.add(mv)
        db.flush()
        if a.activate:
            for other in db.scalars(select(ModelVersion).where(ModelVersion.status == "active")):
                other.status = "candidate"
            mv.status = "active"
        db.commit()
        print(f"등록: id={mv.id} ume {a.version} status={mv.status} params={mv.params}")


if __name__ == "__main__":
    main()
