"""UME 재학습 (지속 학습, 명세 4.6) — 서비스 DB의 로그로 운영 중인 UME를 이어서 학습하고 후보 버전으로 등록한다.

흐름 (python -m recsys.retrain, trainer 컨테이너에서 실행)
  1. 기준 모델 = 운영 중(active) UME 버전. 마지막 학습 이후 쌓인 비 setup 이벤트가 --min-new-logs 미만이면 건너뜀
  2. 데이터셋마다: DB 이벤트(구매 1 · 장바구니 0.5 · 클릭 0.25)를 학습 표본으로, 상품 임베딩은 서비스와 같은 DB 값
  3. 최근 비 setup 이벤트 --holdout 비율을 평가용으로 떼고, 나머지로 기준 가중치에서 이어서 학습 (warm start)
  4. 평가용 이벤트를 맞히는지 MAP@12 · Recall@k를 기준 모델과 새 모델로 같은 방식(타워 점수 z 합)으로 비교
  5. 새 가중치 저장 → 채택 규칙(accept)을 통과하면 model_versions에 status="candidate"로 등록 + metric_snapshots에 두 모델 지표
  6. 기준 버전에 학습형 랭커(params "ranker", B1)가 있으면: 후보 타워 기준으로 랭커도 다시 학습해 기준 랭커와 비교
     (recsys.ranker_train.retrain_ranker). 통과하면 후보의 ranker를 새 폴더(retrain_{시각}/ranker/{source})로, 아니면 기준 랭커 유지.
     랭커 특성(텍스트 · 이미지 점수 · 규칙 순위)이 타워 점수라 타워만 바꾸면 랭커가 옛 타워 기준으로 남기 때문.
  운영 교체(activate)는 하지 않는다 — 관리자가 대시보드 · POST /models/{id}/activate로 결정.

데이터셋 하나라도 표본이 모자라면 그 데이터셋은 기준 가중치를 그대로 쓴다 (artifact_dirs 유지).
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.eval.metrics import accuracy_at_k
from recsys.ume import DEFAULT, Sequences, _mlp, build_sequences, recommend, save_tower, tower_vectors, train_tower

MAP_K = 12      # 모니터링 runner · 대시보드와 같은 후보 비교 지표 (MAP@12)


def accept(base_map, new_map, base_recall, new_recall, map_tolerance: float = 0.02, recall_tolerance: float = 0.10) -> bool:
    """채택 규칙 (주 지표 MAP@12 하나, Recall@k는 보조 안전장치):
      새 MAP@12 ≥ 기준 × (1 − map_tolerance)  그리고  새 Recall@k ≥ 기준 × (1 − recall_tolerance)
    1 epoch 이어서 학습이라 변화가 작고, 평가 고객 5,000명 · 정답 1개의 차이는 측정 오차 수준이므로
    "기준 이상"이 아니라 "크게 나빠지지 않음"으로 판정한다. 운영 교체는 여전히 관리자가 결정한다.
    평가 고객이 없어 값이 없으면 판정하지 않고 채택(기존 동작)."""
    if base_map is None or new_map is None:
        return True
    ok_map = new_map >= base_map * (1 - map_tolerance)
    ok_recall = base_recall is None or new_recall is None or new_recall >= base_recall * (1 - recall_tolerance)
    return bool(ok_map and ok_recall)

WEIGHTS = {"purchase": 1.0, "add_to_cart": 0.5, "click": 0.25}       # 학습 정답 가중치 (조회 view는 정답으로 쓰지 않음)
FINE_TUNE = {"epochs": 1, "lr": 1e-4}                                # 이어서 학습: 짧게 · 작은 학습률


# ---------------------------------------------------------------- DB 읽기

def load_base(db, base_id: int | None):
    from sqlalchemy import select

    from app.models import ModelVersion
    stmt = select(ModelVersion).where(ModelVersion.name == "ume")
    stmt = stmt.where(ModelVersion.id == base_id) if base_id else stmt.where(ModelVersion.status == "active")
    base = db.scalars(stmt.order_by(ModelVersion.id.desc())).first()
    if base is None:
        raise SystemExit("기준 UME 버전이 없습니다 (운영 중인 UME가 없거나 --base 확인)")
    return base


def new_log_count(db, since_event_id: int) -> tuple[int, int]:
    """(마지막 학습 이후 비 setup 학습용 이벤트 수, 현재 최대 이벤트 id). 학습 정답 종류(WEIGHTS)만 센다 —
    노출 view · 검색만 쌓여서는 재학습하지 않는다. CT 루프의 /metrics/log-delta?types=click,add_to_cart,purchase(#70)와 같은 수."""
    from sqlalchemy import func, select

    from app.models import Event
    last = int(db.scalar(select(func.max(Event.id))) or 0)
    n = int(db.scalar(select(func.count(Event.id)).where(Event.id > since_event_id, Event.type.in_(list(WEIGHTS)),
                                                         Event.is_setup == False)) or 0)      # noqa: E712
    return n, last


def load_events(db, upto_event_id: int) -> pd.DataFrame:
    from sqlalchemy import select

    from app.models import Event
    rows = db.execute(select(Event.id, Event.user_id, Event.item_id, Event.type, Event.created_at, Event.is_setup)
                      .where(Event.type.in_(list(WEIGHTS)), Event.item_id.is_not(None), Event.id <= upto_event_id)).all()
    df = pd.DataFrame(rows, columns=["event_id", "user_id", "item_id", "type", "ts", "is_setup"])
    df["feedback"] = df["type"].map(WEIGHTS).astype(np.float32)
    return df


# ---------------------------------------------------------------- 학습 · 평가 (데이터셋 하나)

def _tower_from_npz(w: dict, p: dict, device: str):
    import torch
    model = {"user": _mlp(w["user.0.weight"].shape[1], p).to(device), "item": _mlp(w["item.0.weight"].shape[1], p).to(device)}
    for side, net in model.items():
        net.load_state_dict({k: torch.from_numpy(np.asarray(w[f"{side}.{k}"])) for k in net.state_dict()})
        net.eval()
    return model


def recall_at_k(rec: np.ndarray, targets: list[set]) -> float:
    hits = [len(set(r) & t) / len(t) for r, t in zip(rec, targets) if t]
    return float(np.mean(hits)) if hits else 0.0


def split_holdout(ev: pd.DataFrame, holdout: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """평가용 = 최근에 활동한 고객 holdout 비율의 **마지막 비 setup 이벤트** 하나씩 (leave-last-out).
    고객마다 그 전 이벤트가 학습 이력으로 남으므로 평가 고객은 항상 이력이 있다. 평가용 이벤트는 학습에 쓰지 않는다."""
    live = ev[~ev.is_setup]
    per_user = live.groupby("user_id").event_id.agg(["max", "size"])
    per_user = per_user[per_user["size"] >= 2].sort_values("max", ascending=False)   # 이력 1개 이상 남는 고객, 최근 순
    n = int(np.ceil(len(per_user) * holdout))
    if n == 0:
        return ev, ev.iloc[0:0]
    test_ids = per_user["max"].iloc[:n].values
    test = ev[ev.event_id.isin(test_ids)]
    return ev[~ev.event_id.isin(test_ids)], test


def retrain_source(X: dict, ev: pd.DataFrame, base_dir: Path, out_dir: Path, holdout: float, k: int,
                   device: str, max_eval_users: int, min_samples: int, log=print) -> dict:
    """X: {"text": (n, d), "image": (n, d)} — 이 데이터셋 상품만, 행 번호 = ev.item_id. 반환: 보고 dict."""
    base_cfg = json.loads((base_dir / "text" / "config.json").read_text(encoding="utf-8")) \
        if (base_dir / "text" / "config.json").exists() else {}
    p = {**DEFAULT, **{key: base_cfg[key] for key in DEFAULT if key in base_cfg}, **FINE_TUNE}
    train, test = split_holdout(ev, holdout)
    seq = build_sequences(train, p["max_hist"])
    n_samples = len(seq.samples())
    if n_samples < min_samples:
        return {"status": "skipped", "reason": f"학습 표본 {n_samples} < {min_samples}"}

    # 평가 고객: 평가 기간에 정답이 있고 학습 기간에 이력이 있는 고객
    targets = test.groupby("user_id").item_id.apply(set)
    seen = set(train.user_id)
    users = np.array([u for u in targets.index if u in seen], dtype=np.int64)[:max_eval_users]
    catalog = np.arange(len(X["text"]))

    truth = pd.DataFrame([(u, i) for u in users for i in targets[u]], columns=["user_id", "item_id"])

    def score(models) -> tuple[float, float]:
        """(Recall@k, MAP@12). 같은 추천 목록에서 둘 다 계산한다. MAP@12는 공통 평가 코드(recsys.eval.metrics)."""
        U, V = {}, {}
        for m in ("text", "image"):
            U[m], V[m] = tower_vectors(models[m], X[m], seq, users, catalog, p["max_hist"], device)
        rec = recommend(U, V, catalog, min(max(k, MAP_K), len(catalog)), device)
        recall = recall_at_k(rec[:, :k], [targets[u] for u in users])
        mk = min(MAP_K, rec.shape[1])                                  # 상품이 12개보다 적은 카탈로그 방어
        mapk = accuracy_at_k(rec[:, :mk], users, truth, ks=(mk,))[0][f"map@{mk}"]
        return recall, mapk

    base_w = {m: dict(np.load(base_dir / m / "weights.npz")) for m in ("text", "image")}
    base_recall, base_map = score({m: _tower_from_npz(base_w[m], p, device) for m in ("text", "image")}) \
        if len(users) else (None, None)
    t = time.perf_counter()
    new = {m: train_tower(X[m], seq, p, device, log=log, init=base_w[m]) for m in ("text", "image")}
    train_sec = time.perf_counter() - t
    new_recall, new_map = score(new) if len(users) else (None, None)
    meta = {"warm_start_from": str(base_dir), "train_samples": int(n_samples), "trained_until_event_id": int(ev.event_id.max())}
    for m in ("text", "image"):
        save_tower(new[m], out_dir / m, p, {**meta, "modality": m})
    # 모니터링 runner(finish_training)에 넘길 평가 식별자: 같은 평가 집합에서 잰 두 모델만 비교한다
    eval_key = f"leave-last-out:holdout={holdout}:upto={int(ev.event_id.max())}:users={len(users)}"
    return {"status": "trained", "dir": str(out_dir), "train_samples": int(n_samples), "eval_users": int(len(users)),
            f"base_recall@{k}": base_recall, f"new_recall@{k}": new_recall,
            f"base_map@{MAP_K}": base_map, f"new_map@{MAP_K}": new_map, "evaluation_key": eval_key,
            "train_seconds": round(train_sec, 1)}


# ---------------------------------------------------------------- 전체

def run(db, out_root: Path, base_id: int | None = None, min_new_logs: int = 10000, holdout: float = 0.1, k: int = 20,
        device: str = "cpu", max_eval_users: int = 5000, min_samples: int = 100, tolerance: float = 0.02,
        register: bool = True, log=print, recall_tolerance: float = 0.10, ranker: bool = True) -> dict:
    from app.models import MetricSnapshot, ModelVersion
    from app.services.catalog_index import get_index

    base = load_base(db, base_id)
    since = int(base.params.get("trained_until_event_id", 0))
    n_new, last_id = new_log_count(db, since)
    report = {"base_version_id": base.id, "since_event_id": since, "new_logs": n_new, "max_event_id": last_id}
    if n_new < min_new_logs:
        return {**report, "status": "skipped", "reason": f"새 로그 {n_new} < {min_new_logs}"}

    ix = get_index(db)
    ev_all = load_events(db, last_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = out_root / f"retrain_{stamp}"
    dirs = dict(base.params.get("artifact_dirs", {}))
    sources = {}
    for source, base_dir in base.params.get("artifact_dirs", {}).items():
        rows = np.flatnonzero(ix.sources == source)
        ids = ix.ids[rows]
        ev = ev_all[ev_all.item_id.isin(ids)].copy()
        ev["item_id"] = np.searchsorted(ids, ev.item_id.values)          # DB 상품 id → 이 데이터셋 행 번호
        log(f"[{source}] 상품 {len(ids):,} · 이벤트 {len(ev):,}")
        r = retrain_source({"text": ix.text[rows], "image": ix.image[rows]}, ev, Path(base_dir), out / source,
                           holdout, k, device, max_eval_users, min_samples, log)
        sources[source] = r
        if r["status"] == "trained":
            r["accepted"] = accept(r[f"base_map@{MAP_K}"], r[f"new_map@{MAP_K}"], r[f"base_recall@{k}"], r[f"new_recall@{k}"],
                                   tolerance, recall_tolerance)
            if r["accepted"]:
                dirs[source] = r["dir"]
    report.update(status="trained", out=str(out), sources=sources)
    improved = [s for s, r in sources.items() if r.get("accepted")]
    rk_dirs, rankers = dict(base.params.get("ranker") or {}), {}
    if ranker and rk_dirs:
        from recsys.ranker_train import retrain_ranker                  # ranker_eval이 이 모듈을 import → 순환 방지
        for source, base_rk in sorted(rk_dirs.items()):
            if source not in dirs:
                continue
            log(f"[{source}] 랭커 재학습 (타워 {dirs[source]})")
            try:
                rr = retrain_ranker(db, ix, source, dirs[source], base_rk, out / "ranker", tolerance)
            except Exception as e:                                         # 랭커 실패가 타워 재학습 결과를 막지 않게
                rr = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
            rankers[source] = rr
            if rr.get("accepted"):
                rk_dirs[source] = rr["path"]
        report["rankers"] = rankers
    improved_rk = [s for s, r in rankers.items() if r.get("accepted")]
    if register and (improved or improved_rk):
        params = {**base.params, "artifact_dirs": dirs, "trained_until_event_id": last_id, "warm_start_from_version_id": base.id}
        if rk_dirs:
            params["ranker"] = rk_dirs
        mv = ModelVersion(name="ume", version=f"retrain-{stamp}", status="candidate", params=params)
        db.add(mv)
        db.flush()
        for s in improved:
            r = sources[s]
            for metric in (f"recall@{k}", f"map@{MAP_K}"):
                for vid, key in ((base.id, f"base_{metric}"), (mv.id, f"new_{metric}")):
                    if r[key] is not None:
                        db.add(MetricSnapshot(model_version_id=vid, metric=f"{s}/holdout/{metric}", value=r[key], n=r["eval_users"]))
        for s in improved_rk:
            r = rankers[s]
            for vid, key in ((base.id, f"{r['compared_with']}_map@{MAP_K}"), (mv.id, f"new_map@{MAP_K}")):
                db.add(MetricSnapshot(model_version_id=vid, metric=f"{s}/ranker/map@{MAP_K}", value=r[key], n=r["eval_users"]))
        db.commit()
        report["registered_version_id"] = mv.id
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/data/models/ume"), help="새 가중치를 저장할 폴더 (서빙 컨테이너와 공유)")
    ap.add_argument("--base", type=int, default=None, help="기준 UME 버전 id (기본: 운영 중인 UME)")
    ap.add_argument("--min-new-logs", type=int, default=10000, help="마지막 학습 이후 비 setup 이벤트가 이보다 적으면 건너뜀")
    ap.add_argument("--holdout", type=float, default=0.1, help="최근 비 setup 이벤트 중 평가용 비율")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--tolerance", type=float, default=0.02, help="새 MAP@12가 기준의 (1 − tolerance) 이상이어야 채택 (주 지표)")
    ap.add_argument("--recall-tolerance", type=float, default=0.10, help="새 Recall@k가 기준의 (1 − 이 값) 미만이면 거절 (보조 안전장치)")
    ap.add_argument("--no-register", action="store_true", help="가중치 저장 · 평가만 하고 버전 등록은 하지 않음")
    ap.add_argument("--no-ranker", action="store_true", help="기준 버전에 학습형 랭커가 있어도 다시 학습하지 않음 (기준 랭커 유지)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch

    from app.db import SessionLocal
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with SessionLocal() as db:
        r = run(db, a.out, a.base, a.min_new_logs, a.holdout, a.k, device, tolerance=a.tolerance, register=not a.no_register,
                recall_tolerance=a.recall_tolerance, ranker=not a.no_ranker)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
