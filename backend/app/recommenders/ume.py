"""UME 서빙 (numpy만, torch 불필요). 학습: recsys/ume.py, 결과 · 설정 근거: docs/project/오프라인결과.md

요청 1건
  1. 고객 최근 행동(purchase · add_to_cart · click) 중 데이터셋마다 최근 max_hist개
  2. 그 상품들의 콘텐츠 임베딩(DB, 검색 색인과 공유) 평균 → 고객 타워 MLP → 고객 벡터 (텍스트 · 이미지 각각)
  3. 그 데이터셋 전체 상품 벡터(상품 타워 MLP, 메모리 캐시)와 내적 → 고객별 z(텍스트) + z(이미지)
  4. + γ · 인기 점수(데이터셋별 구매 수 log1p 표준화, 최근 N일) + β · [상품 브랜드 = 고객 최다 브랜드]
  5. 데이터셋이 여럿이면 데이터셋 안 순위 백분위로 합침 (검색과 같은 per_source_rank)
  이력이 없거나 모델이 없는 데이터셋뿐이면 인기 모델로 대체.

model_versions.params 예 (드라이브 results/ume/{source}_test_pca = train + valid로 학습한 가중치):
  {"artifact_dirs": {"hm": "/data/models/ume/hm_test_pca", "amazon": "/data/models/ume/amazon_test_pca"},
   "gamma": 2.0, "beta": {"amazon": 2.0}, "pop_recent_days": {"hm": 7}, "max_hist": 50}
"""
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Event, Item
from ..services.catalog_index import CatalogIndex, catalog_signature, get_index
from ..services.search_core import rank_percentile
from .base import Recommender, register
from .popularity import _popular_scored

HISTORY_TYPES = ("purchase", "add_to_cart", "click")
DEFAULTS = {"gamma": 2.0, "beta": {"amazon": 2.0}, "pop_recent_days": {"hm": 7}, "max_hist": 50}


@lru_cache(maxsize=16)
def load_tower(path: str) -> dict:
    """weights.npz: user.0.weight · user.0.bias · user.3.weight · user.3.bias, item.* 같은 구성 (recsys.ume.save_tower)."""
    return dict(np.load(Path(path) / "weights.npz"))


REQUIRED_KEYS = [f"{side}.{i}.{kind}" for side in ("user", "item") for i in (0, 3) for kind in ("weight", "bias")]


def tower_problem(art: str, dim: int | None = None) -> str | None:
    """가중치 폴더 검사. 문제가 없으면 None, 있으면 이유 문자열 (서빙은 그 데이터셋을 빼고 인기로 대체, 등록은 제외 · 활성화 거절).

    텍스트 · 이미지 각각: weights.npz 존재 · 읽기 가능 · 필수 키, 고객 · 상품 타워마다 층 모양
    (0층 (H, D) · bias (H,), 3층 (E, H) · bias (E,)), 두 타워의 입력 D가 같고 상품 임베딩 차원 dim과 같으며 출력 E가 같은지.
    """
    for m in ("text", "image"):
        f = Path(art) / m / "weights.npz"
        if not f.exists():
            return f"{f} 없음"
        try:
            w = load_tower(f"{art}/{m}")
        except Exception as e:                                    # 손상된 npz 등
            return f"{f} 읽기 실패: {type(e).__name__}"
        missing = [k for k in REQUIRED_KEYS if k not in w]
        if missing:
            return f"{f} 키 없음: {missing}"
        shapes = {}
        for side in ("user", "item"):
            w0, b0, w3, b3 = (np.shape(w[f"{side}.{i}.{k}"]) for i, k in ((0, "weight"), (0, "bias"), (3, "weight"), (3, "bias")))
            if len(w0) != 2 or len(w3) != 2 or b0 != (w0[0],) or w3[1] != w0[0] or b3 != (w3[0],):
                return f"{f} {side} 타워 층 모양 불일치: 0층 {w0} · {b0}, 3층 {w3} · {b3}"
            shapes[side] = (w0[1], w3[0])                         # (입력 D, 출력 E)
        if shapes["user"] != shapes["item"]:
            return f"{f} 고객 · 상품 타워 차원 불일치: 고객 {shapes['user']} · 상품 {shapes['item']} (입력, 출력)"
        if dim is not None and shapes["user"][0] != dim:
            return f"{f} 입력 차원 {shapes['user'][0]} ≠ 상품 임베딩 {dim}"
    return None


def mlp(w: dict, side: str, x: np.ndarray) -> np.ndarray:
    """Linear → ReLU → (Dropout: 추론 때 없음) → Linear → L2. x: (d,) 또는 (n, d)."""
    h = np.maximum(x @ w[f"{side}.0.weight"].T + w[f"{side}.0.bias"], 0)
    v = h @ w[f"{side}.3.weight"].T + w[f"{side}.3.bias"]
    return (v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


def _rows_of(ix: CatalogIndex, item_ids: np.ndarray) -> np.ndarray:
    """상품 id → 색인 행 번호 (순서 유지). 색인에 없는 상품은 뺀다."""
    if len(ix) == 0 or len(item_ids) == 0:
        return np.zeros(0, np.int64)
    pos = np.searchsorted(ix.ids, item_ids)
    ok = pos < len(ix.ids)
    ok[ok] = ix.ids[pos[ok]] == item_ids[ok]
    return pos[ok]


def _z(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-6)


_item_cache: dict = {}
_pop_cache: dict = {}


def _item_vectors(ix: CatalogIndex, sel: np.ndarray, art: str):
    """데이터셋 상품 벡터 (텍스트, 이미지). 색인이 바뀌거나(상품 추가) 모델이 바뀌면 다시 계산."""
    key = (id(ix), art)
    if key not in _item_cache:
        if len(_item_cache) > 8:
            _item_cache.clear()
        _item_cache[key] = (mlp(load_tower(f"{art}/text"), "item", ix.text[sel]),
                            mlp(load_tower(f"{art}/image"), "item", ix.image[sel]))
    return _item_cache[key]


def _popularity_z(db: Session, source: str, item_ids: np.ndarray, days: int | None) -> np.ndarray:
    """데이터셋 상품별 인기 점수 (len(item_ids),). 학습 때와 같은 정의: 구매 수(최근 days일) log1p 표준화.
    is_setup=True (초기 이력 적재) 는 공통 통계에서 제외 — 캐시 갱신 판단 · 최근 구매 기준 시각 · 집계 모두.

    캐시는 두 단계로 확인한다 (요청마다 실행되므로 가볍게):
    1) 가장 큰 id 이벤트 한 행(id · 고객 · 상품 · 시각)이 그대로면 바로 캐시 사용 — 기본 키 역순 한 행, 1ms 안팎.
       --reset 재적재로 id가 1부터 다시 매겨져도 그 행의 고객 · 상품 · 시각까지 보므로 잡힌다.
    2) 바뀌었으면 집계에 쓰이는 이벤트(이 데이터셋의 비 setup 구매)의 마지막 행만 다시 본다.
       그것도 그대로면 (초기 이력 · 조회 · 클릭만 추가) 집계를 다시 하지 않는다.
    상품 목록(개수 · 처음 · 끝 id)이나 카탈로그 지문이 바뀌어도 다시 계산한다."""
    non_setup = (Event.is_setup == False)                                               # noqa: E712
    raw = db.execute(select(Event.id, Event.user_id, Event.item_id, Event.created_at)
                     .order_by(Event.id.desc()).limit(1)).first()
    raw = tuple(raw) if raw is not None else None
    catalog = (len(item_ids), int(item_ids[0]) if len(item_ids) else None, int(item_ids[-1]) if len(item_ids) else None)
    scope = (str(db.bind.url), days, catalog_signature(db), catalog)
    cached = _pop_cache.get(source)
    if cached is not None and cached["scope"] == scope and cached["raw"] == raw:
        return cached["z"]
    purchases = (select(Event.id, Event.user_id, Event.item_id, Event.created_at).join(Item, Item.id == Event.item_id)
                 .where(Event.type == "purchase", Item.source == source, non_setup))
    sig = db.execute(purchases.order_by(Event.id.desc()).limit(1)).first()
    sig = tuple(sig) if sig is not None else None
    if cached is not None and cached["scope"] == scope and cached["sig"] == sig:
        cached["raw"] = raw                                     # 집계와 무관한 이벤트만 늘었음
        return cached["z"]
    q = (select(Event.item_id, func.count()).join(Item, Item.id == Event.item_id)
         .where(Event.type == "purchase", Item.source == source, non_setup))
    if days is not None:
        latest = db.scalar(select(func.max(Event.created_at)).join(Item, Item.id == Event.item_id)
                           .where(Event.type == "purchase", Item.source == source, non_setup))
        if latest is not None:
            q = q.where(Event.created_at >= latest - timedelta(days=days))
    counts = dict(db.execute(q.group_by(Event.item_id)).all())
    c = np.log1p(np.array([counts.get(int(i), 0) for i in item_ids], np.float32))
    _pop_cache[source] = {"scope": scope, "raw": raw, "sig": sig, "z": (c - c.mean()) / (c.std() + 1e-6)}
    return _pop_cache[source]["z"]


@register
class UME(Recommender):
    """점수 (`recommend_scored`, 응답 score_kind="model"):
    - 데이터셋 하나: z(텍스트) + z(이미지) + γ · 인기 점수 + β · [선호 브랜드] — 고객별 표준화 점수 단위
    - 데이터셋 여럿: 데이터셋 안 순위 백분위 (0 ~ 1, 1 = 그 데이터셋 1위)
    - 신규 고객 · 모델 없는 데이터셋뿐: 인기 모델 점수(이벤트 가중치 합)
    """
    name = "ume"
    scored_native = True

    def recommend(self, db: Session, user_id: int, k: int) -> list[int]:
        return [i for i, _ in self.recommend_scored(db, user_id, k)]

    def recommend_scored(self, db: Session, user_id: int, k: int) -> list[tuple[int, float]]:
        p = {**DEFAULTS, **self.params}
        dirs: dict = p.get("artifact_dirs", {})
        max_hist = int(p["max_hist"])
        ix = get_index(db)
        dim = ix.text.shape[1] if len(ix) else None

        scores, groups, ids = [], [], []
        for source, art in dirs.items():
            if tower_problem(art, dim):                             # 가중치가 없거나 불완전 → 이 데이터셋은 인기로 대체
                continue
            recent = db.scalars(select(Event.item_id).join(Item, Item.id == Event.item_id)   # 데이터셋별 최근 max_hist개
                                .where(Event.user_id == user_id, Event.type.in_(HISTORY_TYPES), Item.source == source)
                                .order_by(Event.created_at.desc(), Event.id.desc()).limit(max_hist)).all()
            hist = _rows_of(ix, np.asarray(recent, np.int64))       # 임베딩 없는 상품은 빠짐
            if len(hist) == 0:
                continue
            sel = np.flatnonzero(ix.sources == source)
            v_text, v_image = _item_vectors(ix, sel, art)
            u_text = mlp(load_tower(f"{art}/text"), "user", ix.text[hist].mean(0))
            u_image = mlp(load_tower(f"{art}/image"), "user", ix.image[hist].mean(0))
            s = _z(v_text @ u_text) + _z(v_image @ u_image)
            gamma = float(p["gamma"])
            if gamma:
                s = s + gamma * _popularity_z(db, source, ix.ids[sel], (p["pop_recent_days"] or {}).get(source))
            beta = (p["beta"] or {}).get(source, 0.0)
            if beta:
                b = ix.brands[hist]
                b = b[b >= 0]
                if len(b):
                    top = np.bincount(b).argmax()
                    s = s + float(beta) * (ix.brands[sel] == top)
            scores.append(s)
            groups.append(np.full(len(sel), source, object))
            ids.append(ix.ids[sel])
        if not scores:
            self.fallback_reason = "popular"                        # 응답 reason: 개인화가 아니라 인기
            self.score_basis = "event_weight_sum"
            return _popular_scored(db, k)                           # 신규 고객 · 모델 없는 데이터셋
        s, g, all_ids = np.concatenate(scores), np.concatenate(groups), np.concatenate(ids)
        self.score_basis = "ume_zscore"
        if len(scores) > 1:                                         # 여러 데이터셋: 카탈로그 비율대로 섞기
            s = rank_percentile(s[None, :], g)[0]
            self.score_basis = "rank_percentile"
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]                        # 상위 k만 정렬 (상품 8.7만 기준 p95 약 4ms)
        top = top[np.argsort(-s[top], kind="stable")]
        return [(int(i), round(float(v), 4)) for i, v in zip(all_ids[top], s[top])]
