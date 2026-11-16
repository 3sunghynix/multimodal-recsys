"""재정렬 마지막 단계 — MAB 탐색 슬롯 · 신규 상품 노출 · 같은 소분류 연속 제한 (명세 4.3 Stage 3).

추천 모델(인기 · UME 등)이 고른 목록을 받아 두 가지를 한다.

1. 탐색 슬롯 (model_versions.params["explore"]가 있을 때만)
   - 후보: 신규 상품(registered_at이 new_days일 안) + 고객 선호 소분류 상품. 이미 목록에 있거나 고객이
     산 상품은 뺀다. 그룹마다 고객별로 섞고(동점 상품이 돌아가며 노출), 신규 상품은 선호 소분류 것을 앞에 둔다.
   - 슬롯 나누기: 2개면 1개는 신규 상품(노출 보장), 1개는 선호 소분류 상품(클릭 가능성 있는 탐색).
     신규만 몰면 고객 취향과 무관한 상품만 보여 탐색 CTR이 0이 된다 (120명 파일럿에서 확인).
   - 선택: 상품 하나 = 팔(arm) 하나. 보상 = 노출(view) 대비 클릭(click), 비 setup 이벤트로 센다.
     UCB1(기본): CTR + c·√(ln N / n). 노출 0인 상품은 무한대 → 신규 상품이 먼저 노출된다.
     ε-greedy: 확률 ε로 후보 중 무작위, 아니면 CTR 최고.
   - 자리: k ≥ 10이면 2개, 아니면 1개를 positions(기본 4 · 9번째)에 넣고 원래 목록은 그만큼 뒤에서 뺀다.
     응답에서 is_exploration=True, reason="mab_exploration".
2. 같은 소분류 연속 제한 (항상): 같은 소분류가 max_run(기본 2)개를 넘어 이어지면 뒤에서 다른 소분류
   상품을 당겨 와 끊는다. 다른 소분류가 없으면 그대로 둔다.

순수 함수(ucb_index · choose · place · break_runs)는 DB 없이 테스트한다. DB를 쓰는 부분은 함수 안에서 import.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

DEFAULT = {
    "algorithm": "ucb",        # "ucb" | "epsilon_greedy"
    "c": 0.3,                  # UCB 탐색 강도. 보상(CTR)이 0.1 안팎이라 UCB1 원형의 c=1은 보너스가 CTR 차이를 덮어
                               # 거의 돌아가며 고르기가 된다 (모의 실험: 2,000회 뒤 상위 5개 비중 c=1 → 19%, c=0.3 → 63%)
    "epsilon": 0.1,            # ε-greedy 무작위 비율
    "slots": 2,                # k ≥ min_k_two_slots일 때 탐색 슬롯 수 (명세: 상위 N개 중 1 ~ 2개)
    "min_k_two_slots": 10,     # 이보다 작은 k에서는 1개
    "positions": [3, 8],       # 0부터 센 자리 (4 · 9번째). k를 넘으면 마지막 자리로
    "new_days": 7,             # 신규 상품 기준 (명세: 등록 7일 이내)
    "new_slots": 1,            # 탐색 슬롯 중 신규 상품 몫 (나머지는 선호 소분류 상품). 슬롯이 1개면 신규 우선
    "pool": 200,               # 탐색 후보 최대 수
    "preferred_categories": 3, # 고객 이력에서 쓸 소분류 수
}
MAX_RUN = 2                    # 같은 소분류 최대 연속 개수 (명세: 3개 이상 연속 금지)
REASON = "mab_exploration"


@dataclass
class Rec:
    item_id: int
    score: float
    is_exploration: bool = False


def config(params: dict | None) -> dict | None:
    """model_versions.params["explore"] → 설정 dict. 없거나 False면 None (탐색 끔)."""
    raw = (params or {}).get("explore")
    if not raw:
        return None
    return {**DEFAULT, **(raw if isinstance(raw, dict) else {})}


def n_slots(k: int, cfg: dict) -> int:
    if k <= 1:
        return 0
    return min(int(cfg["slots"]), k - 1) if k >= int(cfg["min_k_two_slots"]) else 1


# ---------------------------------------------------------------- 밴딧 (순수)

def ucb_index(views: int, clicks: int, total_views: int, c: float) -> float:
    """UCB1. 노출이 없으면 무한대 (먼저 시도). clicks > views(노출 없이 클릭)는 views로 자른다."""
    if views <= 0:
        return math.inf
    ctr = min(clicks, views) / views
    return ctr + c * math.sqrt(math.log(max(total_views, 2)) / views)


def choose(candidates: list[int], stats: dict[int, tuple[int, int]], n: int, cfg: dict,
           rng: np.random.Generator, new_ids: set[int] | None = None) -> list[int]:
    """후보 중 n개를 고른다. stats: {item_id: (views, clicks)}. 같은 점수면 신규 상품 → 후보 순서.
    반환 순서 = 고른 순서 (점수 높은 것 먼저)."""
    if n <= 0 or not candidates:
        return []
    new_ids = new_ids or set()
    vc = [stats.get(i, (0, 0)) for i in candidates]
    if cfg["algorithm"] == "epsilon_greedy":
        picked, rest = [], list(range(len(candidates)))
        for _ in range(min(n, len(candidates))):
            if rng.random() < float(cfg["epsilon"]):
                j = rest[int(rng.integers(len(rest)))]
            else:   # CTR 최고 (노출 없는 상품은 0으로 본다 — 탐색은 ε가 담당)
                j = max(rest, key=lambda r: (min(vc[r][1], vc[r][0]) / vc[r][0] if vc[r][0] else 0.0,
                                             candidates[r] in new_ids, -r))
            picked.append(j)
            rest.remove(j)
        return [candidates[j] for j in picked]
    if cfg["algorithm"] != "ucb":
        raise ValueError(f"알 수 없는 탐색 알고리즘: {cfg['algorithm']}")
    total = sum(v for v, _ in vc)
    scores = [ucb_index(v, c, total, float(cfg["c"])) for v, c in vc]
    order = sorted(range(len(candidates)), key=lambda r: (-scores[r], candidates[r] not in new_ids, r))
    return [candidates[j] for j in order[:n]]


# ---------------------------------------------------------------- 배치 (순수)

def place(exploit: list[Rec], explore_ids: list[int], positions: list[int], k: int) -> list[Rec]:
    """원래 목록에서 앞 k − len(explore)개를 남기고 탐색 상품을 positions 자리에 넣는다. 결과 길이 ≤ k."""
    explore_ids = [i for i in explore_ids if i not in {r.item_id for r in exploit[:k]}][:k]
    out = [r for r in exploit if r.item_id not in set(explore_ids)][:k - len(explore_ids)]
    spots = sorted({min(int(p), k - 1) for p in positions})
    while len(spots) < len(explore_ids):              # 자리 수가 모자라면 뒤에서부터 채운다
        spots = sorted(set(spots) | {k - 1 - len(spots)})
    for spot, item_id in zip(spots[:len(explore_ids)], explore_ids):
        out.insert(min(spot, len(out)), Rec(item_id, 0.0, True))
    return out


def break_runs(recs: list[Rec], cat_of: dict[int, str], max_run: int = MAX_RUN) -> list[Rec]:
    """같은 소분류가 max_run개를 넘어 이어지면, 뒤쪽에서 그 소분류가 아닌 가장 가까운 상품을 당겨 와 끊는다.
    소분류가 빈 문자열이면 제한하지 않는다. 다른 소분류가 남아 있지 않으면 그대로 둔다."""
    out = list(recs)
    for i in range(max_run, len(out)):
        c = cat_of.get(out[i].item_id, "")
        if not c or any(cat_of.get(out[j].item_id, "") != c for j in range(i - max_run, i)):
            continue
        swap = next((j for j in range(i + 1, len(out)) if cat_of.get(out[j].item_id, "") != c), None)
        if swap is not None:
            out.insert(i, out.pop(swap))
    return out


def max_run_length(recs: list[Rec], cat_of: dict[int, str]) -> int:
    best = run = 0
    prev = None
    for r in recs:
        c = cat_of.get(r.item_id, "")
        run = run + 1 if c and c == prev else (1 if c else 0)
        prev = c
        best = max(best, run)
    return best


# ---------------------------------------------------------------- DB

def leaf(category: str | None) -> str:
    return (category or "").split(">")[-1].strip()


def categories(db, ids: list[int]) -> dict[int, str]:
    from sqlalchemy import select

    from ..models import Item
    if not ids:
        return {}
    return {i: leaf(c) for i, c in db.execute(select(Item.id, Item.category).where(Item.id.in_(ids)))}


def item_stats(db, ids: list[int]) -> dict[int, tuple[int, int]]:
    """상품별 (노출, 클릭) — 비 setup 이벤트만."""
    from sqlalchemy import func, select

    from ..models import Event
    if not ids:
        return {}
    out: dict[int, list[int]] = {}
    rows = db.execute(select(Event.item_id, Event.type, func.count(Event.id))
                      .where(Event.item_id.in_(ids), Event.type.in_(("view", "click")), Event.is_setup == False)  # noqa: E712
                      .group_by(Event.item_id, Event.type))
    for item_id, typ, n in rows:
        out.setdefault(int(item_id), [0, 0])[0 if typ == "view" else 1] += int(n)
    return {i: (v, c) for i, (v, c) in out.items()}


def candidate_pool(db, user_id: int, exclude: set[int], cfg: dict, now: datetime | None = None,
                   rng: np.random.Generator | None = None) -> tuple[list[int], set[int]]:
    """(후보 id 목록, 그중 신규 상품 id). 신규 상품(최대 pool의 절반, 선호 소분류 것 먼저)을 앞에, 남는 자리는
    선호 소분류 상품 무작위로 채운다. 각 그룹 안 순서는 rng로 섞는다 (UCB 동점이면 이 순서로 고른다)."""
    from collections import Counter

    from sqlalchemy import select

    from ..models import Event, Item
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)       # DB 시각은 naive UTC
    rng = rng or np.random.default_rng(user_id)
    pool = int(cfg["pool"])
    bought = set(db.scalars(select(Event.item_id).where(Event.user_id == user_id, Event.type == "purchase",
                                                        Event.item_id.is_not(None))))
    skip = exclude | bought
    since = now - timedelta(days=int(cfg["new_days"]))
    new_rows = [(i, c) for i, c in db.execute(
        select(Item.id, Item.category).where(Item.registered_at.is_not(None), Item.registered_at >= since,
                                             Item.registered_at <= now)
        .order_by(Item.registered_at.desc(), Item.id.desc()).limit(pool * 2)) if i not in skip]
    # 선호 소분류: 고객 이벤트(구매 · 장바구니 · 클릭, 초기 이력 포함) 상품의 소분류 상위 n개
    hist = db.execute(select(Item.category).join(Event, Event.item_id == Item.id)
                      .where(Event.user_id == user_id, Event.type.in_(("purchase", "add_to_cart", "click")))
                      .order_by(Event.id.desc()).limit(200)).scalars().all()
    top = [c for c, _ in Counter(leaf(c) for c in hist if leaf(c)).most_common(int(cfg["preferred_categories"]))]
    # 신규: 선호 소분류 것 먼저, 그룹 안에서는 고객별 무작위 (동점일 때 같은 상품만 반복 노출되지 않게)
    fav_new = [int(i) for i, c in new_rows if leaf(c) in top]
    other_new = [int(i) for i, c in new_rows if leaf(c) not in top]
    # 후보의 절반까지만 신규 (나머지는 선호 소분류 몫 — 신규 상품이 많아도 그쪽 후보가 비지 않게)
    new = (list(rng.permutation(fav_new)) + list(rng.permutation(other_new)))[:max(pool // 2, 1)]
    new = [int(i) for i in new]
    rest: list[int] = []
    if top and len(new) < pool:
        ids = _ids_in_leaves(db, top)
        taken = skip | set(new)
        ids = np.array([i for i in ids if i not in taken], np.int64)
        if len(ids):
            rest = rng.choice(ids, size=min(pool - len(new), len(ids)), replace=False).tolist()
    return new + [int(i) for i in rest], set(new)


def _ids_in_leaves(db, leaves: list[str]) -> list[int]:
    """소분류가 leaves 중 하나인 상품 id. 메모리 색인(catalog_index, 소분류 포함)이 있으면 그걸 쓰고,
    없으면(임베딩 없는 테스트 샘플 등) DB에서 찾는다."""
    from sqlalchemy import or_, select

    from ..models import Item
    from .catalog_index import get_index
    ix = get_index(db)
    if len(ix) and ix.cats is not None:
        return ix.ids[np.isin(ix.cats, leaves)].tolist()
    rows = db.execute(select(Item.id, Item.category).where(or_(*[Item.category.like(f"%{c}") for c in leaves])))
    return [int(i) for i, c in rows if leaf(c) in leaves]


def apply(db, user_id: int, scored: list[tuple[int, float]], k: int, params: dict | None,
          stats_fn=None, now: datetime | None = None, rng: np.random.Generator | None = None) -> list[Rec]:
    """추천 모델 결과 → 탐색 슬롯(설정 시) + 소분류 연속 제한. stats_fn(ids) → {id: (views, clicks)}:
    기본은 DB 집계, 파일럿은 메모리 집계를 넘긴다."""
    recs = [Rec(int(i), float(s)) for i, s in scored][:k]
    cfg = config(params)
    if cfg is not None:
        n = n_slots(min(k, len(recs)), cfg) if recs else 0
        rng = rng or np.random.default_rng([user_id, int((now or datetime.now(timezone.utc)).timestamp()) // 60])
        cands, new_ids = candidate_pool(db, user_id, {r.item_id for r in recs}, cfg, now, rng)
        stats = (stats_fn or (lambda ids: item_stats(db, ids)))(cands)
        new_c = [i for i in cands if i in new_ids]
        old_c = [i for i in cands if i not in new_ids]
        n_new = min(int(cfg["new_slots"]), n) if n >= 2 else n
        if new_c and old_c and n >= 2:       # 신규 몫 + 선호 소분류 몫 (한쪽이 모자라면 다른 쪽에서 채움)
            picks = choose(new_c, stats, n_new, cfg, rng, new_ids)
            picks += choose(old_c, stats, n - len(picks), cfg, rng, new_ids)
        else:
            picks = choose(cands, stats, n, cfg, rng, new_ids)
        recs = place(recs, picks, cfg["positions"], min(k, len(recs)))
    return break_runs(recs, categories(db, [r.item_id for r in recs]))
