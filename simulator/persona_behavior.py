"""서버 실행(server_run)용 페르소나 행동 (A1 후속).

`server_run`의 기본 행동은 상품과 무관한 고정 확률이다. 이 모듈은 `engine.py`의 페르소나 v0.3 규칙
(선호 브랜드 · 소분류 · 예산 · 재구매 · 새로움 · 인기 → 선호 점수 → 클릭 확률, 장바구니 · 구매는 조건부)을
서버 추천 응답에 그대로 적용한다. 로컬 파일럿(catalog_pilot)과 같은 고객 · 같은 확률식이라 결과를 비교할 수 있다.

필요한 것
  catalog_dir : 적재에 쓴 카탈로그 (items.parquet · events.parquet, split=train 이력) — 고객 선호 · 초기 이력 생성
  snapshot    : 서버 DB 상품 id 표 (backend `python -m scripts.export_simulator_items --out ...`)
                초기 이력(is_setup 구매)을 서버 상품 id로 보내는 데만 쓴다. 추천 응답은 source · source_item_id로 바로 찾는다.

결정론: 고객 · 초기 이력은 (seed, customer_id), 행동 추첨은 (seed, customer_id, 세션, 순위, 행동)으로만 정해진다.
모델 · 점수 · 실험 그룹은 행동에 쓰지 않는다 (personas.yaml behavior.use_* = false). 같은 고객이 같은 순위에서 같은
상품을 보면 어느 그룹이든 같은 결과 (공통 난수) — 그룹 차이는 "무엇을 보여 줬는가"에서만 나온다.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import derived_seed, item_key, prepare
from .engine import click_probability, item_features, make_customers
from .server import map_items

ACTIONS = ("click", "add_to_cart", "purchase")


class PersonaBehavior:
    def __init__(self, catalog_dir, snapshot_path, customers: int, seed: int = 42, config=None):
        catalog_dir = Path(catalog_dir)
        items = pd.read_parquet(catalog_dir / "items.parquet")
        events = pd.read_parquet(catalog_dir / "events.parquet")
        self.config = config or load_config(Path(__file__).with_name("personas.yaml"))
        self.config["pilot"].update(planned_customers=customers, seed=seed)
        self.seed = seed
        data = prepare(items, events)
        people, skipped = make_customers(data, self.config)
        if len(people) != customers:
            raise ValueError(f"고객 {customers}명 중 {len(people)}명만 만들 수 있음: {skipped[:3]}")
        # 페르소나별로 묶여 나오므로 섞는다 — 중간에 멈춰도 페르소나가 고르게 들어가게
        order = np.random.default_rng(derived_seed(seed, "server_order")).permutation(len(people))
        self.customers = [people[i] for i in order]
        self.personas = {p["id"]: p for p in self.config["personas"]}
        self.products = {row["item_key"]: row for row in data.items.to_dict("records")}
        # 추천 응답 (source, source_item_id) → 로컬 item_key
        self.by_external = {(r.source, str(r.source_item_id)): item_key(r.source, r.item_id) for r in items.itertuples()}
        snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        mapping, missing = map_items(items, snapshot)
        self.key_to_db = mapping
        need = {k for c in self.customers for k in c["initial_purchase_keys"]}
        if need - set(mapping):
            raise ValueError(f"초기 이력 상품 {len(need - set(mapping))}개가 서버 DB에 없음 (카탈로그 · 스냅샷이 같은 적재인지 확인)")
        self.stats = Counter()

    def persona(self, ci: int) -> str:
        return self.customers[ci]["persona"]

    def setup_purchases(self, ci: int) -> list[int]:
        """초기 이력 (is_setup=True 구매)의 서버 상품 id. 추천 개인화의 출발점, 집계 · 인기 · 재학습 카운터에서는 빠진다."""
        return [int(self.key_to_db[k]) for k in self.customers[ci]["initial_purchase_keys"]]

    def actions(self, ci: int, si: int, recommendations) -> dict[int, list[str]]:
        """recommendations: [(position, source, source_item_id)] → {position: [click, add_to_cart, purchase 중 일어난 것]}.
        카탈로그에 없거나 특성이 빠진 상품(가격 · 브랜드 · 소분류 없음)은 보기만 한다 (stats['unknown_items'])."""
        c = self.customers[ci]
        spec = self.personas[c["persona"]]
        overrides = spec.get("behavior_overrides", {})
        out = {}
        for position, source, source_item_id in recommendations:
            self.stats[f"{c['persona']}:view"] += 1
            key = self.by_external.get((source, str(source_item_id)))
            item = self.products.get(key)
            if item is None:
                self.stats["unknown_items"] += 1
                out[position] = []
                continue
            f = item_features(c, item, self.products)
            score = sum(w * f[name] for name, w in spec["preference_weights"].items())
            probs = {"click": click_probability(score, self.config, spec),
                     "add_to_cart": overrides.get("cart_probability", self.config["behavior"]["add_to_cart"]["probability"]),
                     "purchase": overrides.get("purchase_probability", self.config["behavior"]["purchase"]["probability"])}
            done = []
            for action in ACTIONS:                                  # click → add_to_cart → purchase (앞 단계가 있어야 다음)
                draw = np.random.default_rng(derived_seed(self.seed, "behavior", c["customer_id"], f"s{si:04d}", position, action)).random()
                if draw >= probs[action]:
                    break
                done.append(action)
                self.stats[f"{c['persona']}:{action}"] += 1
            out[position] = done
        return out

    def summary(self) -> dict:
        personas = sorted({k.split(":")[0] for k in self.stats if ":" in k})
        rows = {}
        for p in personas:
            v = self.stats[f"{p}:view"]
            rows[p] = {"views": v, **{a: self.stats[f"{p}:{a}"] for a in ACTIONS},
                       "ctr": round(self.stats[f"{p}:click"] / v, 4) if v else None}
        return {"by_persona": rows, "unknown_items": self.stats["unknown_items"],
                "customers_by_persona": dict(Counter(c["persona"] for c in self.customers))}
