"""Slow, explicit reference calculations for recsys.eval.metrics cross-checks.

Not a second production evaluator. Model experiments use recsys.run_eval and
recsys.eval.metrics. This module intentionally keeps readable independent formulas
for hand-calculated examples and tests. See docs/project/지표정의.md.
Caller-selected IDs, cohorts, positive targets and fixed vector versions apply.
Empty cohorts, short lists and missing vectors have explicit reference policies;
compare fast and reference implementations only on their common valid domain.
"""

from itertools import combinations
from math import fsum, isfinite, log2, sqrt
from typing import Collection, Hashable, Mapping, Sequence

from .online import dominant_brand_share

ID = Hashable
Rankings = Mapping[ID, Sequence[ID]]


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def _unique(values, label):
    result = list(values)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate {label}")
    return result


def _ranked(items, k):
    # Repeated recommendations cannot earn repeated credit, even beyond K.
    return _unique(items, "ranked item IDs")[:k]


def _mean(values):
    return fsum(values) / len(values) if values else None


def ranking_metrics(
    rankings: Rankings,
    targets: Mapping[ID, Collection[ID]],
    *,
    eligible_users: Sequence[ID],
    k: int = 12,
) -> dict:
    """Macro-average binary Recall, NDCG and AP over explicit eligible users.

    Targets are deduplicated per customer. A missing/empty recommendation list
    earns zero for a customer with relevant targets. A customer without targets
    is excluded from accuracy averages and is counted explicitly.
    """
    _check_k(k)
    users = _unique(eligible_users, "eligible user IDs")
    per_user = []
    no_targets = 0
    no_recommendations = 0
    for user in users:
        ranked = _ranked(rankings.get(user, ()), k)
        relevant = set(targets.get(user, ()))
        if not relevant:
            no_targets += 1
            continue
        if not ranked:
            no_recommendations += 1
        hits = 0
        dcg = 0.0
        precision_sum = 0.0
        for position, item in enumerate(ranked, start=1):
            if item in relevant:
                hits += 1
                dcg += 1.0 / log2(position + 1)
                precision_sum += hits / position
        ideal = fsum(1.0 / log2(j + 1) for j in range(1, min(len(relevant), k) + 1))
        per_user.append({
            "user_id": user,
            "relevant_items": len(relevant),
            "returned_items": len(ranked),
            "hits": hits,
            "recall": hits / len(relevant),
            "ndcg": dcg / ideal,
            "ap": precision_sum / min(len(relevant), k),
        })
    return {
        "k": k,
        "recall": _mean([u["recall"] for u in per_user]),
        "ndcg": _mean([u["ndcg"] for u in per_user]),
        "map": _mean([u["ap"] for u in per_user]),
        "n_eligible_users": len(users),
        "n_evaluated_users": len(per_user),
        "n_excluded_no_targets": no_targets,
        "n_evaluated_without_recommendations": no_recommendations,
        "per_user": per_user,
    }


def catalog_coverage(rankings: Rankings, catalog: Collection[ID], *, k: int = 12) -> dict:
    """Unique recommended items / unique items in the fixed eligible catalog."""
    _check_k(k)
    catalog_ids = set(catalog)
    recommended = set()
    for items in rankings.values():
        recommended.update(_ranked(items, k))
    if recommended - catalog_ids:
        raise ValueError("recommendations contain items outside the evaluation catalog")
    return {
        "value": len(recommended) / len(catalog_ids) if catalog_ids else None,
        "n_recommended_items": len(recommended),
        "n_catalog_items": len(catalog_ids),
        "n_users": len(rankings),
    }


def intralist_diversity(
    rankings: Rankings, item_vectors: Mapping[ID, Sequence[float]], *, k: int = 12,
) -> dict:
    """Macro-average within-list mean cosine distance (range 0..2).

    Use one fixed content-vector version for every compared model. Missing,
    zero, nonfinite and dimension-mismatched vectors are errors. Lists with
    fewer than two items have undefined diversity and are counted separately.
    Reference implementation is O(users * K² * dimension), not a speed benchmark.
    """
    _check_k(k)
    normalized = {}
    dimension = None
    per_user = []
    short_lists = 0

    def vector(item):
        nonlocal dimension
        if item not in normalized:
            if item not in item_vectors:
                raise ValueError(f"missing content vector for item {item!r}")
            v = tuple(float(x) for x in item_vectors[item])
            if not v or not all(isfinite(x) for x in v):
                raise ValueError("content vectors must be nonempty and finite")
            if dimension is None:
                dimension = len(v)
            if len(v) != dimension:
                raise ValueError("content vector dimensions differ")
            norm = sqrt(fsum(x*x for x in v))
            if not isfinite(norm) or norm == 0:
                raise ValueError("content vector norm must be finite and nonzero")
            normalized[item] = tuple(x / norm for x in v)
        return normalized[item]

    for user, items in rankings.items():
        ranked = _ranked(items, k)
        # Validate even a single returned item's vector; do not hide missing data.
        vectors = [vector(item) for item in ranked]
        if len(vectors) < 2:
            short_lists += 1
            continue
        distances = [
            1.0 - max(-1.0, min(1.0, fsum(x*y for x, y in zip(a, b))))
            for a, b in combinations(vectors, 2)
        ]
        per_user.append({"user_id": user, "value": _mean(distances), "n_pairs": len(distances)})
    return {
        "value": _mean([u["value"] for u in per_user]),
        "n_users": len(per_user),
        "n_excluded_short_lists": short_lists,
        "vector_dimension": dimension,
        "per_user": per_user,
    }


def evaluate_report(
    *, rankings: Rankings, targets: Mapping[ID, Collection[ID]],
    eligible_users: Sequence[ID], catalog: Collection[ID], k: int = 12,
    item_vectors: Mapping[ID, Sequence[float]] | None = None,
    user_brand_events: Mapping[ID, Sequence[ID | None]] | None = None,
    brand_event_type: str = "purchase", metadata: dict | None = None,
) -> dict:
    """Compose a reference-example report; not an operational evaluation entry point.

    No implicit feedback/split filtering.

    Accuracy excludes users with no relevant targets. Coverage/diversity use all
    supplied eligible customers, including those excluded from accuracy. Every
    denominator is reported. Optional components are None when not supplied.
    """
    _check_k(k)
    users = _unique(eligible_users, "eligible user IDs")
    selected = {u: list(rankings.get(u, ())) for u in users}
    catalog_ids = set(catalog)
    for user in users:
        if set(targets.get(user, ())) - catalog_ids:
            raise ValueError("targets contain items outside the evaluation catalog")
    accuracy = ranking_metrics(selected, targets, eligible_users=users, k=k)
    coverage = catalog_coverage(selected, catalog_ids, k=k)
    diversity = intralist_diversity(selected, item_vectors, k=k) if item_vectors is not None else None
    brands = dominant_brand_share(
        {u: user_brand_events.get(u, ()) for u in users}, event_type=brand_event_type,
    ) if user_brand_events is not None else None
    return {
        "definition_version": "0.1",
        "metadata": dict(metadata or {}),
        "ranking": accuracy, "catalog_coverage": coverage,
        "intralist_diversity": diversity, "dominant_brand_share": brands,
    }
