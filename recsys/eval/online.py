"""Online event-window metrics; no database access or implicit event filtering.

The caller fixes the event type, time window, cohort and transport deduplication.
Do not confuse the customer mean below with offline dataset loyalty medians.
"""
from collections import Counter
from math import fsum
from typing import Hashable, Mapping, Sequence

ID = Hashable


def _mean(values):
    return fsum(values) / len(values) if values else None


def dominant_brand_share(
    user_brand_events: Mapping[ID, Sequence[ID | None]], *, event_type: str,
) -> dict:
    """Mean of each customer's most frequent known brand share.

    The caller supplies events of exactly one declared type/window. Repeated
    genuine events count repeatedly; deduplicate transport retries upstream.
    Unknown brands are excluded and counted, not pooled into a fake brand.
    A higher value is concentration, not automatically better recommendations.
    """
    if event_type not in {"view", "click", "purchase"}:
        raise ValueError("event_type must be view, click or purchase")
    per_user = []
    excluded = 0
    missing_total = 0
    for user, brands in user_brand_events.items():
        counts = Counter(b for b in brands if b is not None)
        missing = sum(b is None for b in brands)
        missing_total += missing
        n = sum(counts.values())
        if not n:
            excluded += 1
            continue
        per_user.append({
            "user_id": user, "value": max(counts.values()) / n,
            "n_events": n, "n_unknown_brand_events": missing,
        })
    return {
        "event_type": event_type,
        "value": _mean([u["value"] for u in per_user]),
        "n_users": len(per_user),
        "n_excluded_no_known_brand_events": excluded,
        "n_unknown_brand_events": missing_total,
        "per_user": per_user,
    }
