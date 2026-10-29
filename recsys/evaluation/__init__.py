"""Pure evaluation functions; no database, model, or API dependency."""

from .metrics import (
    catalog_coverage,
    dominant_brand_share,
    evaluate_report,
    intralist_diversity,
    ranking_metrics,
)

__all__ = [
    "ranking_metrics", "catalog_coverage", "intralist_diversity",
    "dominant_brand_share", "evaluate_report",
]
