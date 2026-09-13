"""Evaluation package (Phase r1 §2 restructure).

* ``probabilistic_metrics`` — pooled/sealed metric helpers, ECE, stability
                              (former ``core/optimization/metrics.py``).
"""

from core.evaluation.probabilistic_metrics import (  # noqa: F401
    auc,
    brier,
    logloss,
    period_stability,
    pooled_metrics,
    regression_metrics,
    regression_period_stability,
)

__all__ = [
    "auc", "brier", "logloss", "period_stability", "pooled_metrics",
    "regression_metrics", "regression_period_stability",
]
