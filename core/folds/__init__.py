"""Folds package (Phase r1 §2 restructure).

* ``walk_forward`` — fold geometry (former ``core/folds.py``).
* ``oof``          — OOF population rules and binary metrics (former
                     ``core/oof.py``).
"""

from core.folds.walk_forward import (  # noqa: F401
    Fold,
    FoldError,
    WindowFold,
    build_folds,
    fold_summary,
    fold_table,
    make_folds,
    train_test_boundary,
    walk_forward_splits,
)
from core.folds.oof import (  # noqa: F401
    OOF_REQUIRED_COLUMNS,
    OOFError,
    OOFValidationReport,
    brier_score,
    log_loss,
    oof_metrics,
    roc_auc,
    validate_oof_rows,
)

__all__ = [
    "Fold", "FoldError", "WindowFold", "build_folds", "fold_summary",
    "fold_table", "make_folds", "train_test_boundary", "walk_forward_splits",
    "OOF_REQUIRED_COLUMNS", "OOFError", "OOFValidationReport", "brier_score",
    "log_loss", "oof_metrics", "roc_auc", "validate_oof_rows",
]
