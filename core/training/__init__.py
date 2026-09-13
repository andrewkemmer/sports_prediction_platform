"""Training package (Phase r1 §2 restructure).

* ``blending`` — shared ensemble blend + adaptive weight selection
                 (deduplicated from the per-sport training modules; the
                 sport moneyline trainers rebind to it in r1/r2).
"""

from core.training.blending import (  # noqa: F401
    CLIP,
    DEFAULT_ENSEMBLE_PRIOR,
    adaptive_weights,
    blend_member_predictions,
    blend_probabilities,
)

__all__ = [
    "CLIP", "DEFAULT_ENSEMBLE_PRIOR", "adaptive_weights",
    "blend_member_predictions", "blend_probabilities",
]
