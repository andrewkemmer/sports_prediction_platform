"""Frozen optimizer bounds and declarative run config.

Bounds (Phase 7 spec — frozen, not tunable per run):

* <= 1,000 candidate features per sport/model,
* 5–10 training-horizon/lookback candidates,
* 5–8 model candidates,
* pairwise interactions only,
* fixed platform fold geometry / OOF rules from study.yaml (never
  optimized over),
* bounded wall clock (``max_seconds``),
* deterministic RNG seed.

Configuration is a declarative mapping (study.yaml extension block or a
plain dict) — never generated code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

MAX_CANDIDATE_FEATURES = 1_000
MIN_HORIZON_CANDIDATES = 5
MAX_HORIZON_CANDIDATES = 10
MIN_MODEL_CANDIDATES = 5
MAX_MODEL_CANDIDATES = 8
MAX_INTERACTION_ORDER = 2  # pairwise only
DEFAULT_SEED = 42
DEFAULT_MAX_SECONDS = 600.0
# Frozen per-scope sweep cap: the bounded walk-forward sweep evaluates at
# most this many candidate feature sets per scope. The runner truncates
# the deterministic candidate priority order to a stable prefix (every
# drop is logged), so the run stays inside its wall-clock budget AND
# remains fully reproducible — unlike a time-based cutoff alone.
DEFAULT_SWEEP_CAP = 60
# Parallel worker count for the candidate-set sweep (0/1 = sequential).
# Pure execution detail: each candidate set's walk-forward OOF is
# independent and returned in deterministic order, so results are
# bit-identical for any n_jobs.


@dataclass(frozen=True)
class Bounds:
    """The frozen per-run candidate bounds (validated)."""

    max_candidate_features: int = MAX_CANDIDATE_FEATURES
    horizon_choices: tuple[int, ...] = (3, 5, 8, 10, 12, 15)
    model_names: tuple[str, ...] = (
        "xgboost", "lightgbm", "logistic", "randomforest", "mlp",
    )
    interaction_order: int = MAX_INTERACTION_ORDER
    seed: int = DEFAULT_SEED
    max_seconds: float = DEFAULT_MAX_SECONDS
    sweep_cap: int = DEFAULT_SWEEP_CAP
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.max_candidate_features < 1 or \
                self.max_candidate_features > MAX_CANDIDATE_FEATURES:
            raise ValueError(
                f"max_candidate_features must be in [1, {MAX_CANDIDATE_FEATURES}]")
        if not (MIN_HORIZON_CANDIDATES <= len(self.horizon_choices)
                <= MAX_HORIZON_CANDIDATES):
            raise ValueError(
                f"need {MIN_HORIZON_CANDIDATES}–{MAX_HORIZON_CANDIDATES} "
                "horizon/lookback candidates")
        if any(h < 1 for h in self.horizon_choices):
            raise ValueError("horizon windows must be >= 1")
        if not (MIN_MODEL_CANDIDATES <= len(self.model_names)
                <= MAX_MODEL_CANDIDATES):
            raise ValueError(
                f"need {MIN_MODEL_CANDIDATES}–{MAX_MODEL_CANDIDATES} model "
                "candidates")
        if self.interaction_order != MAX_INTERACTION_ORDER:
            raise ValueError("interaction_order is frozen to pairwise (2)")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")
        if self.sweep_cap < 1:
            raise ValueError("sweep_cap must be >= 1")
        if self.n_jobs < 0:
            raise ValueError("n_jobs must be >= 0 (0/1 = sequential)")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "Bounds":
        raw = raw or {}
        return cls(
            max_candidate_features=int(
                raw.get("max_candidate_features", MAX_CANDIDATE_FEATURES)),
            horizon_choices=tuple(raw.get("horizon_choices",
                                          (3, 5, 8, 10, 12, 15))),
            model_names=tuple(raw.get("model_names",
                                      ("xgboost", "lightgbm", "logistic",
                                       "randomforest", "mlp"))),
            seed=int(raw.get("seed", DEFAULT_SEED)),
            max_seconds=float(raw.get("max_seconds", DEFAULT_MAX_SECONDS)),
            sweep_cap=int(raw.get("sweep_cap", DEFAULT_SWEEP_CAP)),
            n_jobs=int(raw.get("n_jobs", 1)),
        )


@dataclass(frozen=True)
class ModelScope:
    """One model's independent optimization scope.

    The moneyline (binary win probability) and the market/run-engine
    (score/totals/spread distribution) models each carry their own raw
    field pool and feature list — one is NEVER inferred from the other.
    """

    sport: str
    model: str                       # "moneyline" | "market"
    raw_fields: tuple[str, ...]      # raw fields from the durable catalog
    prod_features: tuple[str, ...]   # incumbent served feature list
    target: str                      # column optimized against
    horizon_field: str = ""          # column the horizon candidates shift

    def __post_init__(self) -> None:
        if self.model not in ("moneyline", "market"):
            raise ValueError("model must be 'moneyline' or 'market'")
        if not self.raw_fields:
            raise ValueError(f"{self.sport}/{self.model}: empty raw pool")
        if not self.prod_features:
            raise ValueError(f"{self.sport}/{self.model}: empty incumbent list")


@dataclass(frozen=True)
class OptimizationConfig:
    """A declarative optimization run configuration."""

    bounds: Bounds
    scopes: tuple[ModelScope, ...] = field(default_factory=tuple)
    coverage_min: float = 0.90            # coverage gate threshold
    logloss_tolerance: float = 0.005      # pooled logloss tolerance vs incumbent
    auc_tolerance: float = 0.005          # sealed AUC tolerance vs incumbent
    ece_tolerance: float = 0.010          # ECE tolerance vs incumbent
    stability_max_delta: float = 0.020    # late-vs-early fold logloss delta

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OptimizationConfig":
        scopes = tuple(
            ModelScope(
                sport=str(s["sport"]),
                model=str(s["model"]),
                raw_fields=tuple(s["raw_fields"]),
                prod_features=tuple(s["prod_features"]),
                target=str(s["target"]),
                horizon_field=str(s.get("horizon_field", "")),
            )
            for s in raw.get("scopes", [])
        )
        thresholds = raw.get("gates", {}) or {}
        return cls(
            bounds=Bounds.from_mapping(raw.get("bounds")),
            scopes=scopes,
            coverage_min=float(thresholds.get("coverage_min", 0.90)),
            logloss_tolerance=float(
                thresholds.get("logloss_tolerance", 0.005)),
            auc_tolerance=float(thresholds.get("auc_tolerance", 0.005)),
            ece_tolerance=float(thresholds.get("ece_tolerance", 0.010)),
            stability_max_delta=float(
                thresholds.get("stability_max_delta", 0.020)),
        )


def load_optimization_config(raw: Mapping[str, Any] | None = None,
                             env_var: str = "OPTIMIZATION_CONFIG_JSON"
                             ) -> OptimizationConfig:
    """Load the declarative run config from a mapping or a JSON string in
    ``env_var``. Missing config raises — there is no silent default run."""
    if raw is None:
        blob = os.environ.get(env_var, "")
        if not blob:
            raise ValueError(
                f"optimization config required: pass a mapping or set {env_var}")
        import json

        raw = json.loads(blob)
    if "bounds" not in raw and "scopes" not in raw:
        raise ValueError("optimization config must declare bounds/scopes")
    return OptimizationConfig.from_mapping(raw)
