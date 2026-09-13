"""NHL study configuration loader.

The study YAML is the single source of truth for warmup, fold geometry, OOF
rules, minimum training rows, Elo parameters, trailing windows, and market
grids. Production run-window environment variables (``NHL_START_DATE`` /
``NHL_END_DATE`` / ``NHL_FULL_REPULL``) scope only ingestion/serving —
never training history. This module parses the YAML into typed dataclasses
and fails loudly on unknown/invalid fields (no silent defaults drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from core.config import WalkForwardConfig
from core.config.platform import WarmupConfig


class NHLStudyError(ValueError):
    """Raised when the NHL study configuration is missing or invalid."""


STUDY_PATH = Path(__file__).resolve().parent / "study.yaml"

_ALLOWED_TOP = {
    "sport", "kind", "study_id", "warmup_seasons", "oof_first_season",
    "windows", "elo", "venue", "warmup", "walk_forward", "oof", "training",
    "market", "retention", "allowlisted_families",
    "retention_exempt_families",
    "prediction_cutoff_buffer_minutes", "availability_enforcement",
}


@dataclass(frozen=True)
class OOFRules:
    """Fixed OOF rules (Phase 4: frozen, no dynamic optimization)."""

    require_prob_strictly_inside: bool = True
    undecided_rows_excluded: bool = True
    require_unique_game_id: bool = True
    require_decided_only: bool = True
    min_training_rows: int = 50

    def __post_init__(self) -> None:
        if self.min_training_rows < 1:
            raise NHLStudyError("oof.min_training_rows must be >= 1")


@dataclass(frozen=True)
class WindowsConfig:
    """Reference-frozen trailing windows (games)."""

    form_window: int = 4
    winpct_window: int = 12
    ewm_halflife: float = 2.0
    goalie_window: int = 10      # display-only; never a model feature


@dataclass(frozen=True)
class EloConfig:
    prior: float = 1500.0
    k: float = 20.0
    scale: float = 400.0


@dataclass(frozen=True)
class VenueConfig:
    prime_time_hour: int = 19


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen training geometry (Phase 4: no optimization)."""

    ensemble_members: tuple[str, ...] = (
        "xgboost", "lightgbm", "logistic", "randomforest", "mlp")
    retrain_cadence_days: int = 7
    min_train_days_warmup: int = 0
    random_seed: int = 42
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    adaptive_weight_metric: str = "auc"
    adaptive_weight_temperature: float = 0.015
    adaptive_weight_floor: float = 0.05
    adaptive_weight_cap: float = 0.45

    def __post_init__(self) -> None:
        if self.retrain_cadence_days < 1:
            raise NHLStudyError(
                "training.retrain_cadence_days must be >= 1")
        if not self.ensemble_members:
            raise NHLStudyError(
                "training.ensemble_members must be non-empty")
        if sorted(self.ensemble_members) != sorted(
                ("xgboost", "lightgbm", "logistic", "randomforest", "mlp")):
            raise NHLStudyError(
                "Phase 4 freezes the intended five-member stack; got "
                f"{self.ensemble_members}")
        if self.random_seed < 0:
            raise NHLStudyError("training.random_seed must be >= 0")


@dataclass(frozen=True)
class MarketConfig:
    """Run-engine distribution grids (NHL-sized, frozen)."""

    spread_grid: tuple[int, int] = (-4, 4)
    total_grid: tuple[int, int] = (5, 12)
    half_stop_lines: tuple[float, ...] = (-1.5, 1.5, 5.5, 6.5)
    margin_pmf_max: int = 13
    total_pmf_max: int = 16
    margin_sigma: float = 2.8
    total_sigma: float = 2.0
    sigma_floor_margin: float = 2.0
    sigma_cap_margin: float = 4.0
    sigma_floor_total: float = 1.4
    sigma_cap_total: float = 2.8
    p_tie_max: float = 0.30
    fixed_totals: tuple[int, ...] = (5, 6, 7)
    canonical_spreads: tuple[int, ...] = (1,)

    @property
    def spread_lines(self) -> list[int]:
        lo, hi = self.spread_grid
        return list(range(lo, hi + 1))

    @property
    def total_lines(self) -> list[int]:
        lo, hi = self.total_grid
        return list(range(lo, hi + 1))


#: Phase 7.5b canonical contract ownership: the feature contracts are
#: DECLARED in ``sports.nhl.feature_registry`` and imported here — the
#: study config binds them onto the study dataclass but never declares
#: feature lists directly.
from sports.nhl.feature_registry import (  # noqa: E402
    MARKET_CONTRACT_VERSION,
    MARKET_FEATURE_COLS,
    MONEYLINE_CONTRACT_VERSION,
    MONEYLINE_FEATURE_COLS,
    PRIOR_CONTRACT_VERSION,
)


@dataclass(frozen=True)
class NHLStudy:
    """The parsed NHL study configuration."""

    study_id: str
    kind: str
    sport: str
    warmup_seasons: tuple[int, ...]
    oof_first_season: int
    windows: WindowsConfig
    elo: EloConfig
    venue: VenueConfig
    warmup: WarmupConfig
    walk_forward: WalkForwardConfig
    min_val_games: int
    max_eval_folds: int
    oof: OOFRules
    training: TrainingConfig
    market: MarketConfig
    frontend_days: int
    allowlisted_families: tuple[str, ...]
    retention_exempt_families: tuple[str, ...]
    prediction_cutoff_buffer_minutes: int
    availability_enforcement: str
    moneyline_feature_cols: tuple[str, ...] = MONEYLINE_FEATURE_COLS
    moneyline_contract_version: str = MONEYLINE_CONTRACT_VERSION
    market_feature_cols: tuple[str, ...] = MARKET_FEATURE_COLS
    market_contract_version: str = MARKET_CONTRACT_VERSION
    source_path: Path | None = None

    @property
    def all_seasons(self) -> list[int]:
        return list(self.warmup_seasons) + [
            s for s in range(self.oof_first_season,
                             self.oof_first_season + 8)]

    @property
    def elo_k(self) -> float:
        return self.elo.k

    @property
    def elo_prior(self) -> float:
        return self.elo.prior

    @property
    def elo_scale(self) -> float:
        return self.elo.scale

    @property
    def prime_time_hour(self) -> int:
        return self.venue.prime_time_hour

    @property
    def form_window(self) -> int:
        return self.windows.form_window

    @property
    def winpct_window(self) -> int:
        return self.windows.winpct_window

    @property
    def shots_window(self) -> int:
        return self.windows.shots_window

    @property
    def ewm_halflife(self) -> float:
        return self.windows.ewm_halflife

    @property
    def goalie_window(self) -> int:
        return self.windows.goalie_window


def load_nhl_study(path: str | Path | None = None) -> NHLStudy:
    """Load and validate the NHL study configuration.

    Fails loudly (``NHLStudyError``) on a missing file, unknown top-level
    fields, missing required sections, or invalid values — never on silent
    defaults.
    """
    p = Path(path) if path is not None else STUDY_PATH
    if not p.is_file():
        raise NHLStudyError(f"NHL study config not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise NHLStudyError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise NHLStudyError(f"study config {p} must be a mapping")

    unknown = set(data) - _ALLOWED_TOP
    if unknown:
        raise NHLStudyError(f"unknown study fields: {sorted(unknown)}")

    for required in ("sport", "kind", "study_id", "warmup_seasons",
                     "oof_first_season", "warmup", "walk_forward", "oof"):
        if required not in data:
            raise NHLStudyError(
                f"study config missing required field: {required}")

    if str(data["sport"]) != "nhl":
        raise NHLStudyError(
            f"study config is for sport {data['sport']!r}, not nhl")
    if str(data["kind"]) != "production":
        raise NHLStudyError(
            f"study kind must be 'production', got {data['kind']!r}")

    warmup_seasons = tuple(int(s) for s in data["warmup_seasons"])
    oof_first_season = int(data["oof_first_season"])
    if not warmup_seasons or oof_first_season <= max(warmup_seasons):
        raise NHLStudyError(
            "oof_first_season must be after every warmup season")

    w_raw = data.get("windows") or {}
    windows = WindowsConfig(
        form_window=int(w_raw.get("form_window", 4)),
        winpct_window=int(w_raw.get("winpct_window", 12)),
        ewm_halflife=float(w_raw.get("ewm_halflife", 2.0)),
        goalie_window=int(w_raw.get("goalie_window", 10)),
    )
    e_raw = data.get("elo") or {}
    elo = EloConfig(prior=float(e_raw.get("prior", 1500.0)),
                    k=float(e_raw.get("k", 20.0)),
                    scale=float(e_raw.get("scale", 400.0)))
    v_raw = data.get("venue") or {}
    venue = VenueConfig(prime_time_hour=int(v_raw.get("prime_time_hour", 19)))

    w = data["warmup"]
    try:
        warmup = WarmupConfig(
            min_history_days=int(w["min_history_days"]),
            min_games_per_team=int(w["min_games_per_team"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NHLStudyError(f"invalid warmup block: {exc}") from exc

    wf_raw = data["walk_forward"]
    try:
        walk_forward = WalkForwardConfig(
            min_train_games=int(wf_raw["min_train_games"]),
            step_days=int(wf_raw["step_days"]),
            embargo_days=int(wf_raw.get("embargo_days", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NHLStudyError(f"invalid walk_forward block: {exc}") from exc
    min_val_games = int(wf_raw.get("min_val_games", 1))
    max_eval_folds = int(wf_raw.get("max_eval_folds", 0))
    if min_val_games < 1:
        raise NHLStudyError("walk_forward.min_val_games must be >= 1")
    if max_eval_folds < 0:
        raise NHLStudyError("walk_forward.max_eval_folds must be >= 0")

    o = data["oof"]
    try:
        oof = OOFRules(
            require_prob_strictly_inside=bool(o.get(
                "require_prob_strictly_inside", True)),
            undecided_rows_excluded=bool(o.get(
                "undecided_rows_excluded", True)),
            require_unique_game_id=bool(o.get("require_unique_game_id", True)),
            require_decided_only=bool(o.get("require_decided_only", True)),
            min_training_rows=int(o["min_training_rows"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NHLStudyError(f"invalid oof block: {exc}") from exc

    t_raw = data.get("training") or {}
    training = TrainingConfig(
        ensemble_members=tuple(t_raw.get(
            "ensemble_members", TrainingConfig.ensemble_members)),
        retrain_cadence_days=int(t_raw.get(
            "retrain_cadence_days", TrainingConfig.retrain_cadence_days)),
        min_train_days_warmup=int(t_raw.get(
            "min_train_days_warmup", TrainingConfig.min_train_days_warmup)),
        random_seed=int(t_raw.get("random_seed", 42)),
        ensemble_weights=dict(t_raw.get("ensemble_weights") or {}),
        adaptive_weight_metric=str(t_raw.get(
            "adaptive_weight_metric", "auc")),
        adaptive_weight_temperature=float(t_raw.get(
            "adaptive_weight_temperature", 0.015)),
        adaptive_weight_floor=float(t_raw.get(
            "adaptive_weight_floor", 0.05)),
        adaptive_weight_cap=float(t_raw.get("adaptive_weight_cap", 0.45)),
    )

    m_raw = data.get("market") or {}
    market = MarketConfig(
        spread_grid=tuple(m_raw.get("spread_grid", (-4, 4))),
        total_grid=tuple(m_raw.get("total_grid", (5, 12))),
        half_stop_lines=tuple(m_raw.get(
            "half_stop_lines", (-1.5, 1.5, 5.5, 6.5))),
        margin_pmf_max=int(m_raw.get("margin_pmf_max", 13)),
        total_pmf_max=int(m_raw.get("total_pmf_max", 16)),
        margin_sigma=float(m_raw.get("margin_sigma", 2.8)),
        total_sigma=float(m_raw.get("total_sigma", 2.0)),
        sigma_floor_margin=float(m_raw.get("sigma_floor_margin", 2.0)),
        sigma_cap_margin=float(m_raw.get("sigma_cap_margin", 4.0)),
        sigma_floor_total=float(m_raw.get("sigma_floor_total", 1.4)),
        sigma_cap_total=float(m_raw.get("sigma_cap_total", 2.8)),
        p_tie_max=float(m_raw.get("p_tie_max", 0.30)),
        fixed_totals=tuple(m_raw.get("fixed_totals", (5, 6, 7))),
        canonical_spreads=tuple(m_raw.get("canonical_spreads", (1,))),
    )

    r_raw = data.get("retention") or {}
    frontend_days = int(r_raw.get("frontend_days", 20))
    if frontend_days != 20:
        raise NHLStudyError(
            f"retention.frontend_days must be 20 (got {frontend_days})")

    # §7.1 PIT gate (B-001 WS4): buffer hard-pinned, mode validated.
    buffer = int(data.get("prediction_cutoff_buffer_minutes", 60))
    if buffer != 60:
        raise NHLStudyError(
            f"prediction_cutoff_buffer_minutes must be 60 per §7.1 (NHL), "
            f"got {buffer}")
    from core.validation.future_availability import (
        normalize_enforcement_mode)
    availability_enforcement = normalize_enforcement_mode(
        data.get("availability_enforcement", "report_only"), "nhl study")

    allow = tuple(data.get("allowlisted_families") or ())
    if not allow:
        raise NHLStudyError("allowlisted_families must be non-empty")
    exempt = tuple(data.get("retention_exempt_families") or ())
    overlap = set(allow) & set(exempt)
    if overlap:
        raise NHLStudyError(
            "families cannot be both allowlisted and retention-exempt: "
            f"{sorted(overlap)}")

    return NHLStudy(
        study_id=str(data["study_id"]),
        kind=str(data["kind"]),
        sport=str(data["sport"]),
        warmup_seasons=warmup_seasons,
        oof_first_season=oof_first_season,
        windows=windows,
        elo=elo,
        venue=venue,
        warmup=warmup,
        walk_forward=walk_forward,
        min_val_games=min_val_games,
        max_eval_folds=max_eval_folds,
        oof=oof,
        training=training,
        market=market,
        frontend_days=frontend_days,
        allowlisted_families=allow,
        retention_exempt_families=exempt,
        prediction_cutoff_buffer_minutes=buffer,
        availability_enforcement=availability_enforcement,
        moneyline_feature_cols=MONEYLINE_FEATURE_COLS,
        moneyline_contract_version=MONEYLINE_CONTRACT_VERSION,
        market_feature_cols=MARKET_FEATURE_COLS,
        market_contract_version=MARKET_CONTRACT_VERSION,
        source_path=p,
    )
