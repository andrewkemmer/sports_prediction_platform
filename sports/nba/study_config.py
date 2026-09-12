"""NBA study configuration (Phase 5).

The study YAML is the single source of truth for warmup, fold geometry,
OOF rules, minimum training rows, Elo parameters, trailing windows, and
market grids. Production run-window environment variables
(``NBA_START_DATE`` / ``NBA_END_DATE`` / ``NBA_FULL_REPULL``) scope only
ingestion/serving — never training history. This module parses the YAML
into typed dataclasses and fails loudly on unknown/invalid fields (no
silent defaults drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from core.config import WalkForwardConfig
from core.warmup import WarmupConfig


class NBAStudyError(ValueError):
    """Raised when the NBA study configuration is missing or invalid."""


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
    """Fixed OOF rules (Phase 5: frozen, no dynamic optimization)."""

    require_prob_strictly_inside: bool = True
    undecided_rows_excluded: bool = True
    require_unique_game_id: bool = True
    require_decided_only: bool = True
    min_training_rows: int = 50

    def __post_init__(self) -> None:
        if self.min_training_rows < 1:
            raise NBAStudyError("oof.min_training_rows must be >= 1")


@dataclass(frozen=True)
class WindowsConfig:
    """Reference-frozen trailing windows (games)."""

    form_window: int = 5
    winpct_window: int = 12
    ewm_halflife: float = 3.0
    player_window: int = 10     # display-only; never a model feature


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
    """Frozen training geometry (Phase 5: no optimization)."""

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
            raise NBAStudyError(
                "training.retrain_cadence_days must be >= 1")
        if not self.ensemble_members:
            raise NBAStudyError(
                "training.ensemble_members must be non-empty")
        if sorted(self.ensemble_members) != sorted(
                ("xgboost", "lightgbm", "logistic", "randomforest", "mlp")):
            raise NBAStudyError(
                "Phase 5 freezes the intended five-member stack; got "
                f"{self.ensemble_members}")
        if self.random_seed < 0:
            raise NBAStudyError("training.random_seed must be >= 0")


@dataclass(frozen=True)
class MarketConfig:
    """Run-engine distribution grids (NBA-sized, frozen)."""

    spread_grid: tuple[int, int] = (-20, 20)
    total_grid: tuple[int, int] = (200, 250)
    half_stop_lines: tuple[float, ...] = ()
    margin_pmf_max: int = 45
    total_pmf_max: int = 300
    margin_sigma: float = 11.0
    total_sigma: float = 18.0
    sigma_floor_margin: float = 8.0
    sigma_cap_margin: float = 16.0
    sigma_floor_total: float = 12.0
    sigma_cap_total: float = 26.0
    p_tie_max: float = 0.0      # an NBA full game CANNOT tie (OT resolves)
    fixed_totals: tuple[int, ...] = (220, 225, 230)
    canonical_spreads: tuple[int, ...] = (0,)

    @property
    def spread_lines(self) -> list[int]:
        lo, hi = self.spread_grid
        return list(range(lo, hi + 1))

    @property
    def total_lines(self) -> list[int]:
        lo, hi = self.total_grid
        return list(range(lo, hi + 1))


#: Phase 7.5b canonical contract ownership: the feature contracts are
#: DECLARED in ``sports.nba.feature_registry`` and imported here — the
#: study config binds them onto the study dataclass but never declares
#: feature lists directly.
from sports.nba.feature_registry import (  # noqa: E402
    MARKET_CONTRACT_VERSION,
    MARKET_FEATURE_COLS,
    MONEYLINE_CONTRACT_VERSION,
    MONEYLINE_FEATURE_COLS,
    PRIOR_CONTRACT_VERSION,
)

#: Historic single-pool name, re-exported from the registry for backward
#: compatibility with existing consumers (the served pool IS the
#: moneyline contract).
FEATURE_COLUMNS: tuple[str, ...] = MONEYLINE_FEATURE_COLS


@dataclass(frozen=True)
class NBAStudy:
    """The parsed NBA study configuration."""

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
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
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
    def player_window(self) -> int:
        return self.windows.player_window

    @property
    def ewm_halflife(self) -> float:
        return self.windows.ewm_halflife


def load_nba_study(path: str | Path | None = None) -> NBAStudy:
    """Load and validate the NBA study configuration.

    Fails loudly (``NBAStudyError``) on a missing file, unknown top-level
    fields, missing required sections, or invalid values — never on silent
    defaults.
    """
    p = Path(path) if path is not None else STUDY_PATH
    if not p.is_file():
        raise NBAStudyError(f"NBA study config not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise NBAStudyError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise NBAStudyError(f"study config {p} must be a mapping")

    unknown = set(data) - _ALLOWED_TOP
    if unknown:
        raise NBAStudyError(f"unknown study fields: {sorted(unknown)}")

    for required in ("sport", "kind", "study_id", "warmup_seasons",
                     "oof_first_season", "warmup", "walk_forward", "oof"):
        if required not in data:
            raise NBAStudyError(
                f"study config missing required field: {required}")

    if str(data["sport"]) != "nba":
        raise NBAStudyError(
            f"study config is for sport {data['sport']!r}, not nba")
    if str(data["kind"]) != "production":
        raise NBAStudyError(
            f"study kind must be 'production', got {data['kind']!r}")

    warmup_seasons = tuple(int(s) for s in data["warmup_seasons"])
    oof_first_season = int(data["oof_first_season"])
    if not warmup_seasons or oof_first_season <= max(warmup_seasons):
        raise NBAStudyError(
            "oof_first_season must be after every warmup season")

    w_raw = data.get("windows") or {}
    windows = WindowsConfig(
        form_window=int(w_raw.get("form_window", 5)),
        winpct_window=int(w_raw.get("winpct_window", 12)),
        ewm_halflife=float(w_raw.get("ewm_halflife", 3.0)),
        player_window=int(w_raw.get("player_window", 10)),
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
        raise NBAStudyError(f"invalid warmup block: {exc}") from exc

    wf_raw = data["walk_forward"]
    try:
        walk_forward = WalkForwardConfig(
            min_train_games=int(wf_raw["min_train_games"]),
            step_days=int(wf_raw["step_days"]),
            embargo_days=int(wf_raw.get("embargo_days", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NBAStudyError(f"invalid walk_forward block: {exc}") from exc
    min_val_games = int(wf_raw.get("min_val_games", 1))
    max_eval_folds = int(wf_raw.get("max_eval_folds", 0))
    if min_val_games < 1:
        raise NBAStudyError("walk_forward.min_val_games must be >= 1")
    if max_eval_folds < 0:
        raise NBAStudyError("walk_forward.max_eval_folds must be >= 0")

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
        raise NBAStudyError(f"invalid oof block: {exc}") from exc

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
        spread_grid=tuple(m_raw.get("spread_grid", (-20, 20))),
        total_grid=tuple(m_raw.get("total_grid", (200, 250))),
        half_stop_lines=tuple(m_raw.get("half_stop_lines", ())),
        margin_pmf_max=int(m_raw.get("margin_pmf_max", 45)),
        total_pmf_max=int(m_raw.get("total_pmf_max", 300)),
        margin_sigma=float(m_raw.get("margin_sigma", 11.0)),
        total_sigma=float(m_raw.get("total_sigma", 18.0)),
        sigma_floor_margin=float(m_raw.get("sigma_floor_margin", 8.0)),
        sigma_cap_margin=float(m_raw.get("sigma_cap_margin", 16.0)),
        sigma_floor_total=float(m_raw.get("sigma_floor_total", 12.0)),
        sigma_cap_total=float(m_raw.get("sigma_cap_total", 26.0)),
        p_tie_max=float(m_raw.get("p_tie_max", 0.0)),
        fixed_totals=tuple(m_raw.get("fixed_totals", (220, 225, 230))),
        canonical_spreads=tuple(m_raw.get("canonical_spreads", (0,))),
    )

    r_raw = data.get("retention") or {}
    frontend_days = int(r_raw.get("frontend_days", 20))
    if frontend_days != 20:
        raise NBAStudyError(
            f"retention.frontend_days must be 20 (got {frontend_days})")

    # §7.1 PIT gate (B-001 WS4): buffer hard-pinned, mode validated.
    buffer = int(data.get("prediction_cutoff_buffer_minutes", 90))
    if buffer != 90:
        raise NBAStudyError(
            f"prediction_cutoff_buffer_minutes must be 90 per §7.1 (NBA), "
            f"got {buffer}")
    from core.validation.future_availability import (
        normalize_enforcement_mode)
    availability_enforcement = normalize_enforcement_mode(
        data.get("availability_enforcement", "report_only"), "nba study")

    allow = tuple(data.get("allowlisted_families") or ())
    if not allow:
        raise NBAStudyError("allowlisted_families must be non-empty")
    exempt = tuple(data.get("retention_exempt_families") or ())
    overlap = set(allow) & set(exempt)
    if overlap:
        raise NBAStudyError(
            "families cannot be both allowlisted and retention-exempt: "
            f"{sorted(overlap)}")

    return NBAStudy(
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
        feature_columns=FEATURE_COLUMNS,
        moneyline_feature_cols=MONEYLINE_FEATURE_COLS,
        moneyline_contract_version=MONEYLINE_CONTRACT_VERSION,
        market_feature_cols=MARKET_FEATURE_COLS,
        market_contract_version=MARKET_CONTRACT_VERSION,
        source_path=p,
    )
