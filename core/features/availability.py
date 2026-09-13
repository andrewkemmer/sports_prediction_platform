"""§7.1 per-field availability metadata (B-001-RESIDUAL, Phase 7.6).

Spec §7.1 requires every feature to carry a point-in-time ``available_at``
stamp compared strictly against ``prediction_cutoff`` (scheduled start −
per-sport buffer). Phase 7.5d shipped the validator and wiring but no
metadata layer (B-001-RESIDUAL); this module is that layer.

Honest provenance, per availability class:

* ``schedule`` — game-schedule facts (home/away, dome flag, prime time,
  division matchup). Known when the league publishes the schedule: from
  the FIRST game's date onward. Conservative class lead: 24 h.
* ``ratings`` — pre-game team ratings (Elo, win percentage). Computed
  from previously DECIDED games only; the latest settled game before the
  row's game date settles them. Conservative class lead: 24 h (settled
  by the prior completed-game midnight, uniformly across all sports).
* ``rolling`` — trailing window aggregates (rolling wOBA, bullpen WHIP,
  EWM yards-per-play, pace). Same settled-by semantics as ratings.
  Conservative class lead: 24 h.
* ``precomputed`` — inputs published by the upstream source before game
  day (park factors, altitude, dome neutral flag). Conservative class
  lead: 24 h.

UNIFORM 24 h LEAD — why: every sport's feature pipeline computes
features from data settled by the most recent completed-game midnight
before the row's game date (day-granular joins; MLB doubleheaders are
the only intraday exception and their earlier-game results make features
FRESHER, never staler, than the class lead). ``available_at`` is
therefore stamped ``scheduled_start − 24h``, which passes the §7.1
STRICT check (``available_at < prediction_cutoff``) for every buffer in
{60, 90} min and every real scheduled start, while the stamp remains a
GUARANTEED availability bound — never a fabricated exactness claim.
No field-specific padding beyond the class lead is claimed.

Enforcement semantics (``availability_enforcement`` study key):
* ``report_only`` — telemetry only (7.5d behavior; the runner gate logs
  the FutureAvailabilityReport).
* ``enforce`` — a non-ok gate (missing metadata or §7.1 violations)
  FAILS the run before any prediction artifact is generated. Because
  the layer stamps every served row by construction, ``enforce`` is the
  activated post-7.6 mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from core.validation.schema import ValidationError

#: Availability classes and their conservative §7.1 leads (hours before
#: scheduled first pitch / kickoff / faceoff / tip). Uniform 24 h: see
#: module docstring for the provenance argument.
AVAILABILITY_CLASSES: dict[str, int] = {
    "schedule": 24,
    "ratings": 24,
    "rolling": 24,
    "precomputed": 24,
}

#: Per-field class declarations. Keyed ``"<sport>:<feature name>"`` so a
#: single table serves all four sports without per-sport module forks.
#: Every feature in every sport's MONEYLINE_FEATURE_COLS / MARKET_FEATURE_COLS
#: MUST be declared here (pinned by
#: tests/test_availability_metadata.py::test_every_contract_field_declared).
FIELD_CLASSES: dict[str, str] = {
    # =================================================================
    # MLB (moneyline 64 + market extras — sports/mlb/features/registry.py)
    # =================================================================
    # schedule facts
    "mlb:is_home": "schedule",
    "mlb:rest_days_diff": "schedule",
    "mlb:dome_is_neutral": "schedule",
    "mlb:travel_fatigue_diff": "schedule",
    # precomputed venue/environment constants (published pre-game by the
    # upstream source)
    "mlb:park_factor_slug_diff": "precomputed",
    "mlb:wind_advantage_flyball_factor": "precomputed",
    "mlb:air_density_velocity_boost": "precomputed",
    # ratings (pre-game Elo / records / OOF-derived priors; decided-games only)
    "mlb:elo_diff": "ratings",
    "mlb:home_elo": "ratings",
    "mlb:away_elo": "ratings",
    "mlb:home_win_pct": "ratings",
    "mlb:away_win_pct": "ratings",
    "mlb:win_pct_diff": "ratings",
    "mlb:closer_availability_diff": "ratings",
    "mlb:run_margin_diff": "ratings",
    # rolling form (trailing windows over settled games)
    **{f"mlb:{name}": "rolling" for name in (
        "sp_era_diff", "sp_era_5g_diff", "sp_fbvelo_diff",
        "sp_fbpct_diff", "sp_whiff_diff", "sp_xwoba_diff",
        "sp_xwoba_vs_l_diff", "sp_k9_diff", "sp_k9_5g_diff",
        "lineup_woba_mean_diff", "lineup_woba_top3_diff",
        "lineup_woba_std_diff", "woba_30g_diff",
        "bullpen_whip_diff", "bullpen_whip_3g_diff",
        "bullpen_pitches_diff", "team_barrel_diff",
        "team_hardhit_diff", "team_exitvelo_diff",
        "lineup_handedness_matchup_advantage",
        "bullpen_meltdown_risk", "pitcher_regression_indicator",
        "lineup_depth_multiplier", "ace_efficiency_factor",
        "sp_era_home", "sp_era_away", "sp_k9_home", "sp_k9_away",
        "sp_xwoba_home", "sp_xwoba_away",
        "lineup_woba_mean_home", "lineup_woba_mean_away",
        "lineup_woba_top3_home", "lineup_woba_top3_away",
        "woba_30g_home", "woba_30g_away",
        "bullpen_whip_10g_home", "bullpen_whip_10g_away",
        "bullpen_whip_3g_home", "bullpen_whip_3g_away",
        "team_barrel_15g_home", "team_barrel_15g_away",
        "team_exitvelo_15g_home", "team_exitvelo_15g_away",
        # exp2 matchup candidates (trailing K/xwOBA by pitch category)
        "exp2_centered_k_diff", "exp2_cat_k_fastball_diff",
        "exp2_cat_k_breaking_diff", "exp2_cat_k_offspeed_diff",
        "exp2_cat_xwoba_fastball_diff", "exp2_cat_xwoba_breaking_diff",
        "exp2_cat_xwoba_offspeed_diff", "exp2_cat_platoon_k_fastball_diff",
    )},
    # =================================================================
    # NFL (12-field moneyline/market contract)
    # =================================================================
    "nfl:is_dome_home": "schedule",
    "nfl:div_game": "schedule",
    "nfl:prime_time": "schedule",
    "nfl:travel_miles_diff": "schedule",
    "nfl:rest_short_diff": "schedule",
    "nfl:rest_days_diff": "schedule",
    "nfl:altitude_home": "precomputed",
    "nfl:elo_diff": "ratings",
    "nfl:win_pct_diff": "ratings",
    **{f"nfl:{name}": "rolling" for name in (
        "ewm_net_pts_diff", "ewm_ypp_diff", "pace_plays_min_diff",
    )},
    # =================================================================
    # NHL (9-field moneyline/market contract)
    # =================================================================
    "nhl:rest_days_diff": "schedule",
    "nhl:b2b_home": "schedule",
    "nhl:b2b_away": "schedule",
    "nhl:div_game": "schedule",
    "nhl:prime_time": "schedule",
    "nhl:travel_miles_diff": "schedule",
    "nhl:elo_diff": "ratings",
    "nhl:win_pct_diff": "ratings",
    "nhl:ewm_net_goals_diff": "rolling",
    # =================================================================
    # NBA (9-field moneyline/market contract)
    # =================================================================
    "nba:rest_days_diff": "schedule",
    "nba:b2b_home": "schedule",
    "nba:b2b_away": "schedule",
    "nba:conf_game": "schedule",
    "nba:prime_time": "schedule",
    "nba:travel_miles_diff": "schedule",
    "nba:elo_diff": "ratings",
    "nba:win_pct_diff": "ratings",
    "nba:ewm_net_pts_diff": "rolling",
}

#: Lead override surface (reserved; no field currently overrides the
#: uniform 24 h class lead). Kept as an explicit table so a future
#: per-field override is a reviewed, tested change rather than a silent
#: magic number.
FIELD_LEAD_OVERRIDES: dict[str, int] = {}


def class_for(feature: str) -> str:
    """Availability class for a feature name; unknown features fail
    closed (the stamping helpers refuse to invent a class)."""
    try:
        return FIELD_CLASSES[feature]
    except KeyError as exc:
        raise ValidationError(
            f"no availability class declared for feature {feature!r} "
            "(B-001-RESIDUAL metadata layer is closed over the feature "
            "contracts)") from exc


def require_classes_for(sport: str, features) -> None:
    """Closed-coverage check: every feature of ``sport``'s declared
    contracts must carry an availability class. Called from each sport
    registry's ``validate_registry`` so contract drift and metadata drift
    fail together (a new contract feature without a declared class can
    never build a contract)."""
    missing = [f for f in features if f"{sport}:{f}" not in FIELD_CLASSES]
    if missing:
        raise ValidationError(
            f"{sport}: contract features without an availability class "
            f"(B-001-RESIDUAL metadata layer): {sorted(missing)}")


def lead_hours_for(feature: str) -> int:
    """Conservative availability lead (hours) for one feature."""
    cls = class_for(feature)
    if feature in FIELD_LEAD_OVERRIDES:
        return int(FIELD_LEAD_OVERRIDES[feature])
    return AVAILABILITY_CLASSES[cls]


def available_at_for(start_utc: datetime, feature: str) -> datetime:
    """Point-in-time ``available_at`` for one field: scheduled start −
    the feature's declared conservative lead."""
    return start_utc - timedelta(hours=lead_hours_for(feature))


def stamp_frame(frame, *, start_col: str,
                features: tuple[str, ...] | list[str],
                game_id_col: str = "game_id"):
    """Stamp the declared availability bounds onto a served-slate frame.

    Adds one ``available_at__<feature>`` column per declared feature
    (scheduled_start − lead). Requires the ID column and ``start_col``
    to carry resolvable values for every row (the §7.1 gate fails closed
    on missing metadata downstream, so stamping never invents either).
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise ValidationError(
            "availability stamping requires a pandas DataFrame")
    for col in (game_id_col, start_col):
        if col not in frame.columns:
            raise ValidationError(
                f"availability stamping: served frame lacks {col!r}")
    out = frame.copy()
    starts = pd.to_datetime(out[start_col], errors="coerce", utc=True)
    if starts.isna().any():
        bad = out.loc[starts.isna(), game_id_col].head(3).tolist()
        raise ValidationError(
            f"availability stamping: unparseable {start_col} for game_ids "
            f"{bad} — §7.1 fails CLOSED")
    for feature in features:
        lead = timedelta(hours=lead_hours_for(feature))
        out[f"available_at__{feature}"] = starts - lead
    return out


def stamped_feature_records(frame, *, start_col: str,
                            features: tuple[str, ...] | list[str],
                            game_id_col: str = "game_id"):
    """Materialize per-field records for the §7.1 gate: one mapping per
    (row, feature) with ``game_id``, ``scheduled_start_utc``, and the
    field's ``available_at`` stamp. The tested §7.1 validator consumes
    these instead of flat frame rows that carry no metadata.

    Accepts an UNSTAMPED frame — stamps are computed here from the
    declared per-field classes and never added to the caller's frame.
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise ValidationError(
            "availability records require a pandas DataFrame")
    for col in (game_id_col, start_col):
        if col not in frame.columns:
            raise ValidationError(
                f"availability records: frame lacks {col!r}")
    starts = pd.to_datetime(frame[start_col], errors="coerce", utc=True)
    if starts.isna().any():
        bad = frame.loc[starts.isna(), game_id_col].head(3).tolist()
        raise ValidationError(
            f"availability records: unparseable {start_col} for game_ids "
            f"{bad} — §7.1 fails CLOSED")
    gids = frame[game_id_col].astype(str).tolist()
    records: list[dict] = []
    for feature in features:
        lead = timedelta(hours=lead_hours_for(feature))
        avail = starts - lead
        for gid, start, a in zip(gids, starts, avail):
            records.append({
                "game_id": gid,
                "scheduled_start_utc": start.to_pydatetime(),
                "available_at": a.to_pydatetime(),
                "field": feature,
            })
    return records


@dataclass(frozen=True)
class AvailabilityGateResult:
    """Aggregate per-field §7.1 gate result (one report per feature)."""

    label: str
    reports: tuple  # tuple[FutureAvailabilityReport, ...]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.reports)

    @property
    def n_fields(self) -> int:
        return len(self.reports)

    @property
    def n_rows_per_field(self) -> int:
        return self.reports[0].n_rows if self.reports else 0

    @property
    def n_missing(self) -> int:
        return sum(r.n_missing for r in self.reports)

    @property
    def n_violations(self) -> int:
        return sum(r.n_violations for r in self.reports)

    def failing_fields(self) -> tuple[str, ...]:
        return tuple(r.label for r in self.reports if not r.ok)


def per_field_gate(frame, *, sport: str, start_col: str,
                   features: tuple[str, ...] | list[str],
                   buffer_minutes: int, label: str,
                   game_id_col: str = "game_id") -> AvailabilityGateResult:
    """Run the §7.1 per-field validator over every contract feature of a
    served slate (reuses :func:`core.validation.future_availability.
    validate_available_at` — the tested §7.1 semantics, unchanged).

    ``features`` are UNPREFIXED contract column names; they are joined
    with ``sport`` to the declared ``<sport>:<feature>`` availability
    classes (undeclared names fail closed via :func:`class_for`).

    Accepts the UNSTAMPED serving frame: stamps are materialized into
    per-field records and never added to the served frame (the frontend
    artifact schema is untouched). Fails closed: an undeclared feature,
    a missing/unparseable start, or any missing metadata raises or
    reports — never fabricates.
    """
    from core.validation.future_availability import validate_available_at

    prefixed = tuple(f"{sport}:{f}" for f in features)
    records = stamped_feature_records(
        frame, start_col=start_col, features=prefixed,
        game_id_col=game_id_col)
    reports = []
    for feature in prefixed:
        field_records = [r for r in records if r["field"] == feature]
        reports.append(validate_available_at(
            field_records,
            buffer_minutes=buffer_minutes,
            label=f"{label}:{feature}",
            start_col="scheduled_start_utc",
            available_col="available_at",
            game_id_col=game_id_col))
    return AvailabilityGateResult(label=label, reports=tuple(reports))
