"""NHL SportAdapter — the semantic bridge between the NHL pipeline's
sport-specific schemas and the shared frontend interfaces
(``core.contracts``).

Market schemas stay sport-specific: the NHL serves BOTH a full-game
two-way moneyline (every game resolves via OT/SO) and a regulation
three-way moneyline (regulation tie = real outcome). TIE and PUSH remain
distinct:

* ``moneyline`` — full_game scope, ``tie_allowed=False`` (OT/SO resolves
  every full game; no drawn outcome exists);
* ``moneyline_regulation`` — regulation scope, ``tie_allowed=True`` with
  explicit ``p_tie`` (a regulation tie is NOT a full-game tie);
* ``puckline`` / ``total`` — full_game, integer lines can push
  (``push_allowed=True``), half-point lines (the NHL standard ±1.5, 5.5,
  6.5) cannot; a regulation tie is NOT automatically a puckline push
  (settlement reads the final margin incl. OT/SO vs the actual line).

The goalie matchup panel mirrors the MLB pitcher panel layout with the
per-team goalie's save% and GAA over his LAST 10 strictly-prior
appearances. Unannounced starters render as TBD with preserved missing
values — never fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.contracts import (
    ArtifactContract,
    ArtifactFamily,
    ParticipantBox,
)
from core.contracts import SettlementConfig
from sports.nhl.feature_registry import build_feature_contract


class NHLAdapter:
    """The ``SportAdapter`` implementation for the NHL."""

    sport_key = "nhl"

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------

    def artifact_contract(self) -> ArtifactContract:
        """NHL artifact families pinned to the platform naming convention."""
        dated = [
            ("nhl_moneyline_json", "nhl_moneyline_v1_*.json", "json"),
            ("nhl_calibration", "nhl_calibration_*.json", "json"),
            ("nhl_predictions_history", "nhl_predictions_history_*.csv",
             "csv"),
            ("nhl_power_rankings", "nhl_power_rankings_*.csv", "csv"),
            ("nhl_markets", "nhl_run_engine_markets_*.csv", "csv"),
            ("nhl_markets_monitor", "nhl_run_engine_monitor_*.json", "json"),
            ("nhl_goalie_matchup", "nhl_goalie_matchup_*.json", "json"),
            ("nhl_feature_json", "nhl_feature_v1_*.json", "json"),
            ("nhl_model_monitor", "nhl_model_monitor_*.json", "json"),
            ("nhl_shap_game", "nhl_shap_game_*.csv", "csv"),
            ("nhl_feature_drift", "nhl_run_engine_feature_drift_*.csv",
             "csv"),
            ("nhl_feature_coverage", "nhl_run_engine_feature_coverage_*.csv",
             "csv"),
        ]
        exempt = [
            ("nhl_oof_store", "nhl_oof_moneyline_*.csv", "csv"),
            ("nhl_fold_table", "nhl_fold_table_*.csv", "csv"),
        ]
        families = [ArtifactFamily(family=n, pattern=p, ext=e,
                                   description="NHL frontend artifact "
                                               "(20-day retention)")
                    for n, p, e in dated]
        families += [ArtifactFamily(family=n, pattern=p, ext=e,
                                    date_stamped=False,
                                    retention_exempt=True,
                                    description="NHL backend artifact "
                                                "(retention-exempt)")
                     for n, p, e in exempt]
        return ArtifactContract(sport="nhl", families=tuple(families))

    def feature_contract(self):
        return build_feature_contract()

    # ------------------------------------------------------------------
    # Game-card normalization (shared semantic fields)
    # ------------------------------------------------------------------

    def normalize_games(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Derive every display column the game cards expect.

        Produces (when inputs allow): ``game_status`` (Final/Live/
        Scheduled), ``evening_game``, ``day_game``, ``home_team_name``,
        ``away_team_name``, ``model_pick``, ``model_correct``,
        ``is_upset``, ``is_coin_flip``. Never fabricates scores, outcomes,
        or announced participants.
        """
        out = raw.copy()
        # game_status from decision state (no fabricated scores)
        decided = (out.get("home_score").notna()
                   & out.get("away_score").notna()) \
            if "home_score" in out.columns else pd.Series(False,
                                                           index=out.index)
        status = np.where(decided, "Final", "Scheduled")
        if "game_state" in out.columns:
            status = np.where(out["game_state"].isin(("live", "in_progress")),
                              "Live", status)
        out["game_status"] = status

        # evening/day split from UTC start hour
        hour = pd.to_datetime(out.get("start_time_utc"), errors="coerce",
                              utc=True).dt.hour \
            if "start_time_utc" in out.columns \
            else pd.Series(np.nan, index=out.index)
        hour = pd.to_numeric(hour, errors="coerce")
        out["evening_game"] = np.where(hour.isna(), np.nan,
                                       (hour >= 19).astype(float))
        out["day_game"] = np.where(hour.isna(), np.nan,
                                   (hour < 19).astype(float))

        # team names (passed-through when absent — never invented)
        if "home_team_name" not in out.columns:
            out["home_team_name"] = out.get("home_team")
        if "away_team_name" not in out.columns:
            out["away_team_name"] = out.get("away_team")

        # model pick / correctness / upset / coin flip
        p_home = pd.to_numeric(out.get("home_win_prob_model"),
                               errors="coerce")
        pick = np.where(
            p_home.isna(), None,
            np.where(p_home >= 0.5, out.get("home_team"),
                     out.get("away_team")))
        out["model_pick"] = pick

        actual_winner = None
        if decided.any() and {"home_team", "away_team"} <= set(out.columns):
            actual_winner = np.select(
                [out["home_score"] > out["away_score"],
                 out["home_score"] < out["away_score"]],
                [out["home_team"], out["away_team"]], default=None)
        if actual_winner is not None:
            out["model_correct"] = np.where(
                p_home.isna() | ~decided, np.nan,
                (out["model_pick"] == actual_winner).astype(float))
            fav = np.where(p_home >= 0.5, out.get("home_team"),
                           out.get("away_team"))
            out["is_upset"] = np.where(
                p_home.isna() | ~decided, np.nan,
                (fav != actual_winner).astype(float))
        else:
            out["model_correct"] = np.nan
            out["is_upset"] = np.nan

        out["is_coin_flip"] = np.where(
            p_home.isna(), np.nan,
            ((p_home - 0.5).abs() <= 0.02).astype(float))

        # de-duplicate on the game id
        if "game_id" in out.columns:
            out = out.drop_duplicates(subset=["game_id"], keep="last")
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Goalie matchup participant box (MLB pitcher-panel layout)
    # ------------------------------------------------------------------

    def participant_box(self, game_row: pd.Series,
                        side: str) -> ParticipantBox:
        """The per-team starting-goalie box for one game row.

        ``side`` is ``home`` or ``away``. Headline stats: save% and GAA
        over the goalie's LAST 10 strictly-prior appearances. A TBD box is
        returned when the starter is unannounced or the aggregates are
        missing — never a fabricated name.
        """
        name = game_row.get(f"goalie_{side}_name")
        sv = game_row.get(f"goalie_{side}_save_pct")
        gaa = game_row.get(f"goalie_{side}_gaa")
        has_name = isinstance(name, str) and bool(name.strip())
        if not has_name or pd.isna(sv) or pd.isna(gaa):
            return ParticipantBox(role="goalie", name="", stats=(),
                                  is_tbd=True)
        stats = (
            ("Save% (last 10)", f"{float(sv):.3f}"),
            ("GAA (last 10)", f"{float(gaa):.2f}"),
        )
        return ParticipantBox(role="goalie", name=str(name).strip(),
                              stats=stats, is_tbd=False)

    # ------------------------------------------------------------------
    # Market frame + probabilities (sport-specific schema)
    # ------------------------------------------------------------------

    def market_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize the NHL markets artifact into the adapter's frame.

        Keeps the sport-specific markets schema (kind / grids / outcome
        columns) and adds the shared display derivations. Never assigns
        MLB or NFL field meanings to NHL columns."""
        out = raw.copy()
        if "kind" in out.columns:
            out["is_slate"] = out["kind"] == "slate"
        return out

    def market_probs(self, market_row: pd.Series) -> dict[str, float]:
        """Semantic market probabilities for one game row.

        For the regulation three-way market ``p_tie`` carries the explicit
        regulation-tie mass (p_home + p_tie + p_away = 1, never
        renormalized away). For the full-game market the tie mass is 0 by
        construction (OT/SO resolves every game). ``p_push_<line>`` is
        included for integer lines where the wager can push.
        """
        p_home = float(market_row.get("p_home_win_derived"))
        p_tie = float(market_row.get("p_tie") or 0.0)
        p_away = float(market_row.get("p_away_win_derived"))
        probs: dict[str, float] = {"p_home": p_home, "p_away": p_away,
                                   "p_tie": p_tie}
        for key in ("p_cover_offered", "p_push_offered", "p_over_offered",
                    "p_under_offered", "p_push_total_offered"):
            v = market_row.get(key)
            if v is not None and not (isinstance(v, float)
                                      and np.isnan(v)):
                probs[key] = float(v)
        return probs

    # ------------------------------------------------------------------
    # Settlement configs (core.markets)
    # ------------------------------------------------------------------

    def settlement_config(self, market_kind: str) -> SettlementConfig:
        """The settlement config for one of the NHL's market kinds."""
        if market_kind == "moneyline":
            # Full-game: OT/SO resolves every game — a two-way market.
            return SettlementConfig(settlement_scope="full_game",
                                    tie_allowed=False, push_allowed=False)
        if market_kind == "moneyline_regulation":
            # Regulation 3-way: the regulation tie is a real outcome with
            # explicit p_tie.
            return SettlementConfig(settlement_scope="regulation",
                                    tie_allowed=True, push_allowed=False)
        if market_kind in ("puckline", "total"):
            # Integer lines can push; half-point lines cannot (enforced by
            # core.markets regardless). Settlement reads the FINAL margin
            # (incl. OT/SO) vs the actual line.
            return SettlementConfig(settlement_scope="full_game",
                                    tie_allowed=False, push_allowed=True)
        raise KeyError(
            f"unknown NHL market kind {market_kind!r} "
            f"(expected moneyline|moneyline_regulation|puckline|total)")
