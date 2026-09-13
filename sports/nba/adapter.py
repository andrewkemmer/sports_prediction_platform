"""NBA SportAdapter — the semantic bridge between the NBA pipeline's
sport-specific schemas and the shared frontend interfaces
(``core.contracts``).

Market schemas stay sport-specific: an NBA game CANNOT end in a tie —
overtime periods repeat until a winner exists — so every NBA market is
resolved at full_game scope with ``tie_allowed=False`` and TIE is
structurally impossible (never emitted, never fabricated). TIE and PUSH
remain distinct concepts:

* ``moneyline`` — full_game scope, ``tie_allowed=False`` (no drawn
  outcome exists in the sport);
* ``spread`` / ``total`` — full_game, integer lines can push
  (``push_allowed=True``) with explicit ``p_push``; half-point lines
  cannot push (enforced by core.markets regardless).

The participant matchup panel mirrors the MLB pitcher panel layout with
each team's highest-scoring player's PPG and APG over his LAST 10
strictly-prior games. An unavailable expected player renders as TBD with
preserved missing values — never fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.contracts import (
    ArtifactContract,
    ArtifactFamily,
    ParticipantBox,
)
from core.config import load_market_rules
from core.config.sources import SportMarketRules
from core.contracts import SettlementConfig

#: Settlement rules singleton — declared in ``config/market_rules.yaml``
#: (spec §20), loaded once per process, failing loudly on a missing or
#: invalid file.
_MARKET_RULES: SportMarketRules | None = None


def _market_rules() -> SportMarketRules:
    global _MARKET_RULES
    if _MARKET_RULES is None:
        _MARKET_RULES = load_market_rules("nba")
    return _MARKET_RULES
from sports.nba.features.registry import build_feature_contract


class NBAAdapter:
    """The ``SportAdapter`` implementation for the NBA."""

    sport_key = "nba"

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------

    def artifact_contract(self) -> ArtifactContract:
        """NBA artifact families pinned to the platform naming convention."""
        dated = [
            ("nba_moneyline_json", "nba_moneyline_v1_*.json", "json"),
            ("nba_calibration", "nba_calibration_*.json", "json"),
            ("nba_predictions_history", "nba_predictions_history_*.csv",
             "csv"),
            ("nba_power_rankings", "nba_power_rankings_*.csv", "csv"),
            ("nba_markets", "nba_run_engine_markets_*.csv", "csv"),
            ("nba_markets_monitor", "nba_run_engine_monitor_*.json", "json"),
            ("nba_player_matchup", "nba_player_matchup_*.json", "json"),
            ("nba_feature_json", "nba_feature_v1_*.json", "json"),
            ("nba_model_monitor", "nba_model_monitor_*.json", "json"),
            ("nba_shap_game", "nba_shap_game_*.csv", "csv"),
            ("nba_feature_drift", "nba_run_engine_feature_drift_*.csv",
             "csv"),
            ("nba_feature_coverage", "nba_run_engine_feature_coverage_*.csv",
             "csv"),
        ]
        exempt = [
            ("nba_oof_store", "nba_oof_moneyline_*.csv", "csv"),
            ("nba_fold_table", "nba_fold_table_*.csv", "csv"),
        ]
        families = [ArtifactFamily(family=n, pattern=p, ext=e,
                                   description="NBA frontend artifact "
                                               "(20-day retention)")
                    for n, p, e in dated]
        families += [ArtifactFamily(family=n, pattern=p, ext=e,
                                    date_stamped=False,
                                    retention_exempt=True,
                                    description="NBA backend artifact "
                                                "(retention-exempt)")
                     for n, p, e in exempt]
        return ArtifactContract(sport="nba", families=tuple(families))

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
    # Participant matchup box (MLB pitcher-panel layout)
    # ------------------------------------------------------------------

    def participant_box(self, game_row: pd.Series,
                        side: str) -> ParticipantBox:
        """The per-team top-scorer box for one game row.

        ``side`` is ``home`` or ``away``. Headline stats: PPG and APG
        over the player's LAST 10 strictly-prior games. A TBD box is
        returned when the player is unavailable or the aggregates are
        missing — never a fabricated name.
        """
        name = game_row.get(f"player_{side}_name")
        ppg = game_row.get(f"player_{side}_ppg")
        apg = game_row.get(f"player_{side}_apg")
        has_name = isinstance(name, str) and bool(name.strip())
        if not has_name or pd.isna(ppg) or pd.isna(apg):
            return ParticipantBox(role="player", name="", stats=(),
                                  is_tbd=True)
        stats = (
            ("PPG (last 10)", f"{float(ppg):.1f}"),
            ("APG (last 10)", f"{float(apg):.1f}"),
        )
        return ParticipantBox(role="player", name=str(name).strip(),
                              stats=stats, is_tbd=False)

    # ------------------------------------------------------------------
    # Market frame + probabilities (sport-specific schema)
    # ------------------------------------------------------------------

    def market_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize the NBA markets artifact into the adapter's frame.

        Keeps the sport-specific markets schema (kind / grids / outcome
        columns) and adds the shared display derivations. Never assigns
        MLB, NFL, or NHL field meanings to NBA columns."""
        out = raw.copy()
        if "kind" in out.columns:
            out["is_slate"] = out["kind"] == "slate"
        return out

    def market_probs(self, market_row: pd.Series) -> dict[str, float]:
        """Semantic market probabilities for one game row.

        TIE is structurally impossible in the NBA (OT resolves every
        game), so ``p_tie`` is never emitted — never zero-padded as a
        fake three-way market. ``p_push_<line>`` is included for integer
        lines where the wager can push.
        """
        p_home = float(market_row.get("p_home_win_derived"))
        p_away = float(market_row.get("p_away_win_derived"))
        probs: dict[str, float] = {"p_home": p_home, "p_away": p_away}
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
        """The settlement config for one of the NBA's market kinds, read
        from ``config/market_rules.yaml``. An undeclared kind raises
        ``KeyError`` — settlement semantics are never invented here."""
        return _market_rules().settlement_config(market_kind)
