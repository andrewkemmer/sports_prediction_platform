"""NFL SportAdapter — the semantic bridge between the NFL pipeline's
sport-specific schemas and the shared frontend interfaces
(``core.contracts``).

Market schemas stay sport-specific (Phase 0 decision 5): NFL carries
spread/total offers and a three-way full-game moneyline with a real tie
mass. TIE and PUSH remain distinct:

* ``moneyline`` — full_game scope, ``tie_allowed=True`` (NFL full-game
  ties exist), no push concept;
* ``spread`` — full_game, integer lines can push (``push_allowed=True``),
  half-point lines cannot; a tied game is NOT automatically a spread push
  (settlement uses the final margin and the actual line);
* ``total`` — full_game, integer lines can push, half-point totals
  cannot.

The QB matchup panel mirrors the MLB pitcher panel layout with the
reference contract fields (name, passer rating and TD/game over the
starter's last 10 strictly-prior games). Unannounced starters render as
TBD with preserved missing values — never fabricated.
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
from sports.nfl.features.registry import build_feature_contract


class NFLAdapter:
    """The ``SportAdapter`` implementation for the NFL."""

    sport_key = "nfl"

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------

    def artifact_contract(self) -> ArtifactContract:
        """NFL artifact families pinned to the real reference filenames."""
        dated = [
            ("nfl_moneyline_json", "nfl_moneyline_v1_*.json", "json"),
            ("nfl_calibration", "nfl_calibration_*.json", "json"),
            ("nfl_predictions_history", "nfl_predictions_history_*.csv",
             "csv"),
            ("nfl_power_rankings", "nfl_power_rankings_*.csv", "csv"),
            ("nfl_markets", "nfl_run_engine_markets_*.csv", "csv"),
            ("nfl_markets_monitor", "nfl_run_engine_monitor_*.json", "json"),
            ("nfl_qb_matchup", "nfl_qb_matchup_*.json", "json"),
            ("nfl_feature_json", "nfl_feature_v1_*.json", "json"),
            ("nfl_model_monitor", "nfl_model_monitor_*.json", "json"),
            ("nfl_shap_game", "nfl_shap_game_*.csv", "csv"),
            ("nfl_feature_drift", "nfl_run_engine_feature_drift_*.csv",
             "csv"),
            ("nfl_feature_coverage", "nfl_run_engine_feature_coverage_*.csv",
             "csv"),
        ]
        exempt = [
            ("nfl_oof_store", "nfl_oof_moneyline_*.csv", "csv"),
            ("nfl_fold_table", "nfl_fold_table_*.csv", "csv"),
        ]
        families = [ArtifactFamily(family=n, pattern=p, ext=e,
                                   description="NFL frontend artifact "
                                               "(20-day retention)")
                    for n, p, e in dated]
        families += [ArtifactFamily(family=n, pattern=p, ext=e,
                                    date_stamped=False,
                                    retention_exempt=True,
                                    description="NFL backend artifact "
                                                "(retention-exempt)")
                     for n, p, e in exempt]
        return ArtifactContract(sport="nfl", families=tuple(families))

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
            status = np.where(out["game_state"] == "live", "Live", status)
        out["game_status"] = status

        # evening/day split from ET kickoff hour (nflverse gametime)
        hour = pd.to_numeric(
            out.get("gametime", pd.Series("", index=out.index))
            .astype(str).str.split(":").str[0], errors="coerce")
        out["evening_game"] = np.where(hour.isna(), np.nan,
                                       (hour >= 17).astype(float))
        out["day_game"] = np.where(hour.isna(), np.nan,
                                   (hour < 17).astype(float))

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
                [out["home_team"], out["away_team"]], default="TIE")
        if actual_winner is not None:
            out["model_correct"] = np.where(
                p_home.isna() | ~decided, np.nan,
                (out["model_pick"] == actual_winner).astype(float))
            fav = np.where(p_home >= 0.5, out.get("home_team"),
                           out.get("away_team"))
            out["is_upset"] = np.where(
                p_home.isna() | ~decided, np.nan,
                ((fav != actual_winner) & (actual_winner != "TIE")
                 ).astype(float))
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
    # QB matchup participant box (MLB pitcher-panel layout)
    # ------------------------------------------------------------------

    def participant_box(self, game_row: pd.Series,
                        side: str) -> ParticipantBox:
        """The per-team quarterback box for one game row.

        ``side`` is ``home`` or ``away``. Headline stats: passer rating
        and TD/game over the starter's LAST 10 strictly-prior games. A
        TBD box is returned when the starter is unannounced or the
        aggregates are missing — never a fabricated name.
        """
        name = game_row.get(f"qb_{side}_name")
        rating = game_row.get(f"qb_{side}_rating")
        tdp = game_row.get(f"qb_{side}_td_per_game")
        has_name = isinstance(name, str) and bool(name.strip())
        if not has_name or pd.isna(rating) or pd.isna(tdp):
            return ParticipantBox(role="qb", name="", stats=(),
                                  is_tbd=True)
        stats = (
            ("Rating (last 10)", f"{float(rating):.1f}"),
            ("TD/game (last 10)", f"{float(tdp):.1f}"),
        )
        return ParticipantBox(role="qb", name=str(name).strip(),
                              stats=stats, is_tbd=False)

    # ------------------------------------------------------------------
    # Market frame + probabilities (sport-specific schema)
    # ------------------------------------------------------------------

    def market_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize the NFL markets artifact into the adapter's frame.

        Keeps the sport-specific markets schema (kind / grids / outcome
        columns) and adds the shared display derivations. Never assigns
        MLB field meanings to NFL columns."""
        out = raw.copy()
        if "kind" in out.columns:
            out["is_slate"] = out["kind"] == "slate"
        return out

    def market_probs(self, market_row: pd.Series) -> dict[str, float]:
        """Semantic market probabilities for one game row.

        ``p_home`` / ``p_away`` come from the derived moneyline with the
        explicit three-way tie mass preserved: p_home + p_tie + p_away = 1
        (never renormalized away). ``p_push_<line>`` is included for
        integer lines where the wager can push.
        """
        p_home = float(market_row.get("p_home_win_derived"))
        p_tie = float(market_row.get("p_tie") or 0.0)
        p_away = float(market_row.get("p_away_win_derived"))
        probs: dict[str, float] = {"p_home": p_home, "p_away": p_away,
                                   "p_tie": p_tie}
        # offered lines (bookmaker columns are null on fair-only runs)
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
        """The settlement config for one of the NFL's market kinds."""
        if market_kind == "moneyline":
            # NFL full-game ties are real outcomes (three-way market).
            return SettlementConfig(settlement_scope="full_game",
                                    tie_allowed=True, push_allowed=False)
        if market_kind in ("spread", "total"):
            # Integer lines can push; half-point lines cannot (enforced by
            # core.markets regardless of this flag). A tied game is not a
            # spread push: settlement reads the final margin vs the line.
            return SettlementConfig(settlement_scope="full_game",
                                    tie_allowed=False, push_allowed=True)
        raise KeyError(
            f"unknown NFL market kind {market_kind!r} "
            f"(expected moneyline|spread|total)")
