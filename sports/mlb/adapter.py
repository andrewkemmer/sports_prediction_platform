"""MLB SportAdapter — the semantic bridge between the MLB pipeline's
sport-specific schemas and the shared frontend interfaces
(``core.contracts``).

Added in Phase r1/r2 of the §2 conformance rebuild (closes the 7.6-A
open finding: MLB was the only sport without a ``SportAdapter``
implementation; this module resolves that gap by construction — see
docs/TRACKER.md §22, "r0 rewrite of the record").

Market schemas stay sport-specific: MLB prices markets through the
per-game run engine (``models/market/run_engine.py``) — a discrete
run-count distribution with NO tie mass and pushes only on integer
total lines. TIE and PUSH remain distinct:

* ``moneyline`` — full_game scope, no tie, no push (a decided MLB game
  always has a winner);
* ``total`` — full_game, integer lines can push (``push_allowed=True``);
  half-point lines cannot;
* ``run_line`` — full_game, -1.5/+1.5 half-point lines cannot push.

The pitcher matchup panel mirrors the reference MLB dashboard fields
(sp_name/sp_era/sp_k9 per side, the ``sp_*`` slate columns). Unannounced
starters render as TBD with preserved missing values — never fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import load_market_rules
from core.config.sources import SportMarketRules
from core.contracts import (
    ArtifactContract,
    ArtifactFamily,
    ParticipantBox,
    SettlementConfig,
)
from sports.mlb.features.registry import build_feature_contract

#: Settlement rules singleton — declared in ``config/market_rules.yaml``
#: (spec §20), loaded once per process, failing loudly on a missing or
#: invalid file.
_MARKET_RULES: SportMarketRules | None = None


def _market_rules() -> SportMarketRules:
    global _MARKET_RULES
    if _MARKET_RULES is None:
        _MARKET_RULES = load_market_rules("mlb")
    return _MARKET_RULES


class MLBAdapter:
    """The ``SportAdapter`` implementation for MLB."""

    sport_key = "mlb"

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------

    def artifact_contract(self) -> ArtifactContract:
        """MLB artifact families pinned to the real emitted filenames."""
        dated = [
            ("todays_games", "todays_games_*.csv", "csv"),
            ("power_rankings", "power_rankings_*.csv", "csv"),
            ("calibration", "calibration_*.json", "json"),
            ("predictions_history", "predictions_history_*.csv", "csv"),
            ("model_monitor", "model_monitor_*.json", "json"),
            ("markets", "run_engine_markets_*.csv", "csv"),
            ("markets_meta", "run_engine_markets_*.meta.json", "json"),
            ("markets_monitor", "run_engine_monitor_*.json", "json"),
            ("shap_game", "shap_game_*.csv", "csv"),
            ("rolling_brier", "rolling_brier_*.json", "json"),
            ("features_metadata", "features_metadata_*.json", "json"),
        ]
        exempt = [
            ("mlb_oof_store", "mlb_oof_moneyline_*.csv", "csv"),
            ("mlb_fold_table", "mlb_fold_table_*.csv", "csv"),
        ]
        families = [ArtifactFamily(family=n, pattern=p, ext=e,
                                   description="MLB frontend artifact "
                                               "(20-day retention)")
                    for n, p, e in dated]
        families += [ArtifactFamily(family=n, pattern=p, ext=e,
                                    date_stamped=False,
                                    retention_exempt=True,
                                    description="MLB backend artifact "
                                                "(retention-exempt)")
                     for n, p, e in exempt]
        return ArtifactContract(sport="mlb", families=tuple(families))

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

        # evening/day split from the ET first-pitch hour
        hour = pd.Series(pd.NaT, index=out.index)
        if "start_time_utc" in out.columns:
            parsed = pd.to_datetime(out["start_time_utc"], errors="coerce",
                                    utc=True)
            hour = parsed.dt.tz_convert("America/New_York").dt.hour \
                if not parsed.isna().all() else hour
        hour = pd.to_numeric(hour, errors="coerce")
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
    # Pitcher matchup participant box (reference MLB dashboard fields)
    # ------------------------------------------------------------------

    def participant_box(self, game_row: pd.Series,
                        side: str) -> ParticipantBox:
        """The per-team starting-pitcher box for one game row.

        ``side`` is ``home`` or ``away``. Headline stats: ERA and K/9
        from the ``sp_*`` slate fields (the reference panel layout). A
        TBD box is returned when the starter is unannounced or the
        aggregates are missing — never a fabricated name.
        """
        name = game_row.get(f"sp_name_{side}")
        era = game_row.get(f"sp_era_{side}")
        k9 = game_row.get(f"sp_k9_{side}")
        has_name = isinstance(name, str) and bool(name.strip())
        if not has_name or pd.isna(era) or pd.isna(k9):
            return ParticipantBox(role="pitcher", name="", stats=(),
                                  is_tbd=True)
        stats = (
            ("ERA", f"{float(era):.2f}"),
            ("K/9", f"{float(k9):.2f}"),
        )
        return ParticipantBox(role="pitcher", name=str(name).strip(),
                              stats=stats, is_tbd=False)

    # ------------------------------------------------------------------
    # Market frame + probabilities (sport-specific schema)
    # ------------------------------------------------------------------

    def market_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize the MLB run-engine markets artifact into the
        adapter's frame.

        Keeps the sport-specific markets schema (kind / grids / outcome
        columns) and adds the shared display derivations. Never assigns
        another sport's field meanings to MLB columns."""
        out = raw.copy()
        if "kind" in out.columns:
            out["is_slate"] = out["kind"] == "slate"
        return out

    def market_probs(self, market_row: pd.Series) -> dict[str, float]:
        """Semantic market probabilities for one game row.

        ``p_home`` / ``p_away`` come from the derived moneyline — MLB
        carries NO tie mass (a decided game always has a winner; the
        pair sums to 1). ``p_push_total_<line>`` is included for integer
        total lines where the wager can push; half-point lines and run
        lines never push.
        """
        p_home = float(market_row.get("p_home_win_derived"))
        p_away = 1.0 - p_home
        probs: dict[str, float] = {"p_home": p_home, "p_away": p_away}
        # offered/derived per-line push columns (null on fair-only runs)
        for key, val in market_row.items():
            if isinstance(key, str) and key.startswith("p_push"):
                v = val
                if v is not None and not (isinstance(v, float)
                                          and np.isnan(v)):
                    probs[key] = float(v)
        for key in ("p_over_offered", "p_under_offered",
                    "p_push_total_offered"):
            v = market_row.get(key)
            if v is not None and not (isinstance(v, float)
                                      and np.isnan(v)):
                probs[key] = float(v)
        return probs

    # ------------------------------------------------------------------
    # Settlement configs (core.contracts.settlement_rules)
    # ------------------------------------------------------------------

    def settlement_config(self, market_kind: str) -> SettlementConfig:
        """The settlement config for one of MLB's market kinds, read from
        ``config/market_rules.yaml``. An undeclared kind raises
        ``KeyError`` — settlement semantics are never invented here."""
        return _market_rules().settlement_config(market_kind)
