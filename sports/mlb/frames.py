"""Point-in-time wide candidate frame.

The single canonical game frame every downstream consumer (feature
registry, fold builder, OOF, artifact writers) reads. ``get_decided_frame``
encodes the decided-frame rules ONCE — post-game rows only, numeric game_pk
identity required, starter required, deterministic dedup, stable
chronological order — so the four-strike desync bug class from the
reference repo is structurally impossible.

``attach_slate_row`` builds pre-game rows for the prediction window by
carrying each team's/pitcher's latest point-in-time state forward (strictly
prior data only, same semantics the training frame used entering every
historical game).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _numeric_game_pk(games: pd.DataFrame) -> pd.Series | None:
    if "game_pk" not in games.columns:
        return None
    return pd.to_numeric(games["game_pk"], errors="coerce")


def _canon_pk_str(x: Any) -> str:
    """Dtype-immune game_pk string for fold signatures."""
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(x)


def get_decided_frame(games: pd.DataFrame) -> pd.DataFrame:
    """The ONE canonical decided frame every consumer must use.

    Rules (in order):
    1. Post-game only: ``home_win`` non-null.
    2. Identity required: ``game_pk`` parses as a number (int64 dtype).
    2b. Starter required: ``home_starter_id`` non-null when the column
        exists (slate rows never carry one).
    3. Deterministic dedup: one row per game_pk, latest game_date wins.
    4. Stable chronological order (mergesort) — within-day row order is
       part of the canonical contract (LightGBM subsampling is
       row-order-sensitive).
    """
    if games.empty:
        return games.copy()
    if "home_win" not in games.columns:
        raise ValueError("get_decided_frame: frame must carry 'home_win'")

    out = games[games["home_win"].notna()].copy()
    pk = _numeric_game_pk(out)
    if pk is not None:
        out = out[pk.notna()].copy()
        out["game_pk"] = pd.to_numeric(
            out["game_pk"], errors="coerce").astype("int64")
        if "home_starter_id" in out.columns:
            starter_ok = out["home_starter_id"].notna()
            n_filtered = int((~starter_ok).sum())
            if n_filtered:
                logger.warning(
                    "get_decided_frame: excluded %d rows with numeric game_pk "
                    "but no home_starter_id (slate/pregame leak)", n_filtered)
            out = out[starter_ok].copy()
        if out["game_pk"].duplicated().any():
            order = pd.to_datetime(out["game_date"], errors="coerce")
            out = (out.assign(_order=order)
                      .sort_values("_order", kind="mergesort")
                      .drop_duplicates(subset="game_pk", keep="last")
                      .drop(columns="_order"))
    out = out.sort_values("game_date", kind="mergesort")
    return out.reset_index(drop=True)


def fold_signature(decided: pd.DataFrame,
                   retrain_cadence_days: int = 7) -> str:
    """Identity of a decided frame: game_pk sequence + fold geometry.

    Two consumers must never build folds on different row sets; call sites
    assert agreement with this signature.
    """
    from core.folds import walk_forward_splits
    splits = walk_forward_splits(
        decided, retrain_cadence_days=retrain_cadence_days)
    seq = ([_canon_pk_str(x) for x in decided["game_pk"].tolist()]
           if "game_pk" in decided.columns else [])
    payload = []
    for s in splits:
        val = s.get("val_games")
        pks = (sorted(_canon_pk_str(x) for x in val["game_pk"].tolist())
               if val is not None else [])
        payload.append([
            int(s["fold_idx"]),
            str(pd.Timestamp(s["val_start"]).date()),
            str(pd.Timestamp(s["val_end"]).date()),
            bool(s.get("is_partial_tail", False)),
            pks,
        ])
    blob = json.dumps({"folds": payload, "sequence": seq},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def require_matching_signatures(left_name: str, left_sig: str | None,
                                right_name: str, right_sig: str | None) -> None:
    """Fail loudly naming BOTH sides when two consumers' fold signatures
    disagree. A None signature (consumer never ran in-process) is skipped."""
    if left_sig is None or right_sig is None:
        logger.info("fold signature: %s=%s %s=%s — side absent, nothing to "
                    "compare", left_name, left_sig, right_name, right_sig)
        return
    if left_sig != right_sig:
        raise AssertionError(
            f"decided-frame desync: fold signature mismatch — "
            f"{left_name}={left_sig} vs {right_name}={right_sig}. Every "
            f"consumer must derive its frame via get_decided_frame(games).")


# ---------------------------------------------------------------------------
# Elo / records (point-in-time, chronological)
# ---------------------------------------------------------------------------

ELO_K = 20
ELO_HOME_ADV = 65
ELO_REVERT_FACTOR = 1 / 3
ELO_START = 1500.0


def _row_year(row: pd.Series) -> int | None:
    gd = row.get("game_date")
    try:
        return pd.Timestamp(gd).year
    except (TypeError, ValueError):
        return None


def _revert_elo(elo: float) -> float:
    return ELO_START + (elo - ELO_START) * (1 - ELO_REVERT_FACTOR)


def compute_elo_entries(sorted_games: pd.DataFrame) -> pd.DataFrame:
    """Pre-game Elo for both sides over a chronologically sorted frame.

    The row's Elo values are the ratings BEFORE that game is applied
    (point-in-time); season boundaries revert toward the mean.
    """
    elos: dict[str, float] = {}
    prev_year: int | None = None
    home_out, away_out = [], []
    for _, row in sorted_games.iterrows():
        home, away = row["home_team"], row["away_team"]
        cur_year = _row_year(row)
        if prev_year is not None and cur_year is not None \
                and cur_year != prev_year:
            elos = {t: _revert_elo(e) for t, e in elos.items()}
        prev_year = cur_year or prev_year
        h_elo = elos.get(home, ELO_START)
        a_elo = elos.get(away, ELO_START)
        home_out.append(h_elo)
        away_out.append(a_elo)
        actual = row.get("home_win")
        if pd.isna(actual):
            continue
        actual = float(actual)
        exp_home = 1.0 / (1.0 + 10 ** ((a_elo - h_elo - ELO_HOME_ADV) / 400))
        elos[home] = h_elo + ELO_K * (actual - exp_home)
        elos[away] = a_elo + ELO_K * ((1 - actual) - (1 - exp_home))
    return pd.DataFrame({"home_elo": home_out, "away_elo": away_out})


def compute_season_records(sorted_games: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time season records/win% per side over a sorted frame."""
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    run_diff: dict[str, int] = {}
    season: int | None = None
    rows: list[dict] = []
    for _, row in sorted_games.iterrows():
        cur_year = _row_year(row)
        if season is not None and cur_year is not None and cur_year != season:
            wins, losses, run_diff = {}, {}, {}
        season = cur_year or season
        home, away = row["home_team"], row["away_team"]
        hw, hl = wins.get(home, 0), losses.get(home, 0)
        aw, al = wins.get(away, 0), losses.get(away, 0)
        hd, ad = run_diff.get(home, 0), run_diff.get(away, 0)
        n_h, n_a = hw + hl, aw + al
        rows.append({
            "home_wins": hw, "home_losses": hl,
            "away_wins": aw, "away_losses": al,
            "home_win_pct": (hw / n_h) if n_h else np.nan,
            "away_win_pct": (aw / n_a) if n_a else np.nan,
            "home_run_diff": hd, "away_run_diff": ad,
            "home_record": f"{hw}-{hl}", "away_record": f"{aw}-{al}",
        })
        if pd.isna(row.get("home_win")):
            continue
        hs, as_ = row.get("home_score"), row.get("away_score")
        if pd.isna(hs) or pd.isna(as_):
            continue
        hs, as_ = int(hs), int(as_)
        if hs > as_:
            wins[home] = hw + 1
            losses[away] = al + 1
        elif as_ > hs:
            wins[away] = aw + 1
            losses[home] = hl + 1
        run_diff[home] = hd + (hs - as_)
        run_diff[away] = ad + (as_ - hs)
    return pd.DataFrame(rows)


def _smoothed_win_pct(wins: pd.Series, losses: pd.Series,
                     prior_games: float = 90.0,
                     prior_pct: float = 0.5) -> pd.Series:
    """Empirical-Bayes win% shrunk toward .500 early in the season."""
    n = wins + losses
    raw = wins / n.replace(0, np.nan)
    smoothed = (raw * n + prior_pct * prior_games) / (n + prior_games)
    return smoothed.fillna(prior_pct)


def enrich_elo_and_records(games: pd.DataFrame,
                           *, rename_team_woba: bool = False) -> pd.DataFrame:
    """Fill point-in-time Elo + season records on a frame.

    The DuckDB feature factory omits Elo/records; this derives them strictly
    from games completed BEFORE each row (PIT-safe, chronological) and
    optionally renames ``team_woba_30g_*`` to the spec names ``woba_30g_*``.
    Frames that already carry these columns keep their supplied values.
    """
    df = games.copy()
    chrono_cols = {"game_date", "home_team", "away_team", "home_win"}
    if chrono_cols <= set(df.columns):
        dates = pd.to_datetime(df["game_date"], errors="coerce")
        order = dates.argsort(kind="stable")
        srt = df.iloc[order].reset_index(drop=True)
        back = np.argsort(order)  # inverse permutation -> original order

        if "home_elo" not in df.columns or "away_elo" not in df.columns:
            elo = compute_elo_entries(srt)
            for c in ("home_elo", "away_elo"):
                if c not in df.columns:
                    df[c] = pd.Series(elo[c].to_numpy()[back], index=df.index)

        if not all(c in df.columns for c in
                   ("home_wins", "home_losses", "away_wins", "away_losses")):
            rec = compute_season_records(srt)
            for c in ("home_wins", "home_losses", "away_wins", "away_losses",
                      "home_win_pct", "away_win_pct", "home_run_diff",
                      "away_run_diff", "home_record", "away_record"):
                if c not in df.columns:
                    df[c] = pd.Series(rec[c].to_numpy()[back], index=df.index)

        # Smoothed win% (the Elo companion the model trains on).
        if "home_win_pct" in df.columns:
            df["home_win_pct"] = _smoothed_win_pct(
                df["home_wins"], df["home_losses"])
            df["away_win_pct"] = _smoothed_win_pct(
                df["away_wins"], df["away_losses"])

    if rename_team_woba:
        df = df.rename(columns={
            "team_woba_30g_home": "woba_30g_home",
            "team_woba_30g_away": "woba_30g_away",
        })
    return df
