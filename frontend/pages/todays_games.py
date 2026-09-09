"""Shared game board — the single card component behind Today's Games.

Reproduces the MLB reference layout: badge strip, scoreboard, two team
rows with model moneyline probability bars, pre-game line, sport-specific
participant panel, venue line, outcome banner, and per-game SHAP
expander. Cards render identically for every sport; only the participant
panel content differs (registry-driven).

Never fabricates: TBD/null participants, null scores and missing SHAP
render as explicit placeholders exactly as the backend emitted them.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import sports_config
import utils

# The active sport's participant-matchup frame, set once per page render
# (JSON matchup sports only; MLB leaves it None — sp_* fields are inline).
_MATCHUP: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Card color helpers (reference parity)
# ---------------------------------------------------------------------------
def _accent(team: str, winner: str, coin: bool, pick: str, live: bool) -> str:
    if coin and not live:
        return utils.BLUE
    if winner == team:
        return utils.PRIMARY
    if live and pick == team:
        return utils.BLUE
    return utils.RED


def _pct_color(team: str, winner: str, pick: str, coin: bool) -> str:
    if coin:
        return utils.BLUE
    if winner == team:
        return utils.PRIMARY
    if pick == team:
        return "#F87171"
    return "#94A3B8"


def _bar_color(team: str, winner: str, pick: str, coin: bool) -> str:
    if coin:
        return utils.BLUE
    if winner == team:
        return utils.PRIMARY
    if pick == team:
        return "#EF4444"
    return "#334155"


def _winner_of(g: pd.Series) -> str:
    hs, as_ = g.get("home_score"), g.get("away_score")
    try:
        hs_n = None if hs is None or pd.isna(hs) else int(hs)
        a_n = None if as_ is None or pd.isna(as_) else int(as_)
    except (TypeError, ValueError):
        return ""
    if hs_n is None or a_n is None or hs_n == a_n:
        return ""
    return str(g["home_team"]) if hs_n > a_n else str(g["away_team"])


def _score_side(score, abbr: str, is_winner: bool) -> str:
    num = "" if score is None or pd.isna(score) else int(score)
    win_bar = (f'<span class="win-bar" style="background:{utils.PRIMARY};"></span>'
               if is_winner else "")
    return (f'<div class="side"><div class="num-wrap">{win_bar}'
            f'<span class="num">{num}</span></div>'
            f'<span class="abbr">{abbr}</span></div>')


def _team_row(name, team, rec, prob, accent, pct_color, bar_color, picked,
              trophy, is_home) -> str:
    pick_badge = '<span class="fb-pill pick">PICK</span>' if picked else ""
    trophy_html = " 🏆" if trophy else ""
    home_tag = ('<span class="fb-tag" style="font-size:0.7rem;">HOME</span>'
                if is_home else "")
    rec = "" if rec is None or (isinstance(rec, float) and pd.isna(rec)) else rec
    return (
        f'<div class="fb-team">'
        f'<span class="fb-accent" style="background:{accent};"></span>'
        f'<div><span class="name">{name}</span> <span class="sub">{team} {rec}</span> '
        f'{home_tag} {pick_badge}{trophy_html}</div>'
        f'<span class="pct" style="color:{pct_color};">{prob:.0%}</span></div>'
        f'<div class="fb-bar"><div class="fill" style="width:{prob:.0%};'
        f'background:{bar_color};"></div></div>'
    )


# ---------------------------------------------------------------------------
# Participant panel (sport-specific, registry-driven)
# ---------------------------------------------------------------------------
def _participant_box(name, stats, labels, games) -> str:
    """One side's participant panel. TBD/None renders explicitly — the
    backend's null is displayed as 'TBD' with '—' stats, never a guess."""
    if name is None or str(name).strip() in ("", "None", "nan"):
        name_html = '<div class="pname">TBD</div>'
        stats_html = '<div class="pstats">Awaiting lineup</div>'
    else:
        name_html = f'<div class="pname">{name}</div>'
        parts = []
        for v, label in zip(stats, labels):
            parts.append(f"{label} {'—' if v is None else f'{v:.2f}'.rstrip('0').rstrip('.')}" if False else
                         (f"{label} —" if v is None else f"{label} {v:.2f}"))
        n = f" (last {games})" if games else ""
        stats_html = f'<div class="pstats">{" · ".join(parts)}{n}</div>'
    return f'<div class="fb-participant">{name_html}{stats_html}</div>'


def _participants_html(g: pd.Series, sport: str) -> str:
    """The two participant boxes for a card, delegating to the shared pure
    renderer in utils (same code path the tests assert on)."""
    return utils.participant_box_html(sport, g, _MATCHUP)


# ---------------------------------------------------------------------------
# Banner + card
# ---------------------------------------------------------------------------
def _banner_html(status, is_final, is_live, winner, pick, correct, coin, upset,
                 home_team, away_team, g) -> str:
    home_name = g.get("home_team_name", home_team) or home_team
    away_name = g.get("away_team_name", away_team) or away_team
    winner_name = home_name if winner == home_team else (
        away_name if winner == away_team else "")
    if status == "Scheduled":
        first = utils.start_time_et(g.get("start_time_utc", ""))
        suffix = f" — {first}" if first else ""
        return (f'<div class="fb-banner blue">⏳ Pre-game{suffix} · '
                f'prediction locked at start</div>')
    if is_live:
        leader = winner or "—"
        suffix = f" · {leader} leading" if leader != "—" else ""
        return f'<div class="fb-banner blue">● LIVE{suffix}</div>'
    if coin:
        return (f'<div class="fb-banner amber">🪙 {winner_name} Won — '
                f'Coin Flip Game (50/50)</div>')
    if correct:
        return f'<div class="fb-banner green">✓ {winner_name} Won — Model Correct</div>'
    if upset:
        return (f'<div class="fb-banner red">X {winner_name} Won — Upset! '
                f'Model picked {pick}</div>')
    return f'<div class="fb-banner red">X {winner_name} Won — Model picked {pick}</div>'


def _card_html(g: pd.Series, sport: str) -> str:
    home_team, away_team = g["home_team"], g["away_team"]
    home_name = g.get("home_team_name", "") or home_team
    away_name = g.get("away_team_name", "") or away_team
    status = g["game_status"]
    is_final = status == "Final"
    is_live = status == "Live"
    is_scheduled = status == "Scheduled"

    hs, as_ = g.get("home_score"), g.get("away_score")
    h_score_n = None if hs is None or pd.isna(hs) else int(hs)
    a_score_n = None if as_ is None or pd.isna(as_) else int(as_)
    winner = _winner_of(g)

    p_home = float(g.get("home_win_prob_model") or 0.0)
    p_away = float(g.get("away_win_prob_model") or 0.0)
    pick = str(g.get("model_pick") or "")
    is_coin_flip = bool(g.get("is_coin_flip", False)) or pick == ""
    is_upset = bool(g.get("is_upset", False))
    correct = bool(g.get("model_correct") is True) if is_final else False

    # --- top badge strip ---
    day_tag = "☀ Day Game" if g.get("day_game") else "🌙 Night Game"
    center_pill, right_pills = "", ""
    if is_final and is_coin_flip:
        center_pill = '<span class="fb-pill coinflip">🪙 COIN FLIP</span>'
    elif is_final and is_upset:
        center_pill = '<span class="fb-pill upset">⚡ UPSET</span>'
    if is_live:
        right_pills = '<span class="fb-pill live">● LIVE</span>'
    elif is_scheduled:
        right_pills = '<span class="fb-pill final">PRE-GAME</span>'
    elif is_final:
        correct_pill = "" if is_coin_flip else (
            '<span class="fb-pill correct">✓ CORRECT PICK</span>' if correct
            else '<span class="fb-pill miss">X MISS</span>')
        right_pills = correct_pill + '<span class="fb-pill final">FINAL</span>'

    top = (f'<span class="fb-tag">{day_tag}</span>{center_pill}'
           f'<span class="spacer"></span>{right_pills}')

    # --- scoreboard ---
    mid = (utils.start_time_et(g.get("start_time_utc", "")) or "PREGAME"
           if is_scheduled else ("F" if is_final else "LIVE"))
    score = (
        f'<div class="fb-score">'
        f'{_score_side(a_score_n, away_team, is_winner=(winner == away_team))}'
        f'<span class="mid">{mid}</span>'
        f'{_score_side(h_score_n, home_team, is_winner=(winner == home_team))}'
        f'</div>'
    )

    # --- team rows + probability bars ---
    home_row = _team_row(
        home_name, home_team, g.get("home_record", ""), p_home,
        _accent(home_team, winner, is_coin_flip, pick, is_live),
        _pct_color(home_team, winner, pick, is_coin_flip),
        _bar_color(home_team, winner, pick, is_coin_flip),
        picked=(pick == home_team),
        trophy=(winner == home_team and is_final), is_home=True)
    away_row = _team_row(
        away_name, away_team, g.get("away_record", ""), p_away,
        _accent(away_team, winner, is_coin_flip, pick, is_live),
        _pct_color(away_team, winner, pick, is_coin_flip),
        _bar_color(away_team, winner, pick, is_coin_flip),
        picked=(pick == away_team),
        trophy=(winner == away_team and is_final), is_home=False)

    pregame = (f'<div class="fb-pregame">Pre-game: {home_team} {p_home:.0%} '
               f'vs {away_team} {p_away:.0%}</div>')

    participants = _participants_html(g, sport)
    start_et = utils.start_time_et(g.get("start_time_utc", ""))
    venue_raw = g.get("venue")
    venue = ""
    if venue_raw is not None and str(venue_raw).strip() not in ("", "nan", "None"):
        venue = (f'<div class="fb-venue">📍 {venue_raw}'
                 f'{f" · {start_et}" if start_et else ""}</div>')

    banner = _banner_html(status, is_final, is_live, winner, pick, correct,
                          is_coin_flip, is_upset, home_team, away_team, g)

    return (f'<div class="fb-card"><div class="fb-top">{top}</div>'
            f'{score}{home_row}{away_row}{pregame}{participants}{venue}'
            f'{banner}</div>')


# ---------------------------------------------------------------------------
# SHAP expander (shared)
# ---------------------------------------------------------------------------
def _shap_expander(sport: str, g: pd.Series) -> None:
    gid = str(g.get("game_id", ""))
    with st.expander(f"📈 SHAP Features — {gid}", expanded=False):
        shap_df = utils.load_shap(sport, gid)
        if shap_df.empty:
            st.caption("No SHAP file found for this game.")
            return
        chart = utils.shap_chart(shap_df)
        if chart is not None:
            utils.show_chart(chart)
        persp = ""
        if "perspective_team" in shap_df.columns:
            pt = shap_df["perspective_team"].dropna().astype(str)
            pt = pt[pt.str.strip() != ""]
            if not pt.empty and pt.iloc[0] not in ("HOME", "AWAY"):
                persp = f" · Viewing from {pt.iloc[0]}'s perspective"
        st.caption("Positive values increase the favored team's win "
                   "probability; negative decrease it." + persp)


# ---------------------------------------------------------------------------
# Date navigation (shared)
# ---------------------------------------------------------------------------
def _render_date_nav(valid: list[str], date_str: str) -> None:
    prev_d = next((d for d in reversed(valid) if d < date_str), None)
    next_d = next((d for d in valid if d > date_str), None)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if prev_d and st.button("← Prev", width="stretch",
                                key="date_prev"):
            st.session_state["selected_date"] = prev_d
            st.rerun()
    with c2:
        pretty = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        st.markdown(f"<div style='text-align:center;font-weight:800;"
                    f"color:{utils.TEXT};'>{pretty}</div>",
                    unsafe_allow_html=True)
    with c3:
        if next_d and st.button("Next →", width="stretch",
                                key="date_next"):
            st.session_state["selected_date"] = next_d
            st.rerun()


def _render_calendar(valid: list[str], date_str: str) -> None:
    """Compact month calendar with valid-date highlighting."""
    try:
        import calendar
        import datetime as dt

        d = dt.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    except (ValueError, TypeError):
        return
    valid_set = set(valid)
    st.caption("🗓 Dates with published games are highlighted.")
    cal = calendar.Calendar(firstweekday=6).monthdatescalendar(d.year, d.month)
    rows = ["Mon Tue Wed Thu Fri Sat Sun".split()]
    for week in cal:
        rows.append([f"{day.day}" if day.strftime("%Y%m%d") in valid_set
                     else (f"·{day.day}" if day.month == d.month else "")
                     for day in week])
    lines = [" | ".join(f"{c:>4}" for c in row) for row in rows]
    st.code("\n".join(lines), language=None)


# ---------------------------------------------------------------------------
# Page entry — shared across all four sports
# ---------------------------------------------------------------------------
def render() -> None:
    sport = utils.get_sport()
    utils.render_demo_banner(sport, ["games", "participant_matchup", "shap_game"])
    valid = utils.board_dates(sport)
    valid_set = set(valid)

    if st.session_state.get("_nav_sport") != sport or \
            "selected_date" not in st.session_state:
        st.session_state["selected_date"] = utils.nearest_valid_date(valid) or ""
        st.session_state["_nav_sport"] = sport
    date_str = st.session_state["selected_date"]

    if not valid:
        utils.no_data_state(
            f"No {sports_config.resolve_sport(sport)['label']} games published yet.",
            f"Run the {sport.upper()} pipeline to emit "
            f"sports/{sport}/data_delivery artifacts, then reload.")
        return
    if date_str not in valid_set:
        st.session_state["selected_date"] = utils.nearest_valid_date(valid) or valid[-1]
        st.rerun()
        return

    games = utils.load_games_for_date(sport, date_str)
    if games.empty:
        utils.no_data_state(
            f"No games found for {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}.",
            "Pick another highlighted date from the calendar.")
        return

    # Resolve the participant-matchup frame once for the card loop (JSON
    # sports; MLB resolves sp_* inline and this stays None).
    global _MATCHUP
    _MATCHUP = (utils.load_participant_matchup(sport)
                if sports_config.participant_spec(sport).get("source") != "inline"
                else None)

    cal = utils.load_calibration(sport) or {}
    record = cal.get("today_record", {}) if isinstance(cal.get("today_record"), dict) else {}
    wins, losses = record.get("wins", 0), record.get("losses", 0)
    completed = wins + losses
    acc = (wins / completed * 100) if completed else 0.0

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:2px;">
          <div style="color:#94A3B8;font-size:0.9rem;">· {len(games)} games</div>
          <span style="margin-left:auto;background:rgba(16,185,129,.18);color:#34D399;border-radius:999px;padding:2px 10px;font-size:0.8rem;font-weight:700;">
            ✓ {wins}-{losses} Latest day · {acc:.1f}% accuracy
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_date_nav(valid, date_str)
    _render_calendar(valid, date_str)

    # --- filter pills ---
    counts = {
        "All Games": len(games),
        "Final": int((games["game_status"] == "Final").sum()),
        "Scheduled": int((games["game_status"] == "Scheduled").sum()),
        "Live": int((games["game_status"] == "Live").sum()),
    }
    options = [o for o in ("All Games", "Final", "Scheduled", "Live")
               if counts.get(o)]
    fmt = {o: f"{o} ({counts[o]})" for o in options}
    selected = st.pills("Filter", options, format_func=lambda o: fmt[o],
                        key=f"game_filter_{sport}", selection_mode="single",
                        label_visibility="collapsed")
    filtered = games
    if selected == "Final":
        filtered = games[games["game_status"] == "Final"]
    elif selected == "Scheduled":
        filtered = games[games["game_status"] == "Scheduled"]
    elif selected == "Live":
        filtered = games[games["game_status"] == "Live"]

    st.divider()

    # --- game cards (two per row) + SHAP accordions ---
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for col, (_, g) in zip(cols, filtered.iloc[i: i + 2].iterrows()):
            with col:
                st.markdown(_card_html(g, sport), unsafe_allow_html=True)
                _shap_expander(sport, g)

    st.caption("Model outputs are point-in-time — only data available before "
               "each game's scheduled start was used.")


render()
