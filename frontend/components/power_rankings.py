"""Shared Power Rankings page — one table component for all four sports.

Renders rank, team, Elo/record equivalents, recent form and home/away
splits straight from each sport's power-rankings artifact. Missing
artifact → explicit no-data state; nothing synthesized.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import frontend.adapters.registry as sports_config
import frontend.readers.loaders as utils


def _elo_color(elo: float, lo: float, hi: float) -> str:
    """Emerald → slate ramp across the league Elo range (display only)."""
    if hi <= lo:
        return utils.SLATE
    t = max(0.0, min(1.0, (elo - lo) / (hi - lo)))
    if t > 0.66:
        return utils.PRIMARY
    if t > 0.33:
        return "#93C5FD"
    return utils.SLATE


def render() -> None:
    sport = utils.get_sport()
    cfg = sports_config.resolve_sport(sport)
    st.subheader(f"{cfg['emoji']} {cfg['label']} Power Rankings")
    utils.render_demo_banner(sport, ["power_rankings"])

    df = utils.load_power_rankings(sport)
    if df is None or df.empty:
        utils.no_data_state(
            f"No {cfg['label']} power-rankings artifact published yet.",
            f"Expected {sports_config.artifact_patterns(sport).get('power_rankings')} "
            f"under sports/{sport}/data_delivery.")
        return

    elo = pd.to_numeric(df.get("elo"), errors="coerce")
    lo, hi = float(elo.min()) if elo.notna().any() else 0.0, \
        float(elo.max()) if elo.notna().any() else 0.0

    header = ("Rank", "Team", "Elo", "Record", "Win%", "L10", "Home%", "Away%")
    has_run_diff = "run_diff" in df.columns
    if has_run_diff:
        header = header[:5] + ("Diff",) + header[5:]

    rows = []
    for _, r in df.iterrows():
        name = r.get("team_name") or r.get("team") or ""
        team = r.get("team") or ""
        e = r.get("elo")
        try:
            e_v = float(e)
            e_cell = (f'<span class="fb-accent-cell"><span class="fb-accent-line" '
                      f'style="background:{_elo_color(e_v, lo, hi)};"></span>'
                      f'{e_v:.0f}</span>')
        except (TypeError, ValueError):
            e_cell = "—"
        row = [
            str(r.get("rank", "")),
            f'<span class="fb-accent-cell"><b>{name}</b> '
            f'<span style="color:{utils.SLATE};">{team}</span></span>',
            e_cell,
            str(r.get("record", "")),
            f"{float(r['pct']):.3f}" if pd.notna(r.get("pct")) else "—",
        ]
        if has_run_diff:
            rd = r.get("run_diff")
            rd_v = "" if rd is None or pd.isna(rd) else int(rd)
            row.append(f"+{rd_v}" if isinstance(rd_v, int) and rd_v > 0
                       else str(rd_v if rd_v != "" else "—"))
        row += [
            str(r.get("l10", "") or "—"),
            f"{float(r['home_pct']):.3f}" if pd.notna(r.get("home_pct")) else "—",
            f"{float(r['away_pct']):.3f}" if pd.notna(r.get("away_pct")) else "—",
        ]
        rows.append(row)

    th = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                  for row in rows)
    st.markdown(
        f'<div class="fb-box"><table class="fb-table">'
        f"<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.caption("Elo ratings are model-internal strength scores; record, L10 "
               "form and home/away splits come straight from results already "
               "played — nothing here uses future information.")


render()
