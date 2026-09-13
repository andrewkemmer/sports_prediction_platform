"""Shared Totals & Run Lines page (market view) for all four sports.

The page title is identical for every sport; content structure adapts to
the sport's market artifact (MLB's NB run-engine ladder grid vs the
NFL/NHL/NBA fair-line + market-line rows). All components derive from the
shared market semantic view produced by the loader; missing artifacts →
explicit no-data states. Push probabilities are displayed explicitly
wherever the artifact carries them.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

import frontend.adapters.registry as sports_config
import frontend.readers.loaders as utils


def _num(v):
    try:
        x = float(v)
        return x if pd.notna(x) else None
    except (TypeError, ValueError):
        return None


def _fmt(v, spec="{:.3f}") -> str:
    x = _num(v)
    return "—" if x is None else spec.format(x)


def _kpi(label: str, value: str, cap: str = "") -> str:
    cap_html = f'<div class="cap">{cap}</div>' if cap else ""
    return (f'<div class="fb-kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{cap_html}</div>')


# ---------------------------------------------------------------------------
# MLB: run-engine totals ladder grid
# ---------------------------------------------------------------------------
def _render_mlb_totals_grid(markets: pd.DataFrame) -> None:
    st.markdown("**Totals ladder** (p_over / p_push per line)")
    grid = sorted(
        {c.replace("p_over_", "") for c in markets.columns
         if c.startswith("p_over_")},
        key=lambda s: float(s))
    rows = []
    for _, r in markets.iterrows():
        row = [f"{r.get('game_pk', '')}", str(r.get("game_date", ""))[:10],
               f"{r.get('home_team', '')} vs {r.get('away_team', '')}"]
        for line in grid:
            row.append(_fmt(r.get(f"p_over_{line}"), "{:.1%}"))
        rows.append(row)
    header = ["Game", "Date", "Matchup"] + [f"O {g}" for g in grid]
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                  for row in rows)
    st.markdown(
        f'<div class="fb-box" style="overflow-x:auto;"><table class="fb-table">'
        f"<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>",
        unsafe_allow_html=True)

    st.markdown("**Push probabilities** (whole lines only)")
    push_lines = sorted(
        {c.replace("p_push_", "") for c in markets.columns
         if c.startswith("p_push_")},
        key=lambda s: float(s))
    if push_lines:
        push_rows = []
        for _, r in markets.iterrows():
            row = [f"{r.get('game_pk', '')}",
                   f"{r.get('home_team', '')} vs {r.get('away_team', '')}"]
            row += [_fmt(r.get(f"p_push_{l}"), "{:.1%}") for l in push_lines]
            push_rows.append(row)
        th = "".join(f"<th>{h}</th>" for h in
                     ["Game", "Matchup"] + [f"Push {l}" for l in push_lines])
        trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                      for row in push_rows)
        st.markdown(
            f'<div class="fb-box" style="overflow-x:auto;"><table class="fb-table">'
            f"<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>",
            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# NFL / NHL / NBA: fair-line rows with market lines and push semantics
# ---------------------------------------------------------------------------
def _render_lines_rows(markets: pd.DataFrame, sport: str) -> None:
    st.markdown("**Fair lines & market offer**")
    cols = ["game_id", "gameday", "home_team", "away_team", "mu_total",
            "fair_total", "total_line", "p_over_offered", "p_push_total_offered",
            "mu_margin", "fair_spread", "spread_line", "p_cover_offered",
            "p_push_offered"]
    keep = [c for c in cols if c in markets.columns]
    if keep:
        d = markets[keep].copy()
        for c in ("mu_total", "fair_total", "mu_margin", "fair_spread"):
            if c in d.columns:
                d[c] = d[c].map(lambda v: _fmt(v, "{:.1f}"))
        for c in ("p_over_offered", "p_push_total_offered", "p_cover_offered",
                  "p_push_offered"):
            if c in d.columns:
                d[c] = d[c].map(lambda v: _fmt(v, "{:.1%}"))
        st.dataframe(d, hide_index=True, width="stretch",
                     height=min(460, 40 + 28 * max(1, len(d))))
    else:
        st.caption("No market rows in the artifact.")

    st.markdown("**Settled diagnostics** (OOF rows only)")
    diag_cols = [c for c in ("game_id", "p_over_fair", "y_over_fair",
                             "p_under_fair", "y_under_fair", "p_push_total_offered",
                             "p_cover_fair", "y_cover_fair", "y_push_spread_fair")
                 if c in markets.columns]
    if diag_cols:
        settled = markets[markets.get("home_score").notna()] \
            if "home_score" in markets.columns else markets.iloc[0:0]
        d = settled[diag_cols].copy()
        if d.empty:
            st.caption("No settled market rows yet (slate is undecided).")
        else:
            for c in diag_cols[1:]:
                if c.startswith("p_"):
                    d[c] = d[c].map(lambda v: _fmt(v, "{:.1%}"))
                else:
                    d[c] = d[c].map(lambda v: _fmt(v, "{:.0f}"))
            st.dataframe(d, hide_index=True, width="stretch",
                         height=min(420, 40 + 28 * max(1, len(d))))


def _render_meta(meta: Optional[dict]) -> None:
    if not meta:
        return
    st.divider()
    cols = st.columns(4)
    n_rows = meta.get("n_rows")
    n_oof = meta.get("n_oof")
    n_slate = meta.get("n_slate")
    seed = (meta.get("config", {}) or {}).get("seed") or meta.get("seed")
    with cols[0]:
        st.markdown(_kpi("Market rows", str(n_rows) if n_rows is not None else "—"),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(_kpi("OOF rows", str(n_oof) if n_oof is not None else "—"),
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(_kpi("Slate rows", str(n_slate) if n_slate is not None else "—"),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(_kpi("MC draws / seed",
                         str(meta.get("n_draws", "—") or "—"),
                         f"seed {seed}" if seed is not None else ""),
                    unsafe_allow_html=True)


def _render_monitor(sport: str) -> None:
    mon = utils.load_markets_monitor(sport)
    if not mon:
        st.caption("No markets-monitor artifact published.")
        return
    st.divider()
    st.markdown("**Markets monitor**")
    persisted = mon.get("markets_persisted")
    err = mon.get("markets_persist_error")
    pill = "ok" if (persisted and not err) else "alert"
    st.markdown(
        f'<div class="fb-box">Status: <span class="fb-status-pill {pill}">'
        f'{"persisted" if persisted else "not persisted"}</span>'
        + (f' · error: {err}' if err else "")
        + f' · date: {mon.get("date", "—")}</div>',
        unsafe_allow_html=True)
    metrics = mon.get("market_metrics", {}) or {}
    if metrics:
        items = list(metrics.items())[:8]
        th = "".join(f"<th>{h}</th>" for h in ("Metric", "Value"))
        trs = "".join(f"<tr><td>{k}</td><td>{_fmt(v)}</td></tr>"
                      for k, v in items)
        st.markdown(
            f'<div class="fb-box"><table class="fb-table">'
            f"<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>",
            unsafe_allow_html=True)


def render() -> None:
    sport = utils.get_sport()
    cfg = sports_config.resolve_sport(sport)
    st.subheader(f"{cfg['emoji']} Totals & Run Lines")
    utils.render_demo_banner(sport, ["markets"])

    markets, meta = utils.load_markets(sport)
    if markets is None or markets.empty:
        utils.no_data_state(
            f"No {cfg['label']} market artifacts published yet.",
            f"Expected {sports_config.artifact_patterns(sport).get('markets')} "
            f"under sports/{sport}/data_delivery.")
        return

    slate = markets[markets.get("kind", "").astype(str) == "slate"] \
        if "kind" in markets.columns else markets
    st.caption(f"{len(markets)} market rows"
               + (f" · {len(slate)} on the current slate" if "kind" in markets.columns else ""))

    kind = sports_config.market_kind(sport)
    if kind == "mlb_grid":
        _render_mlb_totals_grid(markets)
    else:
        _render_lines_rows(markets, sport)

    _render_meta(meta)
    _render_monitor(sport)
    st.caption("Market probabilities are model-derived from the same OOF "
               "engine as the moneyline; push columns are shown explicitly "
               "where the artifact prices them.")


render()
