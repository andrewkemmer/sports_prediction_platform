"""Shared multi-sport navigation (extracted from the former Home.py).

Defines the five-page navigation used across every sport (the single
source of truth for the sidebar page order):

    Today's Games · Power Rankings · Calibration · Model Monitor ·
    Totals & Run Lines

The sidebar carries a registry-driven sport toggle (MLB / NFL / NHL /
NBA) above the dashboard nav. ``st.session_state["sport"]`` is the single
source of truth that every loader/page reads via
``readers.loaders.get_sport()``; all pages render through shared
components against the active sport's ``sports/<sport>/data_delivery``
artifacts. Adding a sport requires one registry entry — zero UI-code
changes.

``frontend/app.py`` calls :func:`run` after ``st.set_page_config``; the
page paths below resolve relative to ``frontend/`` (Streamlit resolves
them against the main script's directory).
"""

import streamlit as st

import frontend.adapters.registry as sports_config
import frontend.readers.loaders as utils


def run() -> None:
    """Render the sidebar (brand, sport toggle, freshness) and run the
    five-page navigation against the active sport."""
    # Set the sport default BEFORE the toggle renders; stale values are
    # normalized to a valid registry key so the picker always renders against
    # a real entry.
    st.session_state.setdefault("sport", sports_config.DEFAULT_SPORT)
    _sport_key = sports_config.normalize_sport_key(st.session_state["sport"])
    if _sport_key != st.session_state["sport"]:
        st.session_state["sport"] = _sport_key

    with st.sidebar:
        utils.render_brand_header()
        _picked = st.pills(
            "Sport",
            options=list(sports_config.SPORTS.keys()),
            format_func=lambda s: f"{sports_config.SPORTS[s]['emoji']} {sports_config.SPORTS[s]['label']}",
            selection_mode="single",
            default=st.session_state["sport"],
            key="sport_picker",
            label_visibility="collapsed",
        )
        if _picked is not None and _picked != st.session_state["sport"]:
            st.session_state["sport"] = _picked
            st.rerun()
        utils.render_source_note()
        utils.render_last_updated(utils.get_sport())
        st.divider()

    # Navigation — the literal list below is the sidebar-order contract; the
    # per-sport page set is resolved through the registry.
    pages = [
        st.Page("components/todays_games.py", title="Today's Games", icon="📅", url_path="todays-games", default=True),
        st.Page("components/power_rankings.py", title="Power Rankings", icon="🏆", url_path="power-rankings"),
        st.Page("components/calibration.py", title="Calibration", icon="📊", url_path="calibration"),
        st.Page("components/model_monitor.py", title="Model Monitor", icon="🛰️", url_path="model-monitor"),
        st.Page("components/market_panel.py", title=sports_config.MARKETS_PAGE_LABEL, icon="🎯", url_path="markets"),
    ]
    _sport = utils.get_sport()
    _sport_config = sports_config.resolve_sport(_sport)
    if sports_config.is_unknown_sport(_sport):
        st.warning(f"Unknown sport '{_sport}' — showing {_sport_config['label']}.")
    _active_url_paths = sports_config.active_page_url_paths(_sport)
    _active_pages = [
        page for page, url_path in zip(pages, sports_config.ALL_PAGE_URL_PATHS)
        if url_path in _active_url_paths
    ]
    if not _active_pages:
        _active_pages = list(pages)
    nav = st.navigation(_active_pages, position="sidebar")
    nav.run()
