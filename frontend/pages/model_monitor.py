"""Shared Model Monitor page — health, ensemble, drift, rolling Brier.

Renders each sport's model_monitor artifact through one component: health
KPIs, the ensemble member table, feature drift/coverage tables, the
rolling Brier chart and the version history. Missing artifact → explicit
no-data state.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import sports_config
import utils


def _status_pill(status: str) -> str:
    s = str(status or "").lower()
    cls = {"ok": "ok", "healthy": "ok", "stable": "ok",
           "warn": "warn", "watch": "warn", "drifting": "warn",
           "alert": "alert", "drifted": "alert"}.get(s, "warn")
    return f'<span class="fb-status-pill {cls}">{status or "—"}</span>'


def _health_kpis(mon: dict) -> None:
    m = mon.get("metrics", {}) or {}
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f'<div class="fb-kpi"><div class="label">Last retrained</div>'
            f'<div class="value" style="font-size:1.1rem;">'
            f'{mon.get("last_retrained") or "—"}</div>'
            f'<div class="cap">next: {mon.get("next_retrain") or "—"}</div></div>',
            unsafe_allow_html=True)
    with cols[1]:
        brier = m.get("brier")
        baseline = mon.get("brier_baseline")
        delta = (brier - baseline) if (brier is not None and baseline is not None) else None
        cap = (f'{mon.get("brier_baseline_label") or "baseline"}: {baseline:.4f}'
               if baseline is not None else "")
        color = utils.PRIMARY if (delta is not None and delta <= 0) else utils.RED
        brier_disp = f"{brier:.4f}" if brier is not None else "—"
        st.markdown(
            f'<div class="fb-kpi"><div class="label">OOF Brier</div>'
            f'<div class="value" style="color:{color};">{brier_disp}</div>'
            f'<div class="cap">{cap}</div></div>',
            unsafe_allow_html=True)
    with cols[2]:
        auc = m.get("auc")
        auc_disp = f"{auc:.3f}" if auc is not None else "—"
        st.markdown(
            f'<div class="fb-kpi"><div class="label">OOF AUC</div>'
            f'<div class="value">{auc_disp}</div></div>',
            unsafe_allow_html=True)
    with cols[3]:
        drift = mon.get("drift_summary", {}) or {}
        n_alerts = drift.get("alerts", len([1 for f in mon.get("feature_drift", [])
                                            if str(f.get("status", "")).lower()
                                            in ("alert", "drifted")]))
        n_warn = drift.get("warnings", 0)
        pill = _status_pill("ok" if not n_alerts else "alert")
        st.markdown(
            f'<div class="fb-kpi"><div class="label">Feature drift</div>'
            f'<div class="value">{n_alerts} alerts</div>'
            f'<div class="cap">{n_warn} warnings {pill}</div></div>',
            unsafe_allow_html=True)


def _rolling_brier_chart(series: list[dict]):
    if not series:
        return None
    try:
        import altair as alt

        d = pd.DataFrame(series)
        if "date" not in d.columns or "brier" not in d.columns:
            return None
        d["brier"] = pd.to_numeric(d["brier"], errors="coerce")
        d = d.dropna(subset=["brier"])
        if d.empty:
            return None
        chart = (
            alt.Chart(d)
            .mark_line(color=utils.BLUE, point=True)
            .encode(
                x=alt.X("date:N", title="Date", sort=None),
                y=alt.Y("brier:Q", title="Rolling OOF Brier"),
            )
            .properties(height=280)
        )
        return chart
    except Exception:
        return None


def render() -> None:
    sport = utils.get_sport()
    cfg = sports_config.resolve_sport(sport)
    st.subheader(f"{cfg['emoji']} {cfg['label']} Model Monitor")
    utils.render_demo_banner(sport, ["model_monitor"])

    mon = utils.load_model_monitor(sport)
    if mon is None:
        utils.no_data_state(
            f"No {cfg['label']} model-monitor artifact published yet.",
            f"Expected {sports_config.artifact_patterns(sport).get('model_monitor')} "
            f"under sports/{sport}/data_delivery.")
        return

    _health_kpis(mon)

    st.divider()
    st.markdown("**Ensemble members** (walk-forward OOF)")
    ensemble = mon.get("ensemble", []) or []
    if ensemble:
        th = "".join(f"<th>{h}</th>" for h in
                     ("Member", "Weight", "AUC", "Brier", "Log loss", "n_eval"))
        trs = ""
        for e in ensemble:
            w = e.get("weight")
            w_disp = f"{float(w):.3f}" if w is not None else "—"
            trs += (
                "<tr>"
                f"<td><b>{e.get('name', '—')}</b></td>"
                f"<td>{w_disp}</td>"
                f"<td>{e.get('auc', '—')}</td>"
                f"<td>{e.get('brier', '—')}</td>"
                f"<td>{e.get('logloss', '—')}</td>"
                f"<td>{e.get('n_eval', '—')}</td>"
                "</tr>"
            )
        st.markdown(
            f'<div class="fb-box"><table class="fb-table">'
            f"<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>",
            unsafe_allow_html=True)
    else:
        st.caption("No ensemble table in the artifact.")

    st.markdown("**Rolling Brier**")
    chart = _rolling_brier_chart(mon.get("rolling_brier", []))
    if chart is not None:
        utils.show_chart(chart)
    else:
        st.caption("No rolling-Brier series in the artifact.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Feature drift (PSI)**")
        drift = mon.get("feature_drift", []) or []
        if drift:
            d = pd.DataFrame(drift)
            keep = [c for c in ("feature", "psi", "status") if c in d.columns]
            st.dataframe(d[keep], hide_index=True, width="stretch",
                         height=min(420, 40 + 28 * len(d)))
        else:
            st.caption("No feature-drift table in the artifact.")
    with right:
        st.markdown("**Feature coverage**")
        cov = mon.get("feature_coverage", []) or []
        if cov:
            d = pd.DataFrame(cov)
            keep = [c for c in ("feature", "pct_nonnull", "status")
                    if c in d.columns]
            st.dataframe(d[keep], hide_index=True, width="stretch",
                         height=min(420, 40 + 28 * len(d)))
        else:
            st.caption("No feature-coverage table in the artifact.")

    st.markdown("**Version history**")
    vh = mon.get("version_history", []) or []
    if vh:
        th = "".join(f"<th>{h}</th>" for h in
                     ("Version", "Date", "AUC", "Brier", "Log loss", "Note"))
        trs = ""
        for v in vh:
            trs += (
                "<tr>"
                f"<td>{v.get('version', '—')}</td>"
                f"<td>{v.get('date', '—')}</td>"
                f"<td>{v.get('auc', '—')}</td>"
                f"<td>{v.get('brier', '—')}</td>"
                f"<td>{v.get('logloss', '—')}</td>"
                f"<td>{v.get('note', '—')}</td>"
                "</tr>"
            )
        st.markdown(
            f'<div class="fb-box"><table class="fb-table">'
            f"<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>",
            unsafe_allow_html=True)
    else:
        st.caption("No version history in the artifact.")

    if mon.get("upset_note"):
        st.caption(str(mon["upset_note"]))


render()
