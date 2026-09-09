"""Shared Calibration page — KPIs, reliability curves, daily summary.

Renders each sport's calibration JSON through one component: metric KPI
cards (AUC, Brier, logloss, calibration error — raw + calibrated),
the reliability (calibration-bucket) chart, and the recent daily summary
table. Missing artifact → explicit no-data state.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import sports_config
import utils


def _kpi(label: str, value: str, cap: str = "") -> str:
    cap_html = f'<div class="cap">{cap}</div>' if cap else ""
    return (f'<div class="fb-kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{cap_html}</div>')


def _kpi_row(metrics: dict) -> None:
    auc = metrics.get("auc")
    brier = metrics.get("brier")
    logloss = metrics.get("logloss")
    ece = metrics.get("ece", metrics.get("ece_calibrated"))
    cols = st.columns(4)
    with cols[0]:
        st.markdown(_kpi("AUC", f"{auc:.3f}" if auc is not None else "—",
                         f"calibrated {metrics.get('auc_calibrated'):.3f}"
                         if metrics.get("auc_calibrated") is not None else ""),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(_kpi("Brier", f"{brier:.4f}" if brier is not None else "—",
                         f"calibrated {metrics.get('brier_calibrated'):.4f}"
                         if metrics.get("brier_calibrated") is not None else ""),
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(_kpi("Log loss", f"{logloss:.4f}" if logloss is not None else "—",
                         f"calibrated {metrics.get('logloss_calibrated'):.4f}"
                         if metrics.get("logloss_calibrated") is not None else ""),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(_kpi("Calibration error", f"{ece:.4f}" if ece is not None else "—"),
                    unsafe_allow_html=True)


def _reliability_chart(buckets: list[dict]):
    """Reference reliability curve: predicted vs actual per bucket."""
    if not buckets:
        return None
    try:
        import altair as alt

        d = pd.DataFrame(buckets)
        for col in ("mean_predicted", "mean_actual", "count"):
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["mean_predicted", "mean_actual"])
        if d.empty:
            return None
        base = alt.Chart(d).encode(
            x=alt.X("mean_predicted:Q", title="Mean predicted win probability",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("mean_actual:Q", title="Observed win rate",
                    scale=alt.Scale(domain=[0, 1])),
        )
        line = base.mark_line(point=True, color=utils.PRIMARY)
        diag = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(color=utils.SLATE, strokeDash=[4, 4])
            .encode(x="x:Q", y="y:Q")
        )
        return (diag + line).properties(height=320)
    except Exception:
        return None


def render() -> None:
    sport = utils.get_sport()
    cfg = sports_config.resolve_sport(sport)
    st.subheader(f"{cfg['emoji']} {cfg['label']} Calibration")
    utils.render_demo_banner(sport, ["calibration_json"])

    cal = utils.load_calibration(sport)
    if cal is None:
        utils.no_data_state(
            f"No {cfg['label']} calibration artifact published yet.",
            f"Expected {sports_config.artifact_patterns(sport).get('calibration_json')} "
            f"under sports/{sport}/data_delivery.")
        return

    date = cal.get("date")
    n = cal.get("n_games")
    st.caption(f"Walk-forward out-of-sample evaluation · {n} games"
               + (f" · through {date}" if date else ""))

    _kpi_row(cal.get("metrics", {}) or {})

    st.divider()
    st.markdown("**Reliability curve** (pooled OOF)")
    chart = _reliability_chart(cal.get("calibration_buckets", []))
    if chart is not None:
        utils.show_chart(chart)
    else:
        st.caption("No calibration buckets in the artifact.")

    st.markdown("**Recent daily summary**")
    daily = cal.get("daily", []) or []
    if daily:
        recent = daily[-10:]
        th = "".join(f"<th>{h}</th>" for h in
                     ("Date", "Games", "Record", "Brier", "Log loss", "ECE"))
        trs = ""
        for d in recent:
            m = d.get("metrics", {}) or {}
            trs += (
                "<tr>"
                f"<td>{d.get('date', '')}</td>"
                f"<td>{d.get('n_games', '')}</td>"
                f"<td>{d.get('wins', 0)}-{d.get('losses', 0)}</td>"
                f"<td>{m.get('brier', '—')}</td>"
                f"<td>{m.get('logloss', '—')}</td>"
                f"<td>{m.get('ece', '—')}</td>"
                "</tr>"
            )
        st.markdown(
            f'<div class="fb-box"><table class="fb-table">'
            f"<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No daily summary in the artifact.")

    method = (cal.get("calibration", {}) or {}).get("method")
    if method:
        st.caption(f"Calibration method: {method} (fit on out-of-sample folds only).")


render()
