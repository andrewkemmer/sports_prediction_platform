"""Sports Prediction Platform — Streamlit entry point.

Compatibility entry: the legacy repo launched the dashboard with
``streamlit run frontend/Home.py``; this rebuilt tree's canonical entry is
``frontend/app.py``. Both commands run the SAME shared navigation — this
module only exists so the legacy command (and any muscle memory) keeps
working unchanged. All layout lives in ``components/navigation.py``; the
five-page sidebar order contract is:

    Today's Games · Power Rankings · Calibration · Model Monitor ·
    Totals & Run Lines

Run from the repository root::

    streamlit run frontend/Home.py     # legacy command
    streamlit run frontend/app.py      # canonical entry
"""

import sys
from pathlib import Path

# Allow `streamlit run frontend/Home.py` from the repo root: the frontend
# package must be importable as ``frontend.*`` even when Streamlit runs the
# script directly from its own directory.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

import frontend.components.navigation as navigation  # noqa: E402

st.set_page_config(
    page_title="Sports Predictions",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation.run()
