"""Sports Prediction Platform — shared multi-sport Streamlit entry point.

Run from the repository root::

    streamlit run frontend/app.py

All navigation/layout logic lives in ``components/navigation.py``; the
sport registry in ``adapters/registry.py``; artifact loading in
``readers/loaders.py``; the five dashboard pages in ``components/``.
"""

import sys
from pathlib import Path

# Allow `streamlit run frontend/app.py` from the repo root: the frontend
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
