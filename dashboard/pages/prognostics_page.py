"""
Prognostics Page for SIH26054 Dashboard.
Renders Health Index, degradation rate/trend, Remaining Useful Life (RUL),
and future health trajectory forecasts.
"""

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Dict, Any, List, Optional
import streamlit as st
from dashboard.schemas.view_model import DashboardViewModel
from dashboard.components.prognostics_view import (
    render_health_prognostics,
    render_rul_panel,
    render_forecast_panel,
)


def render_prognostics_page(
    vm: DashboardViewModel,
    history_timestamps: Optional[List[float]] = None,
    history_hi: Optional[List[float]] = None,
):
    """Render 4. PROGNOSTICS section."""
    st.markdown("### Health and life prediction")

    render_health_prognostics(
        vm.prognostics,
        history_timestamps=history_timestamps,
        history_hi=history_hi,
    )

    st.markdown("<hr style='border-color: #30363d; margin: 20px 0;' />", unsafe_allow_html=True)

    render_rul_panel(vm.prognostics)

    st.markdown("<hr style='border-color: #30363d; margin: 20px 0;' />", unsafe_allow_html=True)

    render_forecast_panel(vm.prognostics)
