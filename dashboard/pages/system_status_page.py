"""
Data Quality & System Status Page for SIH26054 Dashboard.
Renders sensor availability matrix, stream integrity, and provenance.
"""

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.schemas.view_model import DashboardViewModel
from dashboard.components.data_quality_view import (
    render_quality_summary,
    render_sensor_status_matrix,
    render_provenance_card,
)


def render_system_status_page(vm: DashboardViewModel):
    """Render 5. DATA QUALITY / SYSTEM STATUS section."""
    st.markdown("### Data quality and system status")

    render_quality_summary(vm.data_quality)

    st.markdown("<hr style='border-color: #30363d; margin: 20px 0;' />", unsafe_allow_html=True)

    render_sensor_status_matrix(vm.data_quality)

    st.markdown("<hr style='border-color: #30363d; margin: 20px 0;' />", unsafe_allow_html=True)

    render_provenance_card(vm.data_quality)
