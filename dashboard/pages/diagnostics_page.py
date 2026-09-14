"""
Diagnostics Page for SIH26054 Dashboard.
Renders anomaly detection breakdown, fault diagnosis probabilities,
digital twin state residuals, and explainability evidence.
"""

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.schemas.view_model import DashboardViewModel
from dashboard.components.diagnostics_view import (
    render_anomaly_diagnostics,
    render_fault_classification,
    render_residual_table,
)
from dashboard.components.xai_view import render_xai_evidence


def render_diagnostics_page(vm: DashboardViewModel):
    """Render 3. DIAGNOSTICS section."""
    st.markdown("### Engine Diagnostics")

    col1, col2 = st.columns([1, 1])
    with col1:
        render_anomaly_diagnostics(vm.diagnostics)
    with col2:
        render_residual_table(vm.diagnostics)

    st.markdown("<hr style='border-color: #30363d; margin: 20px 0;' />", unsafe_allow_html=True)

    render_fault_classification(vm.diagnostics)

    st.markdown("<hr style='border-color: #30363d; margin: 20px 0;' />", unsafe_allow_html=True)

    render_xai_evidence(vm.diagnostics)
