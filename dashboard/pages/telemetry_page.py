"""
Live Telemetry Page for SIH26054 Dashboard.
Renders the canonical 7 telemetry channels and operating context.
"""

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Dict, Any, List, Optional
import streamlit as st
from dashboard.schemas.view_model import DashboardViewModel
from dashboard.components.telemetry_charts import (
    render_canonical_telemetry_grid,
    render_operating_context_cards,
)


def render_telemetry_page(
    vm: DashboardViewModel,
    history_data: Optional[Dict[str, Dict[str, List[float]]]] = None,
):
    """Render 2. LIVE TELEMETRY section."""
    st.markdown("### Propulsion Telemetry Streams")

    # Operating context ribbon
    render_operating_context_cards(vm.telemetry)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # 7 Canonical channels comparison grid
    render_canonical_telemetry_grid(vm.telemetry, history_data=history_data)
