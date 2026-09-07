"""
Overview Page for SIH26054 Dashboard.
Renders mission executive summary, high-level KPI cards, and operational snapshot.
"""

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Dict, Any, List, Optional
import streamlit as st
from dashboard.schemas.view_model import DashboardViewModel
from dashboard.components.metric_card import render_metric_card
from dashboard.components.telemetry_charts import render_operating_context_cards
from dashboard.utils.formatters import format_value, format_fault_name
from dashboard.utils.styles import STATUS_COLORS, PLOT_COLORS, render_status_badge


def render_overview_page(
    vm: DashboardViewModel,
    history_data: Optional[Dict[str, Dict[str, List[float]]]] = None,
):
    """Render 1. OVERVIEW section."""
    st.markdown("### Operational Situation Overview")

    # Operator Decision Support Advisory (Consumed directly from Phase 13)
    if vm.overview.advisory is not None:
        adv = vm.overview.advisory
        urgency_colors = {
            "LOW": "#388bfd",
            "MEDIUM": "#d29922",
            "HIGH": "#db6d28",
            "CRITICAL": "#f85149",
        }
        urg_color = urgency_colors.get(adv.urgency, "#8b949e")

        st.markdown(
            f"""
            <div style="background-color: #161b22; border-left: 5px solid {urg_color}; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 11px; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;">
                        OPERATOR DECISION SUPPORT ADVISORY — ACTION CODE: <code>{adv.action_code}</code>
                    </div>
                    <span style="background-color: {urg_color}22; color: {urg_color}; border: 1px solid {urg_color}; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">
                        URGENCY: {adv.urgency}
                    </span>
                </div>
                <div style="font-size: 16px; font-weight: 700; color: #f0f6fc; margin-bottom: 6px;">
                    {adv.headline}
                </div>
                <div style="font-size: 13px; color: #c9d1d9; line-height: 1.5; margin-bottom: 8px;">
                    <b>Recommended Operator Action:</b> {adv.recommended_action}
                </div>
                {f'<div style="font-size: 12px; color: #8b949e; margin-bottom: 8px;"><b>Affected Subsystem:</b> <code>{adv.affected_subsystem}</code></div>' if adv.affected_subsystem else ''}
                <div style="font-size: 11px; color: #8b949e; border-top: 1px solid #21262d; padding-top: 8px; margin-top: 8px;">
                    ℹ️ <i>{adv.disclaimer}</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5 Key Operational Metrics (KPI Cards)
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        render_metric_card(vm.overview.health_card)
    with kpi_cols[1]:
        render_metric_card(vm.overview.rul_card)
    with kpi_cols[2]:
        render_metric_card(vm.overview.anomaly_card)
    with kpi_cols[3]:
        render_metric_card(vm.overview.fault_card)
    with kpi_cols[4]:
        render_metric_card(vm.overview.data_quality_card)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Flight Condition & Operating Context
    render_operating_context_cards(vm.telemetry)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Quick Telemetry Status Snapshot Table
    st.markdown("#### Primary Telemetry Snapshot")
    snap_cols = st.columns(7)
    for idx, (ch, ch_model) in enumerate(vm.telemetry.channels.items()):
        with snap_cols[idx % 7]:
            val_str = format_value(ch_model.observed_value, decimals=1, unit=ch_model.unit)
            st.markdown(
                f"""
                <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 10px; color: #8b949e; text-transform: uppercase;">{ch.upper()}</div>
                    <div style="font-size: 15px; font-weight: 700; color: #f0f6fc; margin: 4px 0;">{val_str}</div>
                    <div>{render_status_badge(ch_model.status)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
