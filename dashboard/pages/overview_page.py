"""
Overview Page for SIH26054 Dashboard.
Structured around an authoritative 5-10 second Operator/Judge Information Hierarchy:
1. ENGINE STATUS (Dominant Condition & Health Index)
2. CURRENT SITUATION (Active Anomaly & Diagnosed Fault)
3. OPERATOR ADVISORY (Authoritative Phase 13 Decision Support)
4. KEY TELEMETRY (Compact Canonical 7 Engineering Table)
5. OPERATING CONTEXT & PROGNOSTICS (Secondary Split)
6. TREND SPOTLIGHT (Focused comparison & deep-dive navigation)
"""

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Dict, Any, List, Optional
import streamlit as st
import plotly.graph_objects as go
from dashboard.schemas.view_model import DashboardViewModel, StatusLevel, CANONICAL_CHANNELS
from dashboard.components.telemetry_charts import build_channel_figure
from dashboard.utils.formatters import format_value, format_fault_name, format_health_index
from dashboard.utils.styles import STATUS_COLORS, PLOT_COLORS, render_status_badge


def render_overview_page(
    vm: DashboardViewModel,
    history_data: Optional[Dict[str, Dict[str, List[float]]]] = None,
):
    """Render authoritative 1. OVERVIEW section."""

    # =========================================================================
    # 1. DOMINANT PRIMARY ENGINE STATUS & SITUATION OVERVIEW
    # =========================================================================
    overall_color = STATUS_COLORS.get(vm.overview.overall_status, "#8b949e")
    hi_card = vm.overview.health_card
    anom_card = vm.overview.anomaly_card
    fault_card = vm.overview.fault_card

    # Diagnosis Probability from authoritative Phase 8 output
    diag_prob_str = (
        f"{vm.diagnostics.diagnostic_confidence * 100.0:.1f}%"
        if (vm.diagnostics.diagnostic_confidence is not None and vm.diagnostics.predicted_fault_class != "Unavailable")
        else "N/A"
    )

    hero_html = (
        f'<div style="background-color: #11151c; border: 1px solid #21262d; border-left: 5px solid {overall_color}; '
        f'border-radius: 6px; padding: 16px 20px; margin-bottom: 16px;">'
        f'<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 20px;">'
        # Left side: Primary Engine Health Condition
        f'<div style="flex: 1 1 300px;">'
        f'<div style="font-size: 11px; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px;">ENGINE CONDITION</div>'
        f'<div style="display: flex; align-items: baseline; gap: 12px; margin: 4px 0;">'
        f'<span style="font-size: 26px; font-weight: 800; color: {overall_color}; letter-spacing: -0.5px;">{vm.overview.overall_status.value}</span>'
        f'<span style="font-size: 13px; color: #8b949e; font-family: monospace;">(Health Index: <b style="color: #f0f6fc;">{hi_card.value}</b>)</span>'
        f'</div>'
        f'<div style="font-size: 12px; color: #8b949e;">Current state: <b style="color: #e6edf3;">{hi_card.subtext or "Nominal"}</b></div>'
        f'</div>'
        # Right side: Current Findings (Anomaly & Fault)
        f'<div style="flex: 1 1 380px; display: flex; gap: 24px; border-left: 1px solid #1e2430; padding-left: 20px;">'
        f'<div>'
        f'<div style="font-size: 11px; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px;">ACTIVE ANOMALY</div>'
        f'<div style="font-size: 16px; font-weight: 700; font-family: monospace; color: #f0f6fc; margin: 4px 0;">{anom_card.value}</div>'
        f'<div style="font-size: 11px; color: #8b949e;">{anom_card.subtext or "Persistence: 0"}</div>'
        f'</div>'
        f'<div>'
        f'<div style="font-size: 11px; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px;">DIAGNOSED FAULT</div>'
        f'<div style="font-size: 16px; font-weight: 700; color: #f0f6fc; margin: 4px 0;">{fault_card.value}</div>'
        f'<div style="font-size: 11px; color: #8b949e;">Diagnosis confidence: <b style="color: #58a6ff; font-family: monospace;">{diag_prob_str}</b></div>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    # =========================================================================
    # 2. OPERATOR DECISION SUPPORT ADVISORY (Authoritative Phase 13)
    # =========================================================================
    if vm.overview.advisory is not None:
        adv = vm.overview.advisory
        urgency_colors = {
            "LOW": "#388bfd",
            "MEDIUM": "#d29922",
            "HIGH": "#db6d28",
            "CRITICAL": "#f85149",
        }
        urg_color = urgency_colors.get(adv.urgency, "#8b949e")

        subsystem_line = (
            f'<span style="color: #8b949e; margin-left: 12px;">Affected Subsystem: <code>{adv.affected_subsystem}</code></span>'
            if adv.affected_subsystem
            else ""
        )

        adv_html = (
            f'<div style="background-color: #11151c; border-left: 4px solid {urg_color}; border: 1px solid #21262d; '
            f'border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
            f'<div style="font-size: 11px; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;">'
            f'DECISION-SUPPORT ADVISORY · ACTION CODE: <code style="color: #58a6ff;">{adv.action_code}</code>'
            f'{subsystem_line}'
            f'</div>'
            f'<span style="background-color: {urg_color}1a; color: {urg_color}; border: 1px solid {urg_color}44; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 700;">'
            f'URGENCY: {adv.urgency}'
            f'</span>'
            f'</div>'
            f'<div style="font-size: 14px; font-weight: 700; color: #f0f6fc; margin-bottom: 4px;">{adv.headline}</div>'
            f'<div style="font-size: 13px; color: #c9d1d9;"><b>Suggested action:</b> {adv.recommended_action}</div>'
            f'<div style="font-size: 10px; color: #6e7681; margin-top: 6px;">ℹ️ <i>{adv.disclaimer}</i></div>'
            f'</div>'
        )
        st.markdown(adv_html, unsafe_allow_html=True)

    # =========================================================================
    # 3. KEY TELEMETRY (Compact Canonical 7 Engineering Table)
    # =========================================================================
    st.markdown('<div class="section-label">PRIMARY PROPULSION TELEMETRY</div>', unsafe_allow_html=True)

    rows_html = []
    for ch in CANONICAL_CHANNELS:
        ch_model = vm.telemetry.channels.get(ch)
        if not ch_model:
            continue
        obs_val = format_value(ch_model.observed_value, decimals=1, unit=ch_model.unit)
        exp_val = format_value(ch_model.expected_value, decimals=1, unit=ch_model.unit)
        res_val = format_value(ch_model.residual, decimals=2, unit=ch_model.unit)
        st_badge = render_status_badge(ch_model.status)
        rows_html.append(
            f'<tr>'
            f'<td><b>{ch_model.display_name}</b> <span style="color: #6e7681; font-size: 11px;">({ch.upper()})</span></td>'
            f'<td class="eng-num" style="color: {PLOT_COLORS["observed"]};">{obs_val}</td>'
            f'<td class="eng-num" style="color: {PLOT_COLORS["expected"]};">{exp_val}</td>'
            f'<td class="eng-num" style="color: #f0f6fc;">{res_val}</td>'
            f'<td style="text-align: right;">{st_badge}</td>'
            f'</tr>'
        )

    table_html = (
        f'<table class="eng-table">'
        f'<thead><tr>'
        f'<th>Channel</th>'
        f'<th>Measured</th>'
        f'<th>Model expected</th>'
        f'<th>Difference</th>'
        f'<th style="text-align: right;">Status</th>'
        f'</tr></thead>'
        f'<tbody>'
        f'{"".join(rows_html)}'
        f'</tbody>'
        f'</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # =========================================================================
    # 4. OPERATING CONTEXT & PROGNOSTICS (Secondary Split)
    # =========================================================================
    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
    ctx_col, prog_col = st.columns([1, 1])

    with ctx_col:
        st.markdown('<div class="section-label">OPERATING CONDITIONS</div>', unsafe_allow_html=True)
        ctx_html = (
            f'<div class="console-panel" style="padding: 10px 14px; margin-bottom: 0;">'
            f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px;">'
            f'<div><span style="font-size: 11px; color: #8b949e;">THROTTLE:</span> <b class="eng-num" style="color: #f0f6fc; margin-left: 6px;">{format_value(vm.telemetry.throttle, decimals=1, unit="%")}</b></div>'
            f'<div><span style="font-size: 11px; color: #8b949e;">LOAD:</span> <b class="eng-num" style="color: #f0f6fc; margin-left: 6px;">{format_value(vm.telemetry.load, decimals=1, unit="%")}</b></div>'
            f'<div><span style="font-size: 11px; color: #8b949e;">ALTITUDE:</span> <b class="eng-num" style="color: #f0f6fc; margin-left: 6px;">{format_value(vm.telemetry.altitude, decimals=0, unit="m")}</b></div>'
            f'<div><span style="font-size: 11px; color: #8b949e;">OAT:</span> <b class="eng-num" style="color: #f0f6fc; margin-left: 6px;">{format_value(vm.telemetry.ambient_temp, decimals=1, unit="°C")}</b></div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(ctx_html, unsafe_allow_html=True)

    with prog_col:
        st.markdown('<div class="section-label">LIFE PREDICTION</div>', unsafe_allow_html=True)
        rul_card = vm.overview.rul_card
        rul_detail = rul_card.subtext or f"State: {vm.prognostics.rul_state}"
        prog_html = (
            f'<div class="console-panel" style="padding: 10px 14px; margin-bottom: 0;">'
            f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px;">'
            f'<div><span style="font-size: 11px; color: #8b949e;">MEDIAN RUL:</span> <b class="eng-num" style="color: #f0f6fc; margin-left: 6px;">{rul_card.value}</b></div>'
            f'<div><span style="font-size: 11px; color: #8b949e;">LIMITING:</span> <code style="color: #58a6ff; margin-left: 6px;">{vm.prognostics.limiting_factor or "NONE"}</code></div>'
            f'</div>'
            f'<div style="font-size: 11px; color: #6e7681; margin-top: 6px; font-family: monospace;">Status: {rul_detail}</div>'
            f'</div>'
        )
        st.markdown(prog_html, unsafe_allow_html=True)

    # =========================================================================
    # 5. TREND SPOTLIGHT (1-2 Key Comparison Charts)
    # =========================================================================
    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">KEY TELEMETRY TRENDS</div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns(2)

    # Chart 1: Cylinder Head Temperature (Thermal)
    with t_col1:
        cht_model = vm.telemetry.channels.get("cht")
        if cht_model:
            cht_hist = history_data.get("cht", {}) if history_data else {}
            fig_cht = build_channel_figure(
                model=cht_model,
                history_timestamps=cht_hist.get("timestamps"),
                history_observed=cht_hist.get("observed"),
                history_expected=cht_hist.get("expected"),
            )
            fig_cht.update_layout(height=180, margin=dict(l=35, r=15, t=25, b=25))
            st.plotly_chart(fig_cht, use_container_width=True, config={"displayModeBar": False})

    # Chart 2: Engine Speed (Mechanical)
    with t_col2:
        rpm_model = vm.telemetry.channels.get("rpm")
        if rpm_model:
            rpm_hist = history_data.get("rpm", {}) if history_data else {}
            fig_rpm = build_channel_figure(
                model=rpm_model,
                history_timestamps=rpm_hist.get("timestamps"),
                history_observed=rpm_hist.get("observed"),
                history_expected=rpm_hist.get("expected"),
            )
            fig_rpm.update_layout(height=180, margin=dict(l=35, r=15, t=25, b=25))
            st.plotly_chart(fig_rpm, use_container_width=True, config={"displayModeBar": False})

    # Footer navigation hints
    st.markdown(
        '<div style="font-size: 11px; color: #6e7681; border-top: 1px solid #21262d; padding-top: 8px; margin-top: 4px; display: flex; justify-content: space-between;">'
        '<span>View <b>Live Telemetry</b> for all seven channels and subsystem groupings.</span>'
        '<span>View <b>Diagnostics</b> for model evidence and physics checks.</span>'
        '</div>',
        unsafe_allow_html=True,
    )


