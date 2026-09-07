"""
Explainable AI (XAI) Component for SIH26054 Dashboard.
Renders Phase 12 evidence fusion, deterministic physics consistency checks,
and SHAP local attribution (only when supplied by backend).
"""

from typing import Dict, Any, List
import streamlit as st
import plotly.graph_objects as go
from dashboard.schemas.view_model import DiagnosticsViewModel
from dashboard.utils.formatters import format_percent, format_value
from dashboard.utils.styles import PLOT_COLORS


def render_xai_evidence(diag: DiagnosticsViewModel):
    """Render explainability and evidence fusion details."""
    st.markdown("#### Explainability & Evidence Fusion")

    if not diag.summary_explanation and not diag.physics_evidence and not diag.shap_top_features and not diag.recommended_operator_action:
        st.info("ℹ️ Explainability and evidence fusion outputs currently unavailable.")
        return

    # Narrative explanation and recommended operator action
    if diag.summary_explanation or diag.recommended_operator_action:
        rec_div = (
            f'<div style="margin-top: 10px; border-top: 1px solid #21262d; padding-top: 8px; font-size: 13px; color: #58a6ff;">'
            f'<b>Recommended Operator Action:</b> {diag.recommended_operator_action}</div>'
            if diag.recommended_operator_action
            else ""
        )
        exp_html = (
            f'<div style="background-color: #11151c; border-left: 4px solid #58a6ff; '
            f'border: 1px solid #21262d; border-radius: 4px; padding: 12px 16px; margin-bottom: 14px;">'
            f'<div style="font-size: 11px; color: #8b949e; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Summary Explanation</div>'
            f'<div style="font-size: 13px; color: #e6edf3; margin-top: 4px; line-height: 1.5;">{diag.summary_explanation or "Nominal operational evidence."}</div>'
            f'{rec_div}'
            f'</div>'
        )
        st.markdown(exp_html, unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    # 1. Deterministic Physics Consistency
    with col1:
        st.markdown("##### Deterministic Physics Evidence")
        if diag.physics_evidence:
            phys = diag.physics_evidence
            p_status = phys.get("status", diag.physics_evidence_status or "Unavailable")
            status_color = "#2ea043" if p_status in ("SUPPORTED", "CONSISTENT") else "#f0883e"
            reason = phys.get("consistency_reason", diag.physics_consistency_reason or "")
            supporting = phys.get("supporting_channels", [])
            conflicting = phys.get("conflicting_channels", [])

            sup_div = (
                f'<div style="font-size: 11px; color: #3fb950; margin-top: 4px;">Supporting: <code>{", ".join(supporting)}</code></div>'
                if supporting
                else ""
            )
            conf_div = (
                f'<div style="font-size: 11px; color: #f85149; margin-top: 2px;">Conflicting: <code>{", ".join(conflicting)}</code></div>'
                if conflicting
                else ""
            )

            phys_html = (
                f'<div style="background-color: #11151c; border: 1px solid #21262d; '
                f'border-radius: 4px; padding: 12px; margin-bottom: 8px;">'
                f'<div>Physics Status: <b style="color: {status_color}">{p_status}</b></div>'
                f'<div style="font-size: 12px; color: #8b949e; margin-top: 4px;">{reason}</div>'
                f'{sup_div}'
                f'{conf_div}'
                f'</div>'
            )
            st.markdown(phys_html, unsafe_allow_html=True)
        else:
            st.caption("Physics consistency assessment unavailable.")

        # Temporal Evidence if present
        if diag.temporal_evidence:
            st.markdown("##### Temporal Evidence")
            temp = diag.temporal_evidence
            temp_html = (
                f'<div style="background-color: #11151c; border: 1px solid #21262d; '
                f'border-radius: 4px; padding: 10px 12px; font-size: 12px; color: #8b949e;">'
                f'<div>Persistence: <code style="color: #f0f6fc;">{temp.get("persistence_status", "N/A")}</code></div>'
                f'<div>Degradation Trend: <code style="color: #f0f6fc;">{temp.get("degradation_trend", "N/A")}</code></div>'
                f'</div>'
            )
            st.markdown(temp_html, unsafe_allow_html=True)

    # 2. Local TreeSHAP Model Attribution
    with col2:
        st.markdown("##### ML Feature Attribution (TreeSHAP)")
        if diag.shap_top_features:
            names = [f.get("feature_name") or f.get("feature") or "unknown" for f in diag.shap_top_features]
            weights = [float(f.get("relative_weight", 0.0)) * 100.0 for f in diag.shap_top_features]

            fig = go.Figure(go.Bar(
                x=weights,
                y=names,
                orientation="h",
                marker=dict(color="#58a6ff"),
            ))
            fig.update_layout(
                title=dict(text="Relative Attribution Weight (%)", font=dict(size=11, color=PLOT_COLORS["text"])),
                margin=dict(l=10, r=10, t=25, b=20),
                height=180,
                paper_bgcolor=PLOT_COLORS["paper_bg"],
                plot_bgcolor=PLOT_COLORS["plot_bg"],
                font=dict(color=PLOT_COLORS["text"], size=10),
                xaxis=dict(gridcolor=PLOT_COLORS["grid"]),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            if diag.shap_disclaimer:
                st.caption(f"ℹ️ *{diag.shap_disclaimer}*")
        else:
            st.caption("TreeSHAP model attribution unavailable.")

        # Fused Evidence if present
        if diag.fused_evidence:
            st.markdown("##### Fused Evidence Summary")
            fused = diag.fused_evidence
            fused_html = (
                f'<div style="background-color: #11151c; border: 1px solid #21262d; '
                f'border-radius: 4px; padding: 10px 12px; font-size: 12px; color: #8b949e;">'
                f'<div>Composite Confidence: <b style="color: #58a6ff;">{fused.get("composite_confidence", "N/A")}</b></div>'
                f'<div>Primary Conflict: <code style="color: #f0f6fc;">{fused.get("primary_conflict", "NONE")}</code></div>'
                f'</div>'
            )
            st.markdown(fused_html, unsafe_allow_html=True)

