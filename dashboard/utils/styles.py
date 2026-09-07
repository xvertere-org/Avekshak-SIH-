"""
Centralized styling, color tokens, and CSS helpers for SIH26054 Dashboard UI.

Enforces a high-contrast, professional aerospace console theme suitable for
technical evaluation and operational ground station monitoring.
"""

from typing import Dict, Optional
from dashboard.schemas.view_model import StatusLevel

# Semantic Color Palette — Reserved strictly for actual system and component state
STATUS_COLORS: Dict[StatusLevel, str] = {
    StatusLevel.HEALTHY: "#2ea043",       # Normal / Healthy Green
    StatusLevel.WARNING: "#d29922",       # Operational Warning Amber
    StatusLevel.CRITICAL: "#f85149",      # Alert Red
    StatusLevel.DEGRADED: "#db6d28",      # Degraded Orange
    StatusLevel.UNAVAILABLE: "#6e7681",   # Muted Steel Gray
    StatusLevel.UNKNOWN: "#8b949e",       # Neutral Gray
}

STATUS_BG_COLORS: Dict[StatusLevel, str] = {
    StatusLevel.HEALTHY: "rgba(46, 160, 67, 0.12)",
    StatusLevel.WARNING: "rgba(210, 153, 34, 0.12)",
    StatusLevel.CRITICAL: "rgba(248, 81, 73, 0.12)",
    StatusLevel.DEGRADED: "rgba(219, 109, 40, 0.12)",
    StatusLevel.UNAVAILABLE: "rgba(110, 118, 129, 0.10)",
    StatusLevel.UNKNOWN: "rgba(139, 148, 158, 0.10)",
}

# Channel Chart Line Styles
PLOT_COLORS = {
    "observed": "#58a6ff",        # Bright Blue for Live Telemetry
    "expected": "#3fb950",        # Green for Digital Twin Expected State
    "forecast": "#bc8cff",        # Purple for Future Forecast
    "forecast_bound": "rgba(188, 140, 255, 0.15)",
    "warning_limit": "#d29922",   # Amber dashed line
    "critical_limit": "#f85149",  # Red dotted line
    "grid": "#1e2430",
    "paper_bg": "#0d1117",
    "plot_bg": "#11151c",
    "text": "#c9d1d9",
}

AEROSPACE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Hide Streamlit default multipage pages list in favor of custom UI tabs */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Global Base */
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Restrained Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }

    /* Compact Demo Watermark */
    .demo-watermark-compact {
        background-color: rgba(210, 153, 34, 0.10);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.35);
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.8px;
        padding: 4px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 8px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Console Engineering Panel */
    .console-panel {
        background-color: #11151c;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    /* Dominant Primary Status Hero Container */
    .status-hero {
        background-color: #11151c;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    /* Section Subheadings */
    .section-label {
        font-size: 11px;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 4px;
    }

    /* Streamlit Native Metric Overrides */
    div[data-testid="stMetric"] {
        background-color: #11151c;
        border: 1px solid #21262d;
        border-radius: 4px;
        padding: 8px 12px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #8b949e !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #f0f6fc !important;
    }

    /* High-Density Engineering Table */
    .eng-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin: 8px 0;
        background-color: #11151c;
        border: 1px solid #21262d;
        border-radius: 6px;
        overflow: hidden;
    }

    .eng-table th {
        background-color: #161b22;
        color: #8b949e;
        text-align: left;
        padding: 8px 12px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        border-bottom: 1px solid #21262d;
    }

    .eng-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #1a202c;
        color: #e6edf3;
    }

    .eng-table tr:last-child td {
        border-bottom: none;
    }

    .eng-table tr:hover td {
        background-color: #161b22;
    }

    .eng-num {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* Restrained Status Pill */
    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border: 1px solid transparent;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Custom dark scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0e14;
    }
    ::-webkit-scrollbar-thumb {
        background: #21262d;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #30363d;
    }
</style>
"""


def render_status_badge(status: StatusLevel, label: Optional[str] = None) -> str:
    """Generate HTML markup for a restrained semantic status badge."""
    color = STATUS_COLORS.get(status, "#8b949e")
    bg = STATUS_BG_COLORS.get(status, "rgba(139, 148, 158, 0.10)")
    text = label if label is not None else status.value
    return (
        f'<span class="status-pill" style="color: {color}; background-color: {bg}; border-color: {color}44;">'
        f'{text}</span>'
    )

