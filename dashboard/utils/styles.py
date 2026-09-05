"""
Centralized styling, color tokens, and CSS helpers for SIH26054 Dashboard UI.

Enforces a high-contrast, professional aerospace dark theme suitable for
technical evaluation and operational ground station monitoring.
"""

from typing import Dict, Optional
from dashboard.schemas.view_model import StatusLevel

# Semantic Color Palette
STATUS_COLORS: Dict[StatusLevel, str] = {
    StatusLevel.HEALTHY: "#2ea043",       # Aerospace Green
    StatusLevel.WARNING: "#d29922",       # Operational Amber
    StatusLevel.CRITICAL: "#f85149",      # Alert Red
    StatusLevel.DEGRADED: "#db6d28",      # Degraded Orange
    StatusLevel.UNAVAILABLE: "#6e7681",   # Muted Steel Gray
    StatusLevel.UNKNOWN: "#8b949e",       # Neutral Gray
}

STATUS_BG_COLORS: Dict[StatusLevel, str] = {
    StatusLevel.HEALTHY: "rgba(46, 160, 67, 0.15)",
    StatusLevel.WARNING: "rgba(210, 153, 34, 0.15)",
    StatusLevel.CRITICAL: "rgba(248, 81, 73, 0.15)",
    StatusLevel.DEGRADED: "rgba(219, 109, 40, 0.15)",
    StatusLevel.UNAVAILABLE: "rgba(110, 118, 129, 0.12)",
    StatusLevel.UNKNOWN: "rgba(139, 148, 158, 0.12)",
}

# Channel Chart Line Styles
PLOT_COLORS = {
    "observed": "#58a6ff",        # Bright Cyan/Blue for Live Telemetry
    "expected": "#3fb950",        # Green for Digital Twin Expected State
    "forecast": "#bc8cff",        # Purple for TimesFM Future Forecast
    "forecast_bound": "rgba(188, 140, 255, 0.18)",
    "warning_limit": "#f0883e",   # Amber dashed line
    "critical_limit": "#f85149",  # Red dotted line
    "grid": "#21262d",
    "paper_bg": "#0d1117",
    "plot_bg": "#161b22",
    "text": "#c9d1d9",
}

AEROSPACE_CSS = """
<style>
    /* Hide Streamlit default multipage pages list in favor of custom UI tabs */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Global Base */
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Top Banner */
    .mission-banner {
        background: linear-gradient(90deg, #161b22 0%, #1f242c 100%);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .demo-watermark {
        background-color: #d29922;
        color: #000000;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 1px;
        padding: 4px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 8px;
    }

    /* KPI Metric Card */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }

    .metric-card:hover {
        border-color: #58a6ff;
    }

    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f0f6fc;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }

    .metric-subtext {
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Status Pill / Badge */
    .status-pill {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border: 1px solid transparent;
    }

    /* Section Cards */
    .section-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #f0f6fc;
        margin-bottom: 12px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
    }

    /* Code & Details block */
    .audit-box {
        background-color: #0d1117;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 12px;
        font-family: monospace;
        font-size: 12px;
        color: #8b949e;
    }
</style>
"""


def render_status_badge(status: StatusLevel, label: Optional[str] = None) -> str:
    """Generate HTML markup for a semantic status pill."""
    color = STATUS_COLORS.get(status, "#8b949e")
    bg = STATUS_BG_COLORS.get(status, "rgba(139, 148, 158, 0.12)")
    text = label if label is not None else status.value
    return (
        f'<span class="status-pill" style="color: {color}; background-color: {bg}; border-color: {color}44;">'
        f'{text}</span>'
    )
