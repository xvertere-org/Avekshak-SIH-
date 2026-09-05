"""
Dashboard Pages package for SIH26054 Digital Twin.
"""

from dashboard.pages.overview_page import render_overview_page
from dashboard.pages.telemetry_page import render_telemetry_page
from dashboard.pages.diagnostics_page import render_diagnostics_page
from dashboard.pages.prognostics_page import render_prognostics_page
from dashboard.pages.system_status_page import render_system_status_page

__all__ = [
    "render_overview_page",
    "render_telemetry_page",
    "render_diagnostics_page",
    "render_prognostics_page",
    "render_system_status_page",
]
