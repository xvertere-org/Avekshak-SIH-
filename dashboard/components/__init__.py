"""
Dashboard Components package for SIH26054 Digital Twin.
"""

from dashboard.components.header import render_header
from dashboard.components.metric_card import render_metric_card
from dashboard.components.telemetry_charts import (
    render_canonical_telemetry_grid,
    render_operating_context_cards,
)
from dashboard.components.diagnostics_view import (
    render_anomaly_diagnostics,
    render_fault_classification,
    render_residual_table,
)
from dashboard.components.prognostics_view import (
    render_health_prognostics,
    render_rul_panel,
    render_projected_trajectory,
)
from dashboard.components.data_quality_view import (
    render_quality_summary,
    render_sensor_status_matrix,
    render_provenance_card,
)
from dashboard.components.xai_view import (
    render_xai_evidence,
)

__all__ = [
    "render_header",
    "render_metric_card",
    "render_canonical_telemetry_grid",
    "render_operating_context_cards",
    "render_anomaly_diagnostics",
    "render_fault_classification",
    "render_residual_table",
    "render_health_prognostics",
    "render_rul_panel",
    "render_projected_trajectory",
    "render_quality_summary",
    "render_sensor_status_matrix",
    "render_provenance_card",
    "render_xai_evidence",
]
