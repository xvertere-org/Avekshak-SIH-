"""
Telemetry stream handling and buffering interfaces.
"""

from typing import Iterator, List, Optional
from telemetry.schema import TelemetryRecord


class TelemetryStreamer:
    """
    Interface for handling telemetry ingestion, buffering, and validation.
    Phase 1: Stub/Interface definition.
    """

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self._buffer: List[TelemetryRecord] = []

    def push(self, record: TelemetryRecord) -> None:
        """Add a telemetry record to the buffer."""
        self._buffer.append(record)
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)

    def get_latest(self) -> Optional[TelemetryRecord]:
        """Retrieve the most recent telemetry record."""
        return self._buffer[-1] if self._buffer else None

    def get_buffer(self) -> List[TelemetryRecord]:
        """Return all buffered records."""
        return list(self._buffer)

    def stream(self) -> Iterator[TelemetryRecord]:
        """Iterate over the buffered records."""
        for record in self._buffer:
            yield record

    def clear(self) -> None:
        """Clear all records from the buffer."""
        self._buffer.clear()
