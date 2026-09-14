"""
Telemetry stream handling and buffering interfaces.
"""

from collections import deque
from typing import Deque, Iterator, List, Optional
from telemetry.schema import TelemetryRecord


class TelemetryStreamer:
    """
    Interface for handling telemetry ingestion, buffering, and validation.
    Optimized: O(1) circular ring buffer using collections.deque.
    """

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self._buffer: Deque[TelemetryRecord] = deque(maxlen=buffer_size)

    def push(self, record: TelemetryRecord) -> None:
        """Add a telemetry record to the circular buffer in O(1) time."""
        self._buffer.append(record)

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
