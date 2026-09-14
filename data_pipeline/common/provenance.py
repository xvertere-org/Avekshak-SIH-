"""
Provenance tracking — checksums and source tracing.
"""

import hashlib
import os
from typing import Optional


def compute_file_checksum(filepath: str, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """Compute the hash checksum of a file."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_file_info(filepath: str) -> dict:
    """Return basic file info: size, checksum, exists."""
    if not os.path.exists(filepath):
        return {"exists": False, "size_bytes": 0, "checksum": None}
    size = os.path.getsize(filepath)
    checksum = compute_file_checksum(filepath) if size < 500_000_000 else "SKIPPED_LARGE_FILE"
    return {"exists": True, "size_bytes": size, "checksum": checksum}
