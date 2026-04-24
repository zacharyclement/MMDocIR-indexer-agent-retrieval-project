"""Shared utility functions for the indexing pipeline."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA256 checksum of a file."""

    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent_directory(file_path: Path) -> None:
    """Ensure the parent directory for a file exists."""

    file_path.parent.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""

    return datetime.now(timezone.utc).isoformat()
