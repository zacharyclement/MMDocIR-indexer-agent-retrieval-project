"""Shared data models for the indexing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class DomainMappingEntry:
    """Represents a single document-to-domain mapping entry."""

    doc_nam: str
    domain: str


@dataclass(frozen=True)
class TargetDocument:
    """Represents a PDF selected for indexing."""

    doc_name: str
    file_path: Path
    domain: str
    source_sha256: str


@dataclass(frozen=True)
class RenderedPage:
    """Represents one rendered PDF page ready for encoding."""

    page_number: int
    width: int
    height: int
    image: Image.Image


@dataclass(frozen=True)
class PageInsertPoint:
    """Represents one page point destined for Qdrant."""

    doc_name: str
    domain: str
    page_number: int
    page_uid: str
    file_path: str
    embeddings: list[list[float]]
    source_sha256: str
    page_width: int
    page_height: int
    indexed_at: str
    run_id: str

    def to_qdrant_payload(self) -> dict[str, object]:
        """Return this point metadata as a Qdrant payload."""

        payload = asdict(self)
        payload.pop("embeddings")
        return payload


@dataclass(frozen=True)
class IndexReportRecord:
    """Represents a per-document indexing outcome written to disk."""

    doc_name: str
    file_path: str
    domain: str | None
    page_count: int | None
    file_hash: str | None
    status: str
    error_message: str | None
    indexed_at: str

    def to_json_dict(self) -> dict[str, object]:
        """Return this record as a JSON-serializable dictionary."""

        return asdict(self)
