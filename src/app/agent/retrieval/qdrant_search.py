"""Qdrant-backed retrieval service for page-level search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.retrieval.domain_catalog import validate_requested_domains
from indexer.encode.colpali import ColPaliPageEncoder
from indexer.shared.errors import DependencyUnavailableError, IndexingRuntimeError

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:  # pragma: no cover - depends on optional runtime dependency.
    QdrantClient = None
    models = None


@dataclass(frozen=True)
class RetrievedPageCandidate:
    """Represents one retrieved page candidate with stored page embeddings."""

    doc_name: str
    domain: str
    page_number: int
    page_uid: str
    file_path: str
    page_image_path: str
    source_sha256: str
    coarse_score: float
    page_embeddings: list[list[float]]


@dataclass(frozen=True)
class RankedPageResult:
    """Represents one ranked retrieval result ready for the agent response."""

    doc_name: str
    domain: str
    page_number: int
    page_uid: str
    file_path: str
    page_image_path: str
    source_sha256: str
    coarse_score: float
    rerank_score: float


@dataclass(frozen=True)
class RetrievalResponse:
    """Represents a retrieval invocation and its ranked results."""

    query: str
    domains: tuple[str, ...]
    doc_names: tuple[str, ...]
    results: list[RankedPageResult]


class QdrantPageSearchService:
    """Coordinates query encoding, Qdrant retrieval, and page-level result shaping."""

    def __init__(
        self,
        db_path: Path,
        collection_name: str,
        encoder: ColPaliPageEncoder,
    ) -> None:
        if QdrantClient is None or models is None:
            raise DependencyUnavailableError(
                "qdrant-client is required for retrieval operations."
            )

        self._collection_name = collection_name
        self._client = QdrantClient(path=str(db_path))
        self._encoder = encoder

    def encode_query(self, query_text: str) -> list[list[float]]:
        """Encode a user query into multivector embeddings."""

        return self._encoder.encode_query(query_text)

    def search_candidates(
        self,
        query_embeddings: list[list[float]],
        domains: Sequence[str] | None,
        doc_names: Sequence[str] | None,
        limit: int,
    ) -> list[RetrievedPageCandidate]:
        """Retrieve coarse page candidates from Qdrant."""

        query_filter = self._build_query_filter(domains, doc_names)
        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_embeddings,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as error:  # pragma: no cover - depends on qdrant runtime.
            raise IndexingRuntimeError(
                f"Failed to query Qdrant collection '{self._collection_name}': {error}"
            ) from error

        hits = getattr(response, "points", None)
        if not isinstance(hits, Sequence):
            raise IndexingRuntimeError(
                "Qdrant query did not return a sequence of points."
            )

        candidates: list[RetrievedPageCandidate] = []
        for hit in hits:
            payload = getattr(hit, "payload", None)
            if not isinstance(payload, Mapping):
                raise IndexingRuntimeError(
                    "Qdrant hit did not include a payload mapping."
                )
            vector = _extract_multivector(getattr(hit, "vector", None))
            candidates.append(
                RetrievedPageCandidate(
                    doc_name=_require_non_empty_string(
                        payload.get("doc_name"),
                        "doc_name",
                    ),
                    domain=_require_non_empty_string(
                        payload.get("domain"),
                        "domain",
                    ),
                    page_number=_require_non_negative_int(
                        payload.get("page_number"),
                        "page_number",
                    ),
                    page_uid=_require_non_empty_string(
                        payload.get("page_uid"),
                        "page_uid",
                    ),
                    file_path=_require_non_empty_string(
                        payload.get("file_path"),
                        "file_path",
                    ),
                    page_image_path=_require_non_empty_string(
                        payload.get("page_image_path"),
                        "page_image_path",
                    ),
                    source_sha256=_require_non_empty_string(
                        payload.get("source_sha256"),
                        "source_sha256",
                    ),
                    coarse_score=_require_numeric_score(getattr(hit, "score", None)),
                    page_embeddings=vector,
                )
            )
        return candidates

    @staticmethod
    def _build_query_filter(
        domains: Sequence[str] | None,
        doc_names: Sequence[str] | None,
    ) -> Any | None:
        normalized_domains = validate_requested_domains(domains)
        normalized_doc_names = _normalize_doc_names(doc_names)
        must_conditions: list[Any] = []
        if normalized_domains:
            must_conditions.append(
                models.FieldCondition(
                    key="domain",
                    match=models.MatchAny(any=list(normalized_domains)),
                )
            )
        if normalized_doc_names:
            must_conditions.append(
                models.FieldCondition(
                    key="doc_name",
                    match=models.MatchAny(any=list(normalized_doc_names)),
                )
            )
        if not must_conditions:
            return None
        return models.Filter(must=must_conditions)


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IndexingRuntimeError(
            f"Field '{field_name}' must be a non-empty string."
        )
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IndexingRuntimeError(
            f"Field '{field_name}' must be a non-negative integer."
        )
    return value


def _require_numeric_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndexingRuntimeError("Retrieval score must be numeric.")
    return float(value)


def _extract_multivector(vector: object) -> list[list[float]]:
    if isinstance(vector, Mapping):
        if not vector:
            raise IndexingRuntimeError("Returned vector mapping was empty.")
        first_value = next(iter(vector.values()))
        return _extract_multivector(first_value)

    if not isinstance(vector, list) or not vector:
        raise IndexingRuntimeError(
            "Returned vector was not a non-empty multivector list."
        )

    normalized: list[list[float]] = []
    expected_dimension: int | None = None
    for row in vector:
        if not isinstance(row, list) or not row:
            raise IndexingRuntimeError("Each multivector row must be a non-empty list.")
        numeric_row = [_to_float(entry) for entry in row]
        if expected_dimension is None:
            expected_dimension = len(numeric_row)
        elif len(numeric_row) != expected_dimension:
            raise IndexingRuntimeError("Multivector rows did not share one dimension.")
        normalized.append(numeric_row)
    return normalized


def _to_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndexingRuntimeError("Vector entries must be numeric.")
    return float(value)


def _normalize_doc_names(doc_names: Sequence[str] | None) -> tuple[str, ...]:
    if doc_names is None:
        return ()
    return tuple(sorted({doc_name.strip() for doc_name in doc_names if doc_name.strip()}))
