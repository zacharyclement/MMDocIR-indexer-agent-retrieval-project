"""Qdrant writer wrapper for page-level insert operations."""

from __future__ import annotations

import uuid
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models

from indexer.shared.errors import IndexingRuntimeError
from indexer.shared.models import PageInsertPoint


def _build_point_id(page_uid: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, page_uid)


class QdrantInsertWriter:
    """Handles local Qdrant connectivity and page-batch upsert operations."""

    def __init__(self, db_path: str, collection_name: str) -> None:
        self._collection_name = collection_name
        self._client = QdrantClient(path=db_path)

    @property
    def client(self) -> object:
        """Return the underlying Qdrant client."""

        return self._client

    def ensure_collection(self, recreate_collection: bool, vector_dimension: int) -> None:
        """Create or recreate the Qdrant collection used by the pipeline."""

        try:
            collection_exists = self._client.collection_exists(self._collection_name)
            if collection_exists and recreate_collection:
                self._client.delete_collection(self._collection_name)
                collection_exists = False
            if collection_exists:
                return

            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                ),
            )
        except Exception as error:  # pragma: no cover - depends on qdrant runtime.
            raise IndexingRuntimeError(
                f"Failed to create Qdrant collection '{self._collection_name}': {error}"
            ) from error

    def upsert_points(self, points: Iterable[PageInsertPoint]) -> int:
        """Upsert page points into Qdrant and return the inserted point count."""

        payload = [
            models.PointStruct(
                id=_build_point_id(point.page_uid),
                vector=point.embeddings,
                payload=point.to_qdrant_payload(),
            )
            for point in points
        ]
        if not payload:
            return 0

        try:
            self._client.upsert(collection_name=self._collection_name, points=payload)
        except Exception as error:  # pragma: no cover - depends on qdrant runtime.
            raise IndexingRuntimeError(f"Failed to upsert points into Qdrant: {error}") from error
        return len(payload)
