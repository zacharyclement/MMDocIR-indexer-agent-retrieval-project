"""Milvus writer wrapper for patch-level insert operations."""

from __future__ import annotations

from typing import Iterable

from indexer.shared.errors import DependencyUnavailableError, IndexingRuntimeError
from indexer.shared.models import PatchInsertRow

try:
    from pymilvus import MilvusClient
except ImportError:  # pragma: no cover - exercised in runtime environments without pymilvus.
    MilvusClient = None


class MilvusInsertWriter:
    """Handles Milvus Lite connectivity and page-batch insert operations."""

    def __init__(self, db_path: str, collection_name: str) -> None:
        if MilvusClient is None:
            raise DependencyUnavailableError(
                "pymilvus is required for Milvus write operations."
            )

        self._collection_name = collection_name
        self._client = MilvusClient(uri=db_path)

    @property
    def client(self) -> object:
        """Return the underlying Milvus client."""

        return self._client

    def insert_rows(self, rows: Iterable[PatchInsertRow]) -> int:
        """Insert patch rows into Milvus and return the inserted row count."""

        payload = [row.to_milvus_payload() for row in rows]
        if not payload:
            return 0

        try:
            self._client.insert(collection_name=self._collection_name, data=payload)
        except Exception as error:  # pragma: no cover - depends on pymilvus runtime.
            raise IndexingRuntimeError(f"Failed to insert rows into Milvus: {error}") from error
        return len(payload)
