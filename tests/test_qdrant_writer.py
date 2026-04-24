"""Tests for Qdrant writer runtime behavior."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from indexer.insert.qdrant_writer import QdrantInsertWriter, _build_point_id
from indexer.shared.models import PageInsertPoint


def test_qdrant_insert_writer_uses_local_path(tmp_path: Path) -> None:
    writer = QdrantInsertWriter(
        db_path=str(tmp_path / "qdrant"),
        collection_name="colpali_page_patches",
    )

    assert writer.client is not None


def test_build_point_id_returns_deterministic_uuid() -> None:
    point_id = _build_point_id("watch_d.pdf::page::1")

    assert isinstance(point_id, UUID)
    assert point_id == _build_point_id("watch_d.pdf::page::1")


def test_qdrant_insert_writer_upserts_points(tmp_path: Path) -> None:
    writer = QdrantInsertWriter(
        db_path=str(tmp_path / "qdrant"),
        collection_name="colpali_page_patches",
    )
    writer.ensure_collection(recreate_collection=False, vector_dimension=128)

    inserted_rows = writer.upsert_points(
        [
            PageInsertPoint(
                doc_name="watch_d.pdf",
                domain="Tutorial/Workshop",
                page_number=1,
                page_uid="watch_d.pdf::page::1",
                file_path="data/watch_d.pdf",
                embeddings=[[0.1] * 128, [0.2] * 128],
                source_sha256="abc123",
                page_width=100,
                page_height=200,
                indexed_at="2026-04-24T00:00:00+00:00",
                run_id="run123",
            )
        ]
    )

    assert inserted_rows == 1
