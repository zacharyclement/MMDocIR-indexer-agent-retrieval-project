"""Tests for flattening ColPali page embeddings into insert rows."""

from pathlib import Path

from indexer.flatten.page_patches import build_patch_rows
from indexer.shared.models import RenderedPage, TargetDocument


def test_build_patch_rows_preserves_page_metadata() -> None:
    target_document = TargetDocument(
        doc_name="2310.05634v2.pdf",
        file_path=Path("data/2310.05634v2.pdf"),
        domain="Academic paper",
        source_sha256="abc123",
    )
    rendered_page = RenderedPage(
        page_number=1,
        width=100,
        height=200,
        image=object(),
    )

    rows = build_patch_rows(
        target_document=target_document,
        rendered_page=rendered_page,
        patch_embeddings=[[0.1] * 128, [0.2] * 128],
        indexed_at="2026-04-24T00:00:00+00:00",
        run_id="run123",
    )

    assert len(rows) == 2
    assert rows[0].page_number == 1
    assert rows[0].patch_id == 0
    assert rows[1].patch_id == 1
    assert rows[0].page_uid == "2310.05634v2.pdf::page::1"
    assert rows[0].page_width == 100
    assert rows[0].page_height == 200
    assert rows[0].embedding == [0.1] * 128
