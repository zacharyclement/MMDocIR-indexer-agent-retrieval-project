"""Tests for flattening ColPali page embeddings into insert points."""

from pathlib import Path

from indexer.flatten.page_patches import build_page_point
from indexer.shared.models import RenderedPage, TargetDocument


def test_build_page_point_preserves_page_metadata() -> None:
    target_document = TargetDocument(
        doc_name="2310.05634v2.pdf",
        file_path=Path("data/2310.05634v2.pdf"),
        domain="Academic paper",
        source_sha256="abc123",
    )
    rendered_page = RenderedPage(
        page_number=0,
        width=100,
        height=200,
        image=object(),
    )

    point = build_page_point(
        target_document=target_document,
        rendered_page=rendered_page,
        page_image_path="artifacts/page_images/abc123/0.png",
        patch_embeddings=[[0.1] * 128, [0.2] * 128],
        indexed_at="2026-04-24T00:00:00+00:00",
        run_id="run123",
    )

    assert point.page_number == 0
    assert point.page_uid == "2310.05634v2.pdf::page::0"
    assert point.page_image_path == "artifacts/page_images/abc123/0.png"
    assert point.page_width == 100
    assert point.page_height == 200
    assert point.embeddings == [[0.1] * 128, [0.2] * 128]
