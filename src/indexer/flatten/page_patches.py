"""Flatten ColPali page embeddings into Qdrant page points."""

from __future__ import annotations

from indexer.shared.models import PageInsertPoint, RenderedPage, TargetDocument


def build_page_point(
    target_document: TargetDocument,
    rendered_page: RenderedPage,
    patch_embeddings: list[list[float]],
    indexed_at: str,
    run_id: str,
) -> PageInsertPoint:
    """Build one Qdrant point for a rendered page."""

    page_uid = f"{target_document.doc_name}::page::{rendered_page.page_number}"
    return PageInsertPoint(
        doc_name=target_document.doc_name,
        domain=target_document.domain,
        page_number=rendered_page.page_number,
        page_uid=page_uid,
        file_path=str(target_document.file_path),
        embeddings=patch_embeddings,
        source_sha256=target_document.source_sha256,
        page_width=rendered_page.width,
        page_height=rendered_page.height,
        indexed_at=indexed_at,
        run_id=run_id,
    )
