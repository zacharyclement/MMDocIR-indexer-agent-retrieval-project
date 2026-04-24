"""Flatten ColPali page embeddings into Milvus insert rows."""

from __future__ import annotations

from indexer.shared.models import PatchInsertRow, RenderedPage, TargetDocument


def build_patch_rows(
    target_document: TargetDocument,
    rendered_page: RenderedPage,
    patch_embeddings: list[list[float]],
    indexed_at: str,
    run_id: str,
) -> list[PatchInsertRow]:
    """Build Milvus insert rows for a rendered page."""

    page_uid = f"{target_document.doc_name}::page::{rendered_page.page_number}"
    rows: list[PatchInsertRow] = []
    for patch_id, patch_embedding in enumerate(patch_embeddings):
        rows.append(
            PatchInsertRow(
                doc_name=target_document.doc_name,
                domain=target_document.domain,
                page_number=rendered_page.page_number,
                patch_id=patch_id,
                page_uid=page_uid,
                file_path=str(target_document.file_path),
                embedding=patch_embedding,
                source_sha256=target_document.source_sha256,
                page_width=rendered_page.width,
                page_height=rendered_page.height,
                indexed_at=indexed_at,
                run_id=run_id,
            )
        )
    return rows
