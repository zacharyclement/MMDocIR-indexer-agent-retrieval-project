"""Index report writer for per-document indexing results."""

from __future__ import annotations

import json
from pathlib import Path

from indexer.shared.models import IndexReportRecord, TargetDocument
from indexer.shared.utils import ensure_parent_directory, utc_now_iso


class IndexReportWriter:
    """Writes JSONL index report records to disk."""

    def __init__(self, report_path: Path) -> None:
        self._report_path = report_path
        ensure_parent_directory(self._report_path)

    def record_success(self, target_document: TargetDocument, page_count: int) -> None:
        """Record a successful indexing result for a document."""

        record = IndexReportRecord(
            doc_name=target_document.doc_name,
            file_path=str(target_document.file_path),
            domain=target_document.domain,
            page_count=page_count,
            file_hash=target_document.source_sha256,
            status="success",
            error_message=None,
            indexed_at=utc_now_iso(),
        )
        self._append_record(record)

    def record_failure(
        self,
        target_document: TargetDocument,
        page_count: int,
        error_message: str,
    ) -> None:
        """Record a failed indexing result for a document."""

        record = IndexReportRecord(
            doc_name=target_document.doc_name,
            file_path=str(target_document.file_path),
            domain=target_document.domain,
            page_count=page_count,
            file_hash=target_document.source_sha256,
            status="failure",
            error_message=error_message,
            indexed_at=utc_now_iso(),
        )
        self._append_record(record)

    def _append_record(self, record: IndexReportRecord) -> None:
        with self._report_path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
