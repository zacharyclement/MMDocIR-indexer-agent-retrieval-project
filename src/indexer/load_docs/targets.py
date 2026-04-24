"""Target-document resolution for the indexing pipeline."""

from __future__ import annotations

from pathlib import Path

from indexer.shared.models import TargetDocument
from indexer.shared.utils import compute_sha256


def resolve_target_documents(
    data_dir: Path,
    mapping: dict[str, str],
    file_name: str | None,
) -> list[TargetDocument]:
    """Resolve the target documents to index from the data directory."""

    if file_name is not None:
        candidate_paths = [data_dir / file_name]
    else:
        candidate_paths = sorted(path for path in data_dir.glob("*.pdf") if path.is_file())

    target_documents: list[TargetDocument] = []
    for candidate_path in candidate_paths:
        doc_name = candidate_path.name
        domain = mapping[doc_name]
        target_documents.append(
            TargetDocument(
                doc_name=doc_name,
                file_path=candidate_path,
                domain=domain,
                source_sha256=compute_sha256(candidate_path),
            )
        )

    return target_documents
