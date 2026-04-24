"""Tests for indexing input validation."""

from pathlib import Path

import pytest

from indexer.load_docs.domain_mapping import load_domain_mapping
from indexer.shared.errors import InputValidationError
from indexer.validate.inputs import find_mapping_gaps, validate_target_files


@pytest.fixture()
def populated_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "2310.05634v2.pdf").write_bytes(b"%PDF-1.4 placeholder")
    (data_dir / "watch_d.pdf").write_bytes(b"%PDF-1.4 placeholder")
    return data_dir


def test_validate_target_files_for_all_documents(populated_data_dir: Path) -> None:
    mapping = load_domain_mapping()

    target_files = validate_target_files(populated_data_dir, mapping, None)

    assert [path.name for path in target_files] == ["2310.05634v2.pdf", "watch_d.pdf"]


def test_validate_target_files_for_single_document(populated_data_dir: Path) -> None:
    mapping = load_domain_mapping()

    target_files = validate_target_files(populated_data_dir, mapping, "watch_d.pdf")

    assert [path.name for path in target_files] == ["watch_d.pdf"]


def test_validate_target_files_rejects_unmapped_documents(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "unknown.pdf").write_bytes(b"%PDF-1.4 placeholder")
    mapping = load_domain_mapping()

    with pytest.raises(InputValidationError, match="missing from the domain mapping"):
        validate_target_files(data_dir, mapping, None)


def test_validate_target_files_rejects_non_pdf_requests(populated_data_dir: Path) -> None:
    mapping = load_domain_mapping()

    with pytest.raises(InputValidationError, match="must have a .pdf extension"):
        validate_target_files(populated_data_dir, mapping, "watch_d.txt")


def test_find_mapping_gaps_returns_unmapped_pdfs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "2310.05634v2.pdf").write_bytes(b"%PDF-1.4 placeholder")
    (data_dir / "missing.pdf").write_bytes(b"%PDF-1.4 placeholder")
    mapping = load_domain_mapping()

    gaps = find_mapping_gaps(data_dir, mapping)

    assert gaps == ["missing.pdf"]
