"""Input validation functions for the indexing pipeline."""

from __future__ import annotations

from pathlib import Path

from indexer.shared.errors import InputValidationError


def validate_data_dir(data_dir: Path) -> None:
    """Validate that the data directory exists."""

    if not data_dir.exists():
        raise InputValidationError(
            f"Data directory '{data_dir}' does not exist. Add your PDFs before indexing."
        )
    if not data_dir.is_dir():
        raise InputValidationError(f"Data path '{data_dir}' is not a directory.")


def validate_requested_file(file_name: str | None) -> None:
    """Validate a requested file selector."""

    if file_name is None:
        return
    if not file_name.lower().endswith(".pdf"):
        raise InputValidationError(
            f"Requested file '{file_name}' must have a .pdf extension."
        )


def validate_target_files(
    data_dir: Path,
    mapping: dict[str, str],
    file_name: str | None,
) -> list[Path]:
    """Validate that the requested target files exist and are mapped."""

    validate_data_dir(data_dir)
    validate_requested_file(file_name)

    if file_name is not None:
        candidate_paths = [data_dir / file_name]
    else:
        candidate_paths = sorted(path for path in data_dir.glob("*.pdf") if path.is_file())

    if not candidate_paths:
        raise InputValidationError(
            f"No PDF files found in '{data_dir}'. Add documents before indexing."
        )

    missing_files = [str(path) for path in candidate_paths if not path.exists()]
    if missing_files:
        joined_paths = ", ".join(missing_files)
        raise InputValidationError(f"Requested PDF files are missing: {joined_paths}")

    unmapped_documents = [path.name for path in candidate_paths if path.name not in mapping]
    if unmapped_documents:
        joined_names = ", ".join(sorted(unmapped_documents))
        raise InputValidationError(
            "Selected PDFs are missing from the domain mapping: " + joined_names
        )

    return candidate_paths


def find_mapping_gaps(data_dir: Path, mapping: dict[str, str]) -> list[str]:
    """Return PDF filenames in the data directory that are missing from the mapping."""

    validate_data_dir(data_dir)
    return sorted(
        path.name
        for path in data_dir.glob("*.pdf")
        if path.is_file() and path.name not in mapping
    )
