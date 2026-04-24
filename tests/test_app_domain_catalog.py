"""Tests for retrieval domain catalog helpers."""

from app.agent.retrieval.domain_catalog import (
    get_available_domains,
    validate_requested_domains,
)
from indexer.shared.errors import InputValidationError


def test_get_available_domains_returns_sorted_unique_values() -> None:
    domains = get_available_domains()

    assert domains == tuple(sorted(set(domains)))
    assert "Academic paper" in domains
    assert "Guidebook" in domains


def test_validate_requested_domains_accepts_known_values() -> None:
    validated_domains = validate_requested_domains(["Guidebook", "Academic paper", "Guidebook"])

    assert validated_domains == ("Academic paper", "Guidebook")


def test_validate_requested_domains_rejects_unknown_values() -> None:
    try:
        validate_requested_domains(["Not a real domain"])
    except InputValidationError as error:
        assert "Unknown domain filter values" in str(error)
    else:
        raise AssertionError("Expected InputValidationError for unknown domain values.")
