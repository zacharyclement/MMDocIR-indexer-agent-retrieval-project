"""Domain catalog helpers for retrieval filtering."""

from __future__ import annotations

from collections.abc import Iterable

from indexer.load_docs.domain_mapping import load_domain_mapping
from indexer.shared.errors import InputValidationError


def get_available_domains() -> tuple[str, ...]:
    """Return the sorted set of configured document domains."""

    domains = {domain.strip() for domain in load_domain_mapping().values() if domain.strip()}
    return tuple(sorted(domains))


def validate_requested_domains(domains: Iterable[str] | None) -> tuple[str, ...]:
    """Validate requested domains against the configured domain catalog."""

    if domains is None:
        return ()

    available_domains = set(get_available_domains())
    normalized_domains = tuple(sorted({domain.strip() for domain in domains if domain.strip()}))
    unknown_domains = [domain for domain in normalized_domains if domain not in available_domains]
    if unknown_domains:
        raise InputValidationError(
            "Unknown domain filter values: " + ", ".join(sorted(unknown_domains))
        )
    return normalized_domains
