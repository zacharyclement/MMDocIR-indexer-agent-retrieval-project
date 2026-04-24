"""Tests for domain mapping behavior."""

from indexer.load_docs.domain_mapping import load_domain_mapping


def test_load_domain_mapping_contains_expected_entries() -> None:
    mapping = load_domain_mapping()

    assert mapping["2310.05634v2.pdf"] == "Academic paper"
    assert mapping["watch_d.pdf"] == "Guidebook"
    assert mapping["2024.ug.eprospectus.pdf"] == "Brochure"
    assert len(mapping) == 25
