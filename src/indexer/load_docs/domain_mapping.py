"""Domain mapping source for document loading."""

from __future__ import annotations

from indexer.shared.errors import InputValidationError
from indexer.shared.models import DomainMappingEntry

DOMAIN_MAPPING_ENTRIES: tuple[DomainMappingEntry, ...] = (
    DomainMappingEntry(doc_nam="PH_2016.06.08_Economy-Final.pdf", domain="Research report / Introduction"),
    DomainMappingEntry(doc_nam="Independents-Report.pdf", domain="Research report / Introduction"),
    DomainMappingEntry(doc_nam="0e94b4197b10096b1f4c699701570fbf.pdf", domain="Tutorial/Workshop"),
    DomainMappingEntry(doc_nam="fdac8d1e9ef56519371df7e6532df27d.pdf", domain="Research report / Introduction"),
    DomainMappingEntry(doc_nam="52b3137455e7ca4df65021a200aef724.pdf", domain="Tutorial/Workshop"),
    DomainMappingEntry(doc_nam="earlybird-110722143746-phpapp02_95.pdf", domain="Research report / Introduction"),
    DomainMappingEntry(doc_nam="ddoseattle-150627210357-lva1-app6891_95.pdf", domain="Tutorial/Workshop"),
    DomainMappingEntry(doc_nam="reportq32015-151009093138-lva1-app6891_95.pdf", domain="Research report / Introduction"),
    DomainMappingEntry(doc_nam="2310.05634v2.pdf", domain="Academic paper"),
    DomainMappingEntry(doc_nam="2401.18059v1.pdf", domain="Academic paper"),
    DomainMappingEntry(doc_nam="2312.10997v5.pdf", domain="Academic paper"),
    DomainMappingEntry(doc_nam="honor_watch_gs_pro.pdf", domain="Guidebook"),
    DomainMappingEntry(doc_nam="nova_y70.pdf", domain="Guidebook"),
    DomainMappingEntry(doc_nam="watch_d.pdf", domain="Guidebook"),
)


def load_domain_mapping() -> dict[str, str]:
    """Load and validate the document-to-domain mapping."""

    mapping: dict[str, str] = {}
    for entry in DOMAIN_MAPPING_ENTRIES:
        doc_name = entry.doc_nam.strip()
        domain = entry.domain.strip()

        if not doc_name:
            raise InputValidationError("Domain mapping contains a blank doc_nam value.")
        if not domain:
            raise InputValidationError(
                f"Domain mapping contains a blank domain for '{entry.doc_nam}'."
            )
        if doc_name in mapping:
            raise InputValidationError(
                f"Domain mapping contains a duplicate doc_nam entry for '{doc_name}'."
            )

        mapping[doc_name] = domain

    return mapping
