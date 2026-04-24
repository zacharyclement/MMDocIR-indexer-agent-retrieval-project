"""System prompt builders for the retrieval chat agent."""

from __future__ import annotations

from collections.abc import Sequence


def build_system_prompt(available_domains: Sequence[str]) -> str:
    """Build the system prompt for the retrieval chat agent."""

    domain_list = ", ".join(available_domains)
    return (
        "You are a retrieval-focused assistant for indexed PDF documents. "
        "Use the retrieval tool whenever the user asks about the indexed corpus or when you need evidence before answering. "
        "When helpful, filter retrieval to the relevant domain. "
        f"Available domains: {domain_list}. "
        "Ground answers in retrieved pages, mention the document name and page number when citing evidence, "
        "and say clearly when the collection does not contain enough evidence. "
        "Keep answers concise, factual, and directly responsive to the user."
    )
