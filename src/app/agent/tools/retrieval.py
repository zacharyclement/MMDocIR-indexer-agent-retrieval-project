"""Retrieval tool factory for the chat agent."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated

from app.agent.retrieval.domain_catalog import validate_requested_domains
from app.agent.retrieval.rerank import rerank_candidates
from app.agent.retrieval.qdrant_search import RetrievalResponse
from indexer.encode.colpali import ColPaliPageEncoder
from indexer.shared.errors import DependencyUnavailableError

if TYPE_CHECKING:
    from app.agent.graph import AgentRuntimeContext
    from app.agent.retrieval.qdrant_search import QdrantPageSearchService


def build_retrieval_tool(
    search_service: "QdrantPageSearchService",
    encoder: ColPaliPageEncoder,
    query_limit: int,
    score_limit: int,
):
    """Build the retrieval tool bound to the configured search service."""

    try:
        from langchain.tools import ToolRuntime, tool
    except ImportError as error:  # pragma: no cover - depends on optional runtime dependency.
        raise DependencyUnavailableError(
            "langchain is required to construct the retrieval tool. Install the app dependencies first."
        ) from error

    from app.agent.graph import AgentRuntimeContext

    globals()["ToolRuntime"] = ToolRuntime
    globals()["AgentRuntimeContext"] = AgentRuntimeContext

    @tool(parse_docstring=True)
    def retrieve_pages(
        query: Annotated[str, "Natural-language search query for the indexed document corpus."],
        runtime: ToolRuntime["AgentRuntimeContext"],
        domains: Annotated[
            list[str] | None,
            "Optional list of domain filters. Use only configured domain names.",
        ] = None,
        limit: Annotated[
            int | None,
            "Maximum number of final ranked pages to return.",
        ] = None,
    ) -> str:
        """Retrieve the most relevant document pages for a user question.

        Use this when the user asks about the indexed documents or when you need evidence
        from the collection before answering.

        Args:
            query: Natural-language retrieval query.
            domains: Optional domain filters. Use only values from the configured domain catalog.
            limit: Maximum number of ranked pages to return.
        """

        runtime_domains = list(runtime.context.selected_domains)
        requested_domains = domains if domains is not None else runtime_domains
        normalized_domains = validate_requested_domains(requested_domains)
        final_limit = limit or score_limit
        query_embeddings = search_service.encode_query(query)
        candidates = search_service.search_candidates(
            query_embeddings=query_embeddings,
            domains=normalized_domains,
            limit=max(final_limit, query_limit),
        )
        ranked_results = rerank_candidates(
            encoder=encoder,
            query_embeddings=query_embeddings,
            candidates=candidates,
            limit=final_limit,
        )
        response = RetrievalResponse(
            query=query,
            domains=normalized_domains,
            results=ranked_results,
        )
        return json.dumps(
            {
                "query": response.query,
                "domains": list(response.domains),
                "results": [asdict(result) for result in response.results],
            }
        )

    return retrieve_pages
