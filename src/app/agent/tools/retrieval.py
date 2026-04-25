"""Retrieval tool factory for the chat agent."""

from __future__ import annotations

import base64
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from app.agent.retrieval.domain_catalog import validate_requested_domains
from app.agent.retrieval.qdrant_search import QdrantPageSearchService, RetrievalResponse
from app.agent.retrieval.rerank import rerank_candidates
from indexer.encode.colpali import ColPaliPageEncoder
from langchain.tools import ToolRuntime, tool

from indexer.shared.errors import IndexingRuntimeError

# Avoid a runtime circular import with app.agent.graph while preserving type checking.
if TYPE_CHECKING:
    from app.agent.graph import AgentRuntimeContext
else:
    AgentRuntimeContext = Any


def build_retrieval_tool(
    search_service: QdrantPageSearchService,
    encoder: ColPaliPageEncoder,
    query_limit: int,
    rerank_limit: int,
    image_limit: int,
):
    """Build the agent-facing retrieval tool for page search.

    The returned tool performs three steps for each invocation: encode the
    text query, fetch coarse page candidates from Qdrant, and rerank those
    candidates in Python with the shared late-interaction encoder. It also
    applies request-scoped domain and exact document-name filters carried in
    the runtime context so UI-selected filters remain enforced even if the
    model omits them on a follow-up tool call.
    """

    @tool(parse_docstring=True, response_format="content_and_artifact")
    def retrieve_pages(
        query: Annotated[
            str,
            "Natural-language search query for the indexed document corpus.",
        ],
        runtime: ToolRuntime[AgentRuntimeContext],
        domains: Annotated[
            list[str] | None,
            "Optional list of domain filters. Use only configured domain names.",
        ] = None,
        doc_names: Annotated[
            list[str] | None,
            "Optional exact document filename filters. Use indexed doc_name values.",
        ] = None,
        limit: Annotated[
            int | None,
            "Maximum number of final ranked pages to return.",
        ] = None,
    ) -> str:
        """Retrieve the most relevant document pages for a user question.

        Use this when the user asks about the indexed documents or when you need
        evidence from the collection before answering.

        Args:
            query: Natural-language retrieval query.
            domains: Optional domain filters. Use only values from the configured
                domain catalog.
            doc_names: Optional exact document filename filters. Use indexed
                `doc_name` values.
            limit: Maximum number of ranked pages to return.
        """

        runtime_domains = list(runtime.context.selected_domains)
        runtime_doc_names = _normalize_doc_names(list(runtime.context.selected_doc_names))
        requested_domains = domains if domains is not None else runtime_domains
        normalized_domains = validate_requested_domains(requested_domains)
        normalized_doc_names = (
            runtime_doc_names if runtime_doc_names else _normalize_doc_names(doc_names)
        )
        final_limit = limit or rerank_limit
        query_embeddings = search_service.encode_query(query)
        candidates = search_service.search_candidates(
            query_embeddings=query_embeddings,
            domains=normalized_domains,
            doc_names=normalized_doc_names,
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
            doc_names=normalized_doc_names,
            results=ranked_results,
        )
        return (
            _build_retrieval_content(response=response, image_limit=image_limit),
            _build_retrieval_artifact(response),
        )

    return retrieve_pages


def _build_retrieval_content(
    response: RetrievalResponse,
    image_limit: int,
) -> list[dict[str, object]]:
    """Build multimodal tool content blocks for the deep-agent response.

    The first block always contains a text summary of the ranked results.
    Additional image blocks are appended only for the top `image_limit` pages
    so the agent can ground its answer in page renders without flooding the
    context window with every retrieved image.
    """

    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": _build_retrieval_summary_text(
                response=response,
                image_limit=image_limit,
            ),
        }
    ]
    for index, result in enumerate(response.results[:image_limit], start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"Retrieved page image {index}: {result.doc_name}, "
                    f"page {result.page_number} "
                    f"(rerank_score={result.rerank_score:.4f}, "
                    f"coarse_score={result.coarse_score:.4f})."
                ),
            }
        )
        content.append(_build_image_content_block(Path(result.page_image_path)))
    return content


def _build_retrieval_summary_text(
    response: RetrievalResponse,
    image_limit: int,
) -> str:
    """Summarize ranked retrieval results in a model-readable text block."""

    if not response.results:
        return (
            "Retrieval returned no matching pages. "
            "Say clearly that the indexed collection does not contain enough evidence."
        )

    lines = [
        f"Retrieved {len(response.results)} ranked pages for query: {response.query}",
        "Use the retrieved results and included page images as evidence "
        "before answering.",
    ]
    for index, result in enumerate(response.results, start=1):
        lines.append(
            f"{index}. {result.doc_name} page {result.page_number} "
            f"(page_uid={result.page_uid}, rerank_score={result.rerank_score:.4f}, "
            f"coarse_score={result.coarse_score:.4f})"
        )
    included_image_count = min(len(response.results), image_limit)
    if included_image_count:
        lines.append(
            f"The next {included_image_count} content blocks include "
            "rendered page images for the top-ranked pages."
        )
    return "\n".join(lines)


def _build_image_content_block(image_path: Path) -> dict[str, object]:
    """Encode one persisted page image as an inline `image_url` content block."""

    if not image_path.is_file():
        raise IndexingRuntimeError(
            f"Retrieved page image artifact does not exist: {image_path}"
        )

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{_infer_mime_type(image_path)};base64,{encoded_image}",
        },
    }


def _infer_mime_type(image_path: Path) -> str:
    """Infer the MIME type for a retrieved page image artifact."""

    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    raise IndexingRuntimeError(
        f"Unsupported retrieved page image type: {image_path.suffix}"
    )


def _build_retrieval_artifact(response: RetrievalResponse) -> dict[str, Any]:
    """Build the structured tool artifact persisted in the agent trace.

    This artifact intentionally mirrors the ranked retrieval payload in a
    JSON-serializable shape so evaluation, debugging, and citation extraction
    can inspect tool usage without reparsing model-facing content blocks.
    """

    return {
        "query": response.query,
        "domains": list(response.domains),
        "doc_names": list(response.doc_names),
        "results": [asdict(result) for result in response.results],
    }


def _normalize_doc_names(doc_names: list[str] | None) -> tuple[str, ...]:
    """Normalize exact document-name filters into a deduplicated tuple."""

    if doc_names is None:
        return ()
    return tuple(sorted({doc_name.strip() for doc_name in doc_names if doc_name.strip()}))
