"""Python-side reranking helpers for retrieval results."""

from __future__ import annotations

from app.agent.retrieval.qdrant_search import RankedPageResult, RetrievedPageCandidate
from indexer.encode.colpali import ColPaliPageEncoder


def rerank_candidates(
    encoder: ColPaliPageEncoder,
    query_embeddings: list[list[float]],
    candidates: list[RetrievedPageCandidate],
    limit: int,
) -> list[RankedPageResult]:
    """Rerank coarse page candidates using MaxSim scoring in Python."""

    if not candidates:
        return []

    rerank_scores = encoder.score_query_to_pages(
        query_embeddings=query_embeddings,
        page_embeddings=[candidate.page_embeddings for candidate in candidates],
    )
    grouped_results: dict[str, RankedPageResult] = {}
    for candidate, rerank_score in zip(candidates, rerank_scores, strict=True):
        ranked_result = RankedPageResult(
            doc_name=candidate.doc_name,
            domain=candidate.domain,
            page_number=candidate.page_number,
            page_uid=candidate.page_uid,
            file_path=candidate.file_path,
            page_image_path=candidate.page_image_path,
            source_sha256=candidate.source_sha256,
            coarse_score=candidate.coarse_score,
            rerank_score=rerank_score,
        )
        existing_result = grouped_results.get(candidate.page_uid)
        if (
            existing_result is None
            or ranked_result.rerank_score > existing_result.rerank_score
        ):
            grouped_results[candidate.page_uid] = ranked_result

    ranked_results = sorted(
        grouped_results.values(),
        key=lambda result: (result.rerank_score, result.coarse_score),
        reverse=True,
    )
    return ranked_results[:limit]
