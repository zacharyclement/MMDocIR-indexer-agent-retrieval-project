"""Tests for retrieval reranking helpers."""

from __future__ import annotations

from app.agent.retrieval.qdrant_search import RetrievedPageCandidate
from app.agent.retrieval.rerank import rerank_candidates


class _FakeEncoder:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score_query_to_pages(
        self,
        query_embeddings: list[list[float]],
        page_embeddings: list[list[list[float]]],
    ) -> list[float]:
        assert query_embeddings == [[1.0, 2.0]]
        assert len(page_embeddings) == len(self._scores)
        return self._scores


def test_rerank_candidates_orders_by_rerank_score() -> None:
    candidates = [
        RetrievedPageCandidate(
            doc_name="alpha.pdf",
            domain="Guidebook",
            page_number=0,
            page_uid="alpha.pdf::page::0",
            file_path="data/alpha.pdf",
            page_image_path="artifacts/page_images/hash-a/0.png",
            source_sha256="hash-a",
            coarse_score=0.4,
            page_embeddings=[[0.1, 0.2]],
        ),
        RetrievedPageCandidate(
            doc_name="beta.pdf",
            domain="Academic paper",
            page_number=1,
            page_uid="beta.pdf::page::1",
            file_path="data/beta.pdf",
            page_image_path="artifacts/page_images/hash-b/1.png",
            source_sha256="hash-b",
            coarse_score=0.7,
            page_embeddings=[[0.3, 0.4]],
        ),
    ]

    ranked_results = rerank_candidates(
        encoder=_FakeEncoder([0.2, 0.9]),
        query_embeddings=[[1.0, 2.0]],
        candidates=candidates,
        limit=2,
    )

    assert [result.doc_name for result in ranked_results] == ["beta.pdf", "alpha.pdf"]
    assert ranked_results[0].page_image_path == "artifacts/page_images/hash-b/1.png"
    assert ranked_results[0].rerank_score == 0.9


def test_rerank_candidates_deduplicates_by_page_uid() -> None:
    candidates = [
        RetrievedPageCandidate(
            doc_name="alpha.pdf",
            domain="Guidebook",
            page_number=0,
            page_uid="alpha.pdf::page::0",
            file_path="data/alpha.pdf",
            page_image_path="artifacts/page_images/hash-a/0.png",
            source_sha256="hash-a",
            coarse_score=0.4,
            page_embeddings=[[0.1, 0.2]],
        ),
        RetrievedPageCandidate(
            doc_name="alpha.pdf",
            domain="Guidebook",
            page_number=0,
            page_uid="alpha.pdf::page::0",
            file_path="data/alpha.pdf",
            page_image_path="artifacts/page_images/hash-a/0b.png",
            source_sha256="hash-a",
            coarse_score=0.8,
            page_embeddings=[[0.3, 0.4]],
        ),
    ]

    ranked_results = rerank_candidates(
        encoder=_FakeEncoder([0.5, 0.7]),
        query_embeddings=[[1.0, 2.0]],
        candidates=candidates,
        limit=5,
    )

    assert len(ranked_results) == 1
    assert ranked_results[0].page_image_path == "artifacts/page_images/hash-a/0b.png"
    assert ranked_results[0].rerank_score == 0.7
