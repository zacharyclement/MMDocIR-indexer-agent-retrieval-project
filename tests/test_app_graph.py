"""Tests for graph-level retrieval result parsing."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from app.agent.graph import _extract_retrieval_citations


def test_extract_retrieval_citations_reads_tool_artifact() -> None:
    result = {
        "messages": [
            ToolMessage(
                content=[{"type": "text", "text": "retrieved evidence"}],
                tool_call_id="tool-call-1",
                name="retrieve_pages",
                artifact={
                    "results": [
                        {
                            "doc_name": "watch_d.pdf",
                            "domain": "Guidebook",
                            "page_number": 2,
                            "page_uid": "watch_d.pdf::page::2",
                            "file_path": "data/watch_d.pdf",
                            "page_image_path": "artifacts/page_images/watch/2.png",
                            "coarse_score": 0.25,
                            "rerank_score": 0.75,
                        }
                    ]
                },
            )
        ]
    }

    citations = _extract_retrieval_citations(result)

    assert len(citations) == 1
    assert citations[0].doc_name == "watch_d.pdf"
    assert citations[0].page_image_path == "artifacts/page_images/watch/2.png"


def test_extract_retrieval_citations_returns_empty_without_artifact() -> None:
    result = {
        "messages": [
            ToolMessage(
                content=[{"type": "text", "text": "retrieved evidence"}],
                tool_call_id="tool-call-1",
                name="retrieve_pages",
            )
        ]
    }

    assert _extract_retrieval_citations(result) == []
