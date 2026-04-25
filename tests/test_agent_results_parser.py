"""Tests for agent result parsing used by evaluation and debugging flows."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.agent_results_parser import AgentResultParser


def test_parse_chat_trace_result_reads_answer_and_tool_artifact() -> None:
    parser = AgentResultParser()
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
            ),
            AIMessage(content="final answer"),
        ]
    }

    trace_result = parser.parse_chat_trace_result(
        result=result,
        thread_id="thread-123",
        model_name="anthropic:claude-sonnet-4-6",
    )

    assert trace_result.thread_id == "thread-123"
    assert trace_result.model_name == "anthropic:claude-sonnet-4-6"
    assert trace_result.answer == "final answer"
    assert len(trace_result.citations) == 1
    assert trace_result.citations[0].doc_name == "watch_d.pdf"
    assert (
        trace_result.citations[0].page_image_path
        == "artifacts/page_images/watch/2.png"
    )


def test_parse_chat_trace_result_returns_empty_citations_without_artifact() -> None:
    parser = AgentResultParser()
    result = {
        "messages": [
            ToolMessage(
                content=[{"type": "text", "text": "retrieved evidence"}],
                tool_call_id="tool-call-1",
                name="retrieve_pages",
            ),
            AIMessage(content="final answer"),
        ]
    }

    trace_result = parser.parse_chat_trace_result(
        result=result,
        thread_id="thread-123",
        model_name="anthropic:claude-sonnet-4-6",
    )

    assert trace_result.answer == "final answer"
    assert trace_result.citations == []
    assert len(trace_result.retrieval_tool_calls) == 1
    assert trace_result.retrieval_tool_calls[0].citations == []


def test_parse_chat_trace_result_preserves_query_filters_and_all_tool_calls() -> None:
    parser = AgentResultParser()
    result = {
        "messages": [
            ToolMessage(
                content=[{"type": "text", "text": "retrieved evidence"}],
                tool_call_id="tool-call-1",
                name="retrieve_pages",
                artifact={
                    "query": "hello world",
                    "domains": ["Guidebook"],
                    "doc_names": ["watch_d.pdf"],
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
                    ],
                },
            ),
            ToolMessage(
                content=[{"type": "text", "text": "retrieved more evidence"}],
                tool_call_id="tool-call-2",
                name="retrieve_pages",
                artifact={
                    "query": "follow up",
                    "domains": ["Guidebook"],
                    "doc_names": ["watch_d.pdf"],
                    "results": [],
                },
            ),
            AIMessage(content="final answer"),
        ]
    }

    trace_result = parser.parse_chat_trace_result(
        result=result,
        thread_id="thread-123",
        model_name="anthropic:claude-sonnet-4-6",
    )

    assert len(trace_result.retrieval_tool_calls) == 2
    assert trace_result.retrieval_tool_calls[0].query == "hello world"
    assert trace_result.retrieval_tool_calls[0].domains == ("Guidebook",)
    assert trace_result.retrieval_tool_calls[0].doc_names == ("watch_d.pdf",)
    assert trace_result.retrieval_tool_calls[0].citations[0].page_number == 2
    assert trace_result.retrieval_tool_calls[1].query == "follow up"
    assert trace_result.citations == []
