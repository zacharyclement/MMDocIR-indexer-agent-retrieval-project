"""Tests for graph-level chat service orchestration."""

from __future__ import annotations

from app.agent.agent_results_parser import ChatTraceResult, RetrievalCitation
from app.agent.graph import DeepAgentChatService


class _FakeResultParser:
    def __init__(self, trace_result: ChatTraceResult) -> None:
        self._trace_result = trace_result
        self.calls: list[tuple[object, str, str]] = []

    def parse_chat_trace_result(
        self,
        result: object,
        thread_id: str,
        model_name: str,
    ) -> ChatTraceResult:
        self.calls.append((result, thread_id, model_name))
        return self._trace_result


def test_chat_with_trace_delegates_to_injected_result_parser() -> None:
    raw_result = {"messages": []}
    parsed_trace = ChatTraceResult(
        thread_id="thread-123",
        model_name="anthropic:claude-sonnet-4-6",
        answer="final answer",
        citations=[],
        retrieval_tool_calls=[],
    )
    parser = _FakeResultParser(parsed_trace)
    service = object.__new__(DeepAgentChatService)

    def fake_invoke_agent(
        *,
        message: str,
        thread_id: str | None,
        model_name: str | None,
        domains: list[str] | None,
        doc_names: list[str] | None,
    ) -> tuple[str, str, object]:
        assert message == "hello"
        assert thread_id is None
        assert model_name is None
        assert domains == ["Guidebook"]
        assert doc_names == ["watch_d.pdf"]
        return ("thread-123", "anthropic:claude-sonnet-4-6", raw_result)

    service._result_parser = parser
    service._invoke_agent = fake_invoke_agent

    trace_result = service.chat_with_trace(
        message="hello",
        thread_id=None,
        model_name=None,
        domains=["Guidebook"],
        doc_names=["watch_d.pdf"],
    )

    assert trace_result is parsed_trace
    assert parser.calls == [
        (raw_result, "thread-123", "anthropic:claude-sonnet-4-6")
    ]


def test_chat_projects_trace_result_to_minimal_chat_result() -> None:
    service = object.__new__(DeepAgentChatService)

    def fake_chat_with_trace(**_: object) -> ChatTraceResult:
        return ChatTraceResult(
            thread_id="thread-123",
            model_name="anthropic:claude-sonnet-4-6",
            answer="final answer",
            citations=[
                RetrievalCitation(
                    doc_name="watch_d.pdf",
                    domain="Guidebook",
                    page_number=2,
                    page_uid="watch_d.pdf::page::2",
                    file_path="data/watch_d.pdf",
                    page_image_path="artifacts/page_images/watch/2.png",
                    coarse_score=0.25,
                    rerank_score=0.75,
                )
            ],
            retrieval_tool_calls=[],
        )

    service.chat_with_trace = fake_chat_with_trace

    chat_result = service.chat(
        message="hello",
        thread_id=None,
        model_name=None,
        domains=["Guidebook"],
        doc_names=["watch_d.pdf"],
    )

    assert chat_result.thread_id == "thread-123"
    assert chat_result.model_name == "anthropic:claude-sonnet-4-6"
    assert chat_result.answer == "final answer"
    assert chat_result.citations[0].page_number == 2
