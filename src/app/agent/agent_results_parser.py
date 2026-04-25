"""Parse raw agent invocation results into stable application result objects.

The richer parsed trace objects produced here are primarily consumed by
application evaluation and debugging workflows. They remain in the agent layer
because this module interprets DeepAgents/LangChain-style result payloads rather
than computing metrics from already-parsed application data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from indexer.shared.errors import InputValidationError


@dataclass(frozen=True)
class RetrievalCitation:
    """Represents one retrieval citation returned by the application."""

    doc_name: str
    domain: str
    page_number: int
    page_uid: str
    file_path: str
    page_image_path: str
    coarse_score: float
    rerank_score: float


@dataclass(frozen=True)
class RetrievalToolCall:
    """Represents one retrieval tool invocation extracted from agent messages."""

    query: str
    domains: tuple[str, ...]
    doc_names: tuple[str, ...]
    citations: list[RetrievalCitation]


@dataclass(frozen=True)
class ChatTraceResult:
    """Represents one completed chat turn together with retrieval trace data.

    The parsed trace is mainly used by evaluation and debugging code that needs
    retrieval-tool details in addition to the final answer and citations.
    """

    thread_id: str
    model_name: str
    answer: str
    citations: list[RetrievalCitation]
    retrieval_tool_calls: list[RetrievalToolCall]


class AgentResultParser:
    """Convert raw agent invocation results into stable application results.

    This parser is primarily used by evaluation and debugging flows that need a
    richer result shape than the main chat UI. It still belongs in the agent
    layer because it translates framework-specific runtime output into
    application-level objects.
    """

    def parse_chat_trace_result(
        self,
        result: object,
        thread_id: str,
        model_name: str,
    ) -> ChatTraceResult:
        """Build a `ChatTraceResult` from one raw agent invocation result."""

        retrieval_tool_calls = self._extract_retrieval_tool_calls(result)
        return ChatTraceResult(
            thread_id=thread_id,
            model_name=model_name,
            answer=self._extract_assistant_text(result),
            citations=self._extract_retrieval_citations(retrieval_tool_calls),
            retrieval_tool_calls=retrieval_tool_calls,
        )

    def _extract_assistant_text(self, result: object) -> str:
        messages = self._extract_messages(result)
        for message in reversed(messages):
            if self._message_type(message) != "ai":
                continue
            content = self._message_content(message)
            if content:
                return content
        raise InputValidationError("The agent did not return an assistant response.")

    def _extract_retrieval_citations(
        self,
        retrieval_tool_calls: list[RetrievalToolCall],
    ) -> list[RetrievalCitation]:
        if not retrieval_tool_calls:
            return []
        return retrieval_tool_calls[-1].citations

    def _extract_retrieval_tool_calls(self, result: object) -> list[RetrievalToolCall]:
        retrieval_tool_calls: list[RetrievalToolCall] = []
        messages = self._extract_messages(result)
        for message in messages:
            if self._message_type(message) != "tool":
                continue
            tool_name = getattr(message, "name", None)
            if tool_name is None and isinstance(message, Mapping):
                mapped_name = message.get("name")
                tool_name = mapped_name if isinstance(mapped_name, str) else None
            if tool_name != "retrieve_pages":
                continue
            payload = self._extract_retrieval_payload(message)
            if payload is None:
                retrieval_tool_calls.append(
                    RetrievalToolCall(
                        query="",
                        domains=(),
                        doc_names=(),
                        citations=[],
                    )
                )
                continue
            retrieval_tool_calls.append(
                RetrievalToolCall(
                    query=self._extract_retrieval_query(payload),
                    domains=self._extract_string_tuple(payload.get("domains")),
                    doc_names=self._extract_string_tuple(payload.get("doc_names")),
                    citations=self._extract_retrieval_result_citations(
                        payload.get("results")
                    ),
                )
            )
        return retrieval_tool_calls

    def _extract_retrieval_query(self, payload: Mapping[str, object]) -> str:
        query = payload.get("query")
        if isinstance(query, str):
            return query
        return ""

    def _extract_string_tuple(self, value: object) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        normalized_values = [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
        return tuple(normalized_values)

    def _extract_retrieval_result_citations(
        self,
        results: object,
    ) -> list[RetrievalCitation]:
        citations: list[RetrievalCitation] = []
        if not isinstance(results, list):
            return citations
        for result_item in results:
            if not isinstance(result_item, Mapping):
                continue
            citations.append(
                RetrievalCitation(
                    doc_name=str(result_item.get("doc_name", "")),
                    domain=str(result_item.get("domain", "")),
                    page_number=int(result_item.get("page_number", 0)),
                    page_uid=str(result_item.get("page_uid", "")),
                    file_path=str(result_item.get("file_path", "")),
                    page_image_path=str(result_item.get("page_image_path", "")),
                    coarse_score=float(result_item.get("coarse_score", 0.0)),
                    rerank_score=float(result_item.get("rerank_score", 0.0)),
                )
            )
        return citations

    def _extract_retrieval_payload(
        self,
        message: object,
    ) -> Mapping[str, object] | None:
        artifact = getattr(message, "artifact", None)
        if artifact is None and isinstance(message, Mapping):
            mapped_artifact = message.get("artifact")
            artifact = mapped_artifact if isinstance(mapped_artifact, Mapping) else None
        if isinstance(artifact, Mapping):
            return artifact
        return None

    def _extract_messages(self, result: object) -> list[object]:
        if isinstance(result, Mapping):
            messages = result.get("messages", [])
            if isinstance(messages, list):
                return messages
        raise InputValidationError("The agent result did not include a messages list.")

    def _message_type(self, message: object) -> str | None:
        message_type = getattr(message, "type", None)
        if isinstance(message_type, str):
            return message_type
        if isinstance(message, Mapping):
            role = message.get("role")
            if isinstance(role, str):
                if role == "assistant":
                    return "ai"
                return role
        return None

    def _message_content(self, message: object) -> str:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, Mapping):
            content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if isinstance(item, Mapping):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            return "\n".join(part for part in text_parts if part)
        return ""
