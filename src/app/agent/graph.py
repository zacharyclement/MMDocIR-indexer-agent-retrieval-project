"""Deep Agent assembly and chat service for the retrieval POC."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.config import AppSettings
from app.agent.llms import build_chat_model, normalize_model_name
from app.agent.prompts.system import build_system_prompt
from app.agent.retrieval.domain_catalog import (
    get_available_domains,
    validate_requested_domains,
)
from app.agent.retrieval.encoder import build_retrieval_encoder
from app.agent.retrieval.qdrant_search import QdrantPageSearchService
from app.agent.tools.retrieval import build_retrieval_tool
from indexer.shared.errors import DependencyUnavailableError, InputValidationError


@dataclass(frozen=True)
class AgentRuntimeContext:
    """Per-request runtime context passed into the deep agent."""

    selected_domains: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalCitation:
    """Represents one retrieval citation returned to the chat UI."""

    doc_name: str
    domain: str
    page_number: int
    page_uid: str
    file_path: str
    page_image_path: str
    coarse_score: float
    rerank_score: float


@dataclass(frozen=True)
class ChatResult:
    """Represents one completed chat turn."""

    thread_id: str
    model_name: str
    answer: str
    citations: list[RetrievalCitation]


class DeepAgentChatService:
    """Owns the retrieval toolchain and deep-agent graphs for the chat server."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._available_domains = get_available_domains()
        self._encoder = build_retrieval_encoder(
            model_name=settings.retrieval_model_name,
            device_name=settings.retrieval_device,
        )
        self._search_service = QdrantPageSearchService(
            db_path=settings.qdrant_path,
            collection_name=settings.collection_name,
            encoder=self._encoder,
        )
        self._retrieval_tool = build_retrieval_tool(
            search_service=self._search_service,
            encoder=self._encoder,
            query_limit=settings.retrieval_query_limit,
            rerank_limit=settings.retrieval_rerank_limit,
            image_limit=settings.retrieval_image_limit,
        )
        self._skill_files = _load_skill_files(settings.skills_dir)
        self._system_prompt = build_system_prompt(self._available_domains)
        self._checkpointer = self._build_checkpointer()
        self._graphs: dict[str, Any] = {}

    @property
    def available_domains(self) -> tuple[str, ...]:
        """Return the configured domain catalog."""

        return self._available_domains

    def chat(
        self,
        message: str,
        thread_id: str | None,
        model_name: str | None,
        domains: Sequence[str] | None,
    ) -> ChatResult:
        """Execute one chat turn through the configured deep agent."""

        if not message.strip():
            raise InputValidationError("Message must not be blank.")

        normalized_model_name = normalize_model_name(
            model_name or self._settings.default_model
        )
        normalized_domains = validate_requested_domains(domains)
        resolved_thread_id = thread_id or uuid.uuid4().hex
        graph = self._get_graph(normalized_model_name)
        agent_input: dict[str, object] = {
            "messages": [{"role": "user", "content": message}],
        }
        if self._skill_files:
            agent_input["files"] = self._skill_files

        result = graph.invoke(
            agent_input,
            config={"configurable": {"thread_id": resolved_thread_id}},
            context=AgentRuntimeContext(selected_domains=normalized_domains),
        )
        return ChatResult(
            thread_id=resolved_thread_id,
            model_name=normalized_model_name,
            answer=_extract_assistant_text(result),
            citations=_extract_retrieval_citations(result),
        )

    def _get_graph(self, model_name: str) -> Any:
        graph = self._graphs.get(model_name)
        if graph is not None:
            return graph

        graph = self._build_graph(model_name)
        self._graphs[model_name] = graph
        return graph

    def _build_graph(self, model_name: str) -> Any:
        try:
            from deepagents import create_deep_agent
        except ImportError as error:  # pragma: no cover - optional dependency.
            raise DependencyUnavailableError(
                "deepagents is required to build the retrieval chat graph. "
                "Install the app dependencies first."
            ) from error

        skills = ["/skills/"] if self._skill_files else None
        return create_deep_agent(
            model=build_chat_model(model_name),
            tools=[self._retrieval_tool],
            system_prompt=self._system_prompt,
            skills=skills,
            checkpointer=self._checkpointer,
            context_schema=AgentRuntimeContext,
        )

    @staticmethod
    def _build_checkpointer() -> Any:
        try:
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError as error:  # pragma: no cover - optional dependency.
            raise DependencyUnavailableError(
                "langgraph is required to enable threaded in-memory chat history. "
                "Install the app dependencies first."
            ) from error

        return MemorySaver()


def build_chat_service(settings: AppSettings | None = None) -> DeepAgentChatService:
    """Build the deep-agent-backed chat service for the server."""

    return DeepAgentChatService(settings or AppSettings())


def _load_skill_files(skills_dir: Path) -> dict[str, object]:
    """Load project skill files into the deep agent state backend input format."""

    if not skills_dir.exists():
        return {}

    try:
        from deepagents.backends.utils import create_file_data
    except ImportError as error:  # pragma: no cover - optional dependency.
        raise DependencyUnavailableError(
            "deepagents is required to load project skill files. "
            "Install the app dependencies first."
        ) from error

    skill_files: dict[str, object] = {}
    for file_path in sorted(path for path in skills_dir.rglob("*") if path.is_file()):
        relative_path = file_path.relative_to(skills_dir).as_posix()
        virtual_path = f"/skills/{relative_path}"
        skill_files[virtual_path] = create_file_data(
            file_path.read_text(encoding="utf-8")
        )
    return skill_files


def _extract_assistant_text(result: object) -> str:
    """Extract the latest assistant message text from an agent invocation result."""

    messages = _extract_messages(result)
    for message in reversed(messages):
        if _message_type(message) != "ai":
            continue
        content = _message_content(message)
        if content:
            return content
    raise InputValidationError("The agent did not return an assistant response.")


def _extract_retrieval_citations(result: object) -> list[RetrievalCitation]:
    """Extract the latest retrieval tool payload from the agent result."""

    messages = _extract_messages(result)
    for message in reversed(messages):
        if _message_type(message) != "tool":
            continue
        tool_name = getattr(message, "name", None)
        if tool_name is None and isinstance(message, Mapping):
            mapped_name = message.get("name")
            tool_name = mapped_name if isinstance(mapped_name, str) else None
        if tool_name != "retrieve_pages":
            continue
        payload = _extract_retrieval_payload(message)
        if payload is None:
            return []
        results = payload.get("results", [])
        if not isinstance(results, list):
            return []
        citations: list[RetrievalCitation] = []
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
    return []


def _extract_retrieval_payload(message: object) -> Mapping[str, object] | None:
    artifact = getattr(message, "artifact", None)
    if artifact is None and isinstance(message, Mapping):
        mapped_artifact = message.get("artifact")
        artifact = mapped_artifact if isinstance(mapped_artifact, Mapping) else None
    if isinstance(artifact, Mapping):
        return artifact
    return None


def _extract_messages(result: object) -> list[object]:
    if isinstance(result, Mapping):
        messages = result.get("messages", [])
        if isinstance(messages, list):
            return messages
    raise InputValidationError("The agent result did not include a messages list.")


def _message_type(message: object) -> str | None:
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


def _message_content(message: object) -> str:
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
