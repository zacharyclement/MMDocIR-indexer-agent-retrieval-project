"""Deep Agent assembly and chat service for the retrieval POC."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.agent_results_parser import (
    AgentResultParser,
    ChatTraceResult,
    RetrievalCitation,
)
from app.agent.config import AppSettings
from app.agent.llms import build_chat_model, normalize_model_name
from app.agent.prompts.system import build_system_prompt
from app.agent.retrieval.domain_catalog import (
    get_available_domains,
    validate_requested_domains,
)
from app.agent.retrieval.encoder import build_retrieval_encoder
from app.agent.retrieval.qdrant_search import (
    QdrantPageSearchService,
    RankedPageResult,
    RetrievedPageCandidate,
)
from app.agent.retrieval.rerank import rerank_candidates
from app.agent.tools.retrieval import build_retrieval_tool
from indexer.shared.errors import DependencyUnavailableError, InputValidationError


@dataclass(frozen=True)
class AgentRuntimeContext:
    """Per-request runtime context passed into the deep agent."""

    selected_domains: tuple[str, ...]
    selected_doc_names: tuple[str, ...]


@dataclass(frozen=True)
class ChatResult:
    """Represents the minimal result shape for a completed chat turn.

    This is the UI-facing response shape returned by `chat()`. It includes the
    final answer plus the final citations, but omits intermediate retrieval-tool
    execution details that most product callers do not need.
    """

    thread_id: str
    model_name: str
    answer: str
    citations: list[RetrievalCitation]


@dataclass(frozen=True)
class RetrievalPreview:
    """Represents one direct retrieval execution for evaluation purposes."""

    query: str
    domains: tuple[str, ...]
    doc_names: tuple[str, ...]
    candidates: list[RetrievedPageCandidate]
    results: list[RankedPageResult]


class DeepAgentChatService:
    """Owns the retrieval toolchain and deep-agent graphs for the chat server.

    The service exposes both a minimal chat API for normal application use and
    a trace-rich variant for evaluation and debugging workflows.
    """

    def __init__(
        self,
        settings: AppSettings,
        result_parser: AgentResultParser | None = None,
    ) -> None:
        self._settings = settings
        self._result_parser = result_parser or AgentResultParser()
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
        doc_names: Sequence[str] | None,
    ) -> ChatResult:
        """Execute one chat turn and return the minimal application result.

        This is the primary method for normal app flows. It intentionally hides
        intermediate retrieval-tool trace details and returns only the final
        answer and citations needed by the UI.
        """

        trace_result = self.chat_with_trace(
            message=message,
            thread_id=thread_id,
            model_name=model_name,
            domains=domains,
            doc_names=doc_names,
        )
        return ChatResult(
            thread_id=trace_result.thread_id,
            model_name=trace_result.model_name,
            answer=trace_result.answer,
            citations=trace_result.citations,
        )

    def chat_with_trace(
        self,
        message: str,
        thread_id: str | None,
        model_name: str | None,
        domains: Sequence[str] | None,
        doc_names: Sequence[str] | None,
    ) -> ChatTraceResult:
        """Execute one chat turn and return answer plus retrieval trace details.

        This variant is intended for evaluation and debugging workflows that
        need the retrieval tool calls used during generation. Those details are
        consumed immediately for deterministic retrieval metrics and retrieved
        context extraction, even when LangSmith tracing is also enabled.
        """

        resolved_thread_id, normalized_model_name, result = self._invoke_agent(
            message=message,
            thread_id=thread_id,
            model_name=model_name,
            domains=domains,
            doc_names=doc_names,
        )
        return self._result_parser.parse_chat_trace_result(
            result=result,
            thread_id=resolved_thread_id,
            model_name=normalized_model_name,
        )

    def preview_retrieval(
        self,
        query: str,
        domains: Sequence[str] | None,
        doc_names: Sequence[str] | None,
        limit: int | None = None,
    ) -> RetrievalPreview:
        """Run the retrieval stack directly and return coarse plus reranked outputs."""

        normalized_domains = validate_requested_domains(domains)
        normalized_doc_names = _normalize_doc_names(doc_names)
        resolved_limit = limit or self._settings.retrieval_rerank_limit
        query_embeddings = self._search_service.encode_query(query)
        candidates = self._search_service.search_candidates(
            query_embeddings=query_embeddings,
            domains=normalized_domains,
            doc_names=normalized_doc_names,
            limit=max(resolved_limit, self._settings.retrieval_query_limit),
        )
        ranked_results = rerank_candidates(
            encoder=self._encoder,
            query_embeddings=query_embeddings,
            candidates=candidates,
            limit=resolved_limit,
        )
        return RetrievalPreview(
            query=query,
            domains=normalized_domains,
            doc_names=normalized_doc_names,
            candidates=candidates,
            results=ranked_results,
        )

    def _invoke_agent(
        self,
        message: str,
        thread_id: str | None,
        model_name: str | None,
        domains: Sequence[str] | None,
        doc_names: Sequence[str] | None,
    ) -> tuple[str, str, object]:
        """Invoke the deep agent and return identifiers plus the raw result payload."""

        if not message.strip():
            raise InputValidationError("Message must not be blank.")

        normalized_model_name = normalize_model_name(
            model_name or self._settings.default_model
        )
        normalized_domains = validate_requested_domains(domains)
        normalized_doc_names = _normalize_doc_names(doc_names)
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
            context=AgentRuntimeContext(
                selected_domains=normalized_domains,
                selected_doc_names=normalized_doc_names,
            ),
        )
        return resolved_thread_id, normalized_model_name, result

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


def build_chat_service(
    settings: AppSettings | None = None,
    result_parser: AgentResultParser | None = None,
) -> DeepAgentChatService:
    """Build the deep-agent-backed chat service for the server."""

    return DeepAgentChatService(
        settings or AppSettings(),
        result_parser=result_parser,
    )


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


def _normalize_doc_names(doc_names: Sequence[str] | None) -> tuple[str, ...]:
    if doc_names is None:
        return ()
    return tuple(
        sorted({doc_name.strip() for doc_name in doc_names if doc_name.strip()})
    )
