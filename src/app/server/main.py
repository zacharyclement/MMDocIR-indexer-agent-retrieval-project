"""FastAPI entrypoint for the retrieval chat POC."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent.config import AppSettings
from app.agent.graph import DeepAgentChatService, build_chat_service
from indexer.shared.errors import (
    DependencyUnavailableError,
    IndexerError,
    InputValidationError,
)
from indexer.shared.logging_utils import configure_logging, get_logger, log_event

LOGGER = get_logger(__name__)


class ChatRequest(BaseModel):
    """Represents one chat request from the HTML UI."""

    message: Annotated[str, Field(min_length=1)]
    thread_id: str | None = None
    model_name: str | None = None
    domains: list[str] | None = None


class CitationResponse(BaseModel):
    """Represents one retrieval citation returned to the HTML UI."""

    doc_name: str
    domain: str
    page_number: int
    page_uid: str
    file_path: str
    page_image_path: str
    coarse_score: float
    rerank_score: float


class ChatResponse(BaseModel):
    """Represents the JSON response returned by the chat endpoint."""

    thread_id: str
    model_name: str
    answer: str
    citations: list[CitationResponse]


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create the FastAPI application for the retrieval chat POC."""

    resolved_settings = settings or AppSettings()
    configure_logging(resolved_settings.log_level)

    app = FastAPI(title="Retrieval Chat POC")
    app.state.settings = resolved_settings
    app.state.agent_service = None

    @app.get("/")
    def index() -> FileResponse:
        static_dir = Path(app.state.settings.static_dir)
        return FileResponse(static_dir / "index.html")

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        try:
            agent_service = _resolve_agent_service(request.app)
            chat_result = agent_service.chat(
                message=payload.message,
                thread_id=payload.thread_id,
                model_name=payload.model_name,
                domains=payload.domains,
            )
        except InputValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except DependencyUnavailableError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        except IndexerError as error:
            log_event(LOGGER, "chat_request_failed", error=str(error))
            raise HTTPException(status_code=500, detail=str(error)) from error

        log_event(
            LOGGER,
            "chat_request_completed",
            thread_id=chat_result.thread_id,
            model_name=chat_result.model_name,
            citation_count=len(chat_result.citations),
        )
        return ChatResponse(
            thread_id=chat_result.thread_id,
            model_name=chat_result.model_name,
            answer=chat_result.answer,
            citations=[
                CitationResponse.model_validate(citation.__dict__)
                for citation in chat_result.citations
            ],
        )

    return app


def _resolve_agent_service(app: FastAPI) -> DeepAgentChatService:
    cached_service = getattr(app.state, "agent_service", None)
    if cached_service is not None and hasattr(cached_service, "chat"):
        return cached_service

    agent_service = build_chat_service(app.state.settings)
    app.state.agent_service = agent_service
    return agent_service


app = create_app()
