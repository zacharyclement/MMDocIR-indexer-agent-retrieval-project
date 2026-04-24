"""Tests for the FastAPI chat server entrypoint."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from app.agent.config import AppSettings
from app.agent.graph import ChatResult, RetrievalCitation
from app.server.main import create_app


@dataclass
class _FakeAgentService:
    def chat(
        self,
        message: str,
        thread_id: str | None,
        model_name: str | None,
        domains: list[str] | None,
    ) -> ChatResult:
        assert message == "hello"
        assert model_name == "anthropic:claude-sonnet-4-6"
        assert domains == ["Guidebook"]
        return ChatResult(
            thread_id=thread_id or "generated-thread",
            model_name=model_name or "anthropic:claude-sonnet-4-6",
            answer="hi there",
            citations=[
                RetrievalCitation(
                    doc_name="watch_d.pdf",
                    domain="Guidebook",
                    page_number=0,
                    page_uid="watch_d.pdf::page::0",
                    file_path="data/watch_d.pdf",
                    coarse_score=0.1,
                    rerank_score=0.2,
                )
            ],
        )


def test_chat_endpoint_returns_serialized_agent_response() -> None:
    app = create_app(AppSettings())
    app.state.agent_service = _FakeAgentService()
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "message": "hello",
            "thread_id": "thread-123",
            "model_name": "anthropic:claude-sonnet-4-6",
            "domains": ["Guidebook"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["thread_id"] == "thread-123"
    assert payload["answer"] == "hi there"
    assert payload["citations"][0]["doc_name"] == "watch_d.pdf"
