"""Tests for the FastAPI chat server entrypoint."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
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
        doc_names: list[str] | None,
    ) -> ChatResult:
        assert message == "hello"
        assert model_name == "anthropic:claude-sonnet-4-6"
        assert domains == ["Guidebook"]
        assert doc_names == ["watch_d.pdf"]
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
                    page_image_path="artifacts/page_images/watch/0.png",
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
            "doc_names": ["watch_d.pdf"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["thread_id"] == "thread-123"
    assert payload["answer"] == "hi there"
    assert payload["citations"][0]["doc_name"] == "watch_d.pdf"
    assert (
        payload["citations"][0]["page_image_path"]
        == "artifacts/page_images/watch/0.png"
    )


def test_create_app_applies_langsmith_settings_to_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    create_app(
        AppSettings(
            langsmith_project="custom-project",
            langsmith_tracing=False,
        )
    )

    assert os.environ["LANGSMITH_PROJECT"] == "custom-project"
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_app_settings_accept_legacy_langsmith_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_PROJECT", "legacy-project")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    settings = AppSettings()

    assert settings.langsmith_project == "legacy-project"
    assert settings.langsmith_tracing is False
