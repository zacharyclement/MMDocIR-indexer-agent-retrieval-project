"""Chat model selection helpers for the retrieval chat agent."""

from __future__ import annotations

from indexer.shared.errors import DependencyUnavailableError
from indexer.shared.errors import InputValidationError

DEFAULT_MODEL_NAME = "anthropic:claude-sonnet-4-6"

_SUPPORTED_MODEL_ALIASES: dict[str, str] = {
    "anthropic:claude-sonnet-4-6": "anthropic:claude-sonnet-4-6",
    "claude-sonnet-4-6": "anthropic:claude-sonnet-4-6",
    "sonnet": "anthropic:claude-sonnet-4-6",
    "anthropic:claude-opus-4-7": "anthropic:claude-opus-4-7",
    "claude-opus-4-7": "anthropic:claude-opus-4-7",
    "opus": "anthropic:claude-opus-4-7",
}


def get_supported_model_names() -> tuple[str, str]:
    """Return the supported fully qualified model names."""

    return (
        "anthropic:claude-sonnet-4-6",
        "anthropic:claude-opus-4-7",
    )


def normalize_model_name(model_name: str | None) -> str:
    """Resolve a user-provided model name or alias to a supported model."""

    normalized_name = (model_name or DEFAULT_MODEL_NAME).strip().lower()
    resolved_name = _SUPPORTED_MODEL_ALIASES.get(normalized_name)
    if resolved_name is None:
        raise InputValidationError(
            "Unsupported model name. Choose one of: "
            f"{', '.join(get_supported_model_names())}."
        )
    return resolved_name


def build_chat_model(model_name: str):
    """Build a LangChain chat model instance for the selected provider model."""

    try:
        from langchain.chat_models import init_chat_model
    except ImportError as error:  # pragma: no cover - depends on optional runtime dependency.
        raise DependencyUnavailableError(
            "langchain is required to build the chat model. Install the app dependencies first."
        ) from error

    return init_chat_model(model=normalize_model_name(model_name))
