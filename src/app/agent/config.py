"""Application settings for the retrieval chat POC."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.agent.llms import DEFAULT_MODEL_NAME, normalize_model_name

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_STATIC_DIR = REPO_ROOT / "src" / "app" / "server" / "static"
APP_SKILLS_DIR = REPO_ROOT / ".agents" / "skills"


class AppSettings(BaseSettings):
    """Runtime settings for the retrieval chat application."""

    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    default_model: str = Field(default=DEFAULT_MODEL_NAME)
    qdrant_path: Path = Field(default=Path("qdrant_data"))
    collection_name: str = Field(default="colpali_page_patches")
    retrieval_model_name: str = Field(default="vidore/colqwen2.5-v0.2")
    retrieval_device: str = Field(default="auto")
    retrieval_query_limit: int = Field(default=24, ge=1)
    retrieval_rerank_limit: int = Field(default=10, ge=1)
    retrieval_image_limit: int = Field(default=5, ge=1)
    static_dir: Path = Field(default=APP_STATIC_DIR)
    skills_dir: Path = Field(default=APP_SKILLS_DIR)
    langsmith_project: str = Field(default="unstructured-tests-retrieval-poc")

    @field_validator("default_model")
    @classmethod
    def validate_default_model(cls, value: str) -> str:
        """Validate the configured default chat model."""

        return normalize_model_name(value)
