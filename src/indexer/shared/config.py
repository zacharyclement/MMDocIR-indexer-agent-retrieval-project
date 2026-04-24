"""Configuration models for the indexing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings for the indexing pipeline."""

    model_config = SettingsConfigDict(env_prefix="INDEXER_", extra="ignore")

    data_dir: Path = Field(default=PACKAGE_ROOT / "data")
    qdrant_path: Path = Field(default=Path("qdrant_data"))
    collection_name: str = Field(default="colpali_page_patches")
    model_name: str = Field(default="vidore/colqwen2-v1.0")
    device: str = Field(default="auto")
    render_zoom: float = Field(default=2.0)
    batch_size_pages: int = Field(default=1)
    recreate_collection: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    report_path: Path = Field(default=Path("artifacts/index_report.jsonl"))

    def resolved_device(self) -> str:
        """Resolve the torch device name to use for inference."""

        if self.device != "auto":
            return self.device

        try:
            import torch
        except ImportError:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def with_overrides(self, **overrides: Any) -> "Settings":
        """Return a copy of the settings with CLI overrides applied."""

        return self.model_copy(update=overrides)
