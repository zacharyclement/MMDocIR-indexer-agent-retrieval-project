"""Retrieval encoder helpers for text-query embeddings."""

from __future__ import annotations

from indexer.encode.colpali import ColPaliPageEncoder
from indexer.shared.config import Settings as IndexerSettings


def resolve_retrieval_device(device_name: str) -> str:
    """Resolve the configured retrieval device into a torch device name."""

    return IndexerSettings(device=device_name).resolved_device()


def build_retrieval_encoder(model_name: str, device_name: str) -> ColPaliPageEncoder:
    """Build the shared ColPali-family encoder used for retrieval."""

    return ColPaliPageEncoder(
        model_name=model_name,
        device=resolve_retrieval_device(device_name),
    )
