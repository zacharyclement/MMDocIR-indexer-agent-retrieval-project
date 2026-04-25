"""Tests for retrieval tool content and artifact shaping."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.agent.retrieval.qdrant_search import RankedPageResult, RetrievalResponse
from app.agent.tools.retrieval import (
    _build_retrieval_artifact,
    _build_retrieval_content,
)


def test_build_retrieval_content_includes_top_five_page_images(tmp_path: Path) -> None:
    results: list[RankedPageResult] = []
    for page_number in range(6):
        image_path = tmp_path / f"{page_number}.png"
        Image.new("RGB", (4, 4), color=(page_number, 0, 0)).save(
            image_path,
            format="PNG",
        )
        results.append(
            RankedPageResult(
                doc_name="doc.pdf",
                domain="Guidebook",
                page_number=page_number,
                page_uid=f"doc.pdf::page::{page_number}",
                file_path="data/doc.pdf",
                page_image_path=str(image_path),
                source_sha256="hash-doc",
                coarse_score=1.0 - (page_number * 0.1),
                rerank_score=2.0 - (page_number * 0.1),
            )
        )

    response = RetrievalResponse(
        query="show me the chart",
        domains=("Guidebook",),
        results=results,
    )

    content = _build_retrieval_content(response=response, image_limit=5)

    image_blocks = [block for block in content if block.get("type") == "image_url"]
    assert content[0]["type"] == "text"
    assert "Retrieved 6 ranked pages" in str(content[0]["text"])
    assert len(image_blocks) == 5
    assert str(image_blocks[0]["image_url"]["url"]).startswith("data:image/png;base64,")


def test_build_retrieval_artifact_preserves_all_ranked_results(tmp_path: Path) -> None:
    image_path = tmp_path / "0.png"
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(image_path, format="PNG")
    response = RetrievalResponse(
        query="show me the chart",
        domains=("Guidebook",),
        results=[
            RankedPageResult(
                doc_name="doc.pdf",
                domain="Guidebook",
                page_number=0,
                page_uid="doc.pdf::page::0",
                file_path="data/doc.pdf",
                page_image_path=str(image_path),
                source_sha256="hash-doc",
                coarse_score=1.0,
                rerank_score=2.0,
            )
        ],
    )

    artifact = _build_retrieval_artifact(response)

    assert artifact["query"] == "show me the chart"
    assert artifact["domains"] == ["Guidebook"]
    assert artifact["results"][0]["page_image_path"] == str(image_path)
