"""PyMuPDF page rendering for the indexing pipeline."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from indexer.shared.errors import DependencyUnavailableError, IndexingRuntimeError
from indexer.shared.models import RenderedPage

try:
    import fitz
except ImportError:  # pragma: no cover - depends on optional runtime dependency.
    fitz = None


class PdfPageRenderer:
    """Renders PDF pages to PIL images using PyMuPDF."""

    def __init__(self, zoom: float) -> None:
        self._zoom = zoom

    def render(self, pdf_path: Path) -> Iterator[RenderedPage]:
        """Yield rendered pages for the given PDF path."""

        if fitz is None:
            raise DependencyUnavailableError(
                "PyMuPDF is required for PDF rendering. Install 'pymupdf'."
            )

        try:
            with fitz.open(pdf_path) as document:
                matrix = fitz.Matrix(self._zoom, self._zoom)
                for loop_index, page in enumerate(document):
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    image_bytes = pixmap.tobytes("png")
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        rgb_image = image.convert("RGB")
                        yield RenderedPage(
                            page_number=loop_index,
                            width=rgb_image.width,
                            height=rgb_image.height,
                            image=rgb_image,
                        )
        except Exception as error:  # pragma: no cover - depends on external PDFs.
            raise IndexingRuntimeError(
                f"Failed to render PDF '{pdf_path.name}': {error}"
            ) from error


def save_rendered_page_image(
    page_image_dir: Path,
    source_sha256: str,
    rendered_page: RenderedPage,
) -> Path:
    """Persist one rendered page image to a stable PNG artifact path."""

    image_directory = page_image_dir / source_sha256
    image_path = image_directory / f"{rendered_page.page_number}.png"
    try:
        image_directory.mkdir(parents=True, exist_ok=True)
        rendered_page.image.save(image_path, format="PNG")
    except Exception as error:  # pragma: no cover - depends on filesystem runtime.
        raise IndexingRuntimeError(
            "Failed to persist rendered page image "
            f"for page {rendered_page.page_number}: {error}"
        ) from error
    return image_path
