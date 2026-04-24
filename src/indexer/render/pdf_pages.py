"""PyMuPDF page rendering for the indexing pipeline."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

from PIL import Image

from indexer.shared.errors import DependencyUnavailableError, IndexingRuntimeError
from indexer.shared.models import RenderedPage

try:
    import fitz
except ImportError:  # pragma: no cover - exercised in runtime environments without pymupdf.
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
                            page_number=loop_index + 1,
                            width=rgb_image.width,
                            height=rgb_image.height,
                            image=rgb_image,
                        )
        except Exception as error:  # pragma: no cover - depends on external PDFs.
            raise IndexingRuntimeError(
                f"Failed to render PDF '{pdf_path.name}': {error}"
            ) from error
