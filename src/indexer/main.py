"""CLI entry point for the ColPali Qdrant indexing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from indexer.flatten.page_patches import build_page_point
from indexer.index_report import IndexReportWriter
from indexer.load_docs.domain_mapping import load_domain_mapping
from indexer.load_docs.targets import resolve_target_documents
from indexer.render.pdf_pages import save_rendered_page_image
from indexer.shared.config import Settings
from indexer.shared.errors import IndexerError, InputValidationError
from indexer.shared.logging_utils import configure_logging, get_logger, log_event
from indexer.shared.models import TargetDocument
from indexer.shared.utils import utc_now_iso
from indexer.validate.inputs import find_mapping_gaps, validate_target_files

if TYPE_CHECKING:
    from indexer.encode.colpali import ColPaliPageEncoder
    from indexer.insert.qdrant_writer import QdrantInsertWriter
    from indexer.render.pdf_pages import PdfPageRenderer
    from indexer.shared.models import RenderedPage

LOGGER = get_logger(__name__)


class IndexingService:
    """Coordinates validation, encoding, and insertion for document indexing."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._report_writer: IndexReportWriter | None = None
        self._renderer: PdfPageRenderer | None = None
        self._encoder: ColPaliPageEncoder | None = None
        self._writer: QdrantInsertWriter | None = None

    def index(self, file_name: str | None) -> int:
        """Index one file or all files and return the inserted row count."""

        mapping = load_domain_mapping()
        validate_target_files(self._settings.data_dir, mapping, file_name)
        target_documents = resolve_target_documents(
            self._settings.data_dir,
            mapping,
            file_name,
        )
        encoder = self._get_encoder()
        writer = self._get_writer()
        writer.ensure_collection(
            recreate_collection=self._settings.recreate_collection,
            vector_dimension=encoder.embedding_dimension,
        )

        run_id = uuid.uuid4().hex
        total_rows = 0
        for target_document in target_documents:
            log_event(
                LOGGER,
                "index_document_started",
                doc_name=target_document.doc_name,
                domain=target_document.domain,
            )
            try:
                page_count, inserted_rows = self._index_document(
                    target_document,
                    run_id,
                )
            except Exception as error:
                self._get_report_writer().record_failure(
                    target_document=target_document,
                    page_count=0,
                    error_message=str(error),
                )
                raise

            total_rows += inserted_rows
            self._get_report_writer().record_success(
                target_document=target_document,
                page_count=page_count,
            )
            log_event(
                LOGGER,
                "index_document_completed",
                doc_name=target_document.doc_name,
                page_count=page_count,
                inserted_rows=inserted_rows,
            )

        return total_rows

    def validate(self, file_name: str | None) -> list[str]:
        """Validate inputs and return the selected PDF names."""

        mapping = load_domain_mapping()
        target_paths = validate_target_files(
            self._settings.data_dir,
            mapping,
            file_name,
        )
        return [path.name for path in target_paths]

    def show_mapping_gaps(self) -> list[str]:
        """Return PDFs present in data/ but missing from the domain mapping."""

        mapping = load_domain_mapping()
        return find_mapping_gaps(self._settings.data_dir, mapping)

    def _index_document(
        self,
        target_document: TargetDocument,
        run_id: str,
    ) -> tuple[int, int]:
        renderer = self._get_renderer()
        encoder = self._get_encoder()
        writer = self._get_writer()
        indexed_at = utc_now_iso()
        page_count = 0
        inserted_rows = 0

        for rendered_page in renderer.render(target_document.file_path):
            page_count += 1
            page_image_path = self._save_page_image(
                target_document=target_document,
                rendered_page=rendered_page,
            )
            patch_embeddings = encoder.encode_page(rendered_page.image)
            point = build_page_point(
                target_document=target_document,
                rendered_page=rendered_page,
                page_image_path=page_image_path,
                patch_embeddings=patch_embeddings,
                indexed_at=indexed_at,
                run_id=run_id,
            )
            inserted_rows += writer.upsert_points([point])
            log_event(
                LOGGER,
                "index_page_completed",
                doc_name=target_document.doc_name,
                page_number=rendered_page.page_number,
                patch_count=len(point.embeddings),
            )

        return page_count, inserted_rows

    def _get_renderer(self) -> PdfPageRenderer:
        if self._renderer is None:
            from indexer.render.pdf_pages import PdfPageRenderer

            self._renderer = PdfPageRenderer(zoom=self._settings.render_zoom)
        return self._renderer

    def _get_encoder(self) -> ColPaliPageEncoder:
        if self._encoder is None:
            from indexer.encode.colpali import ColPaliPageEncoder

            self._encoder = ColPaliPageEncoder(
                model_name=self._settings.model_name,
                device=self._settings.resolved_device(),
            )
        return self._encoder

    def _get_writer(self) -> QdrantInsertWriter:
        if self._writer is None:
            from indexer.insert.qdrant_writer import QdrantInsertWriter

            self._writer = QdrantInsertWriter(
                db_path=str(self._settings.qdrant_path),
                collection_name=self._settings.collection_name,
            )
        return self._writer

    def _save_page_image(
        self,
        target_document: TargetDocument,
        rendered_page: RenderedPage,
    ) -> str:
        image_path = save_rendered_page_image(
            page_image_dir=self._settings.page_image_dir,
            source_sha256=target_document.source_sha256,
            rendered_page=rendered_page,
        )
        log_event(
            LOGGER,
            "page_image_persisted",
            doc_name=target_document.doc_name,
            page_number=rendered_page.page_number,
            page_image_path=str(image_path),
        )
        return str(image_path)

    def _get_report_writer(self) -> IndexReportWriter:
        if self._report_writer is None:
            self._report_writer = IndexReportWriter(self._settings.report_path)
        return self._report_writer


def _write_stdout(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")


def _write_stderr(payload: dict[str, object]) -> None:
    sys.stderr.write(json.dumps(payload) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the indexing CLI."""

    parser = argparse.ArgumentParser(prog="indexer.main")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("index", "validate"):
        command_parser = subparsers.add_parser(command_name)
        target_group = command_parser.add_mutually_exclusive_group(required=True)
        target_group.add_argument("--all", action="store_true")
        target_group.add_argument("--file", type=str)
        if command_name == "index":
            command_parser.add_argument(
                "--recreate-collection",
                action="store_true",
                help="Drop and recreate the Qdrant collection before indexing.",
            )

    subparsers.add_parser("show-mapping-gaps")
    subparsers.add_parser("describe-collection")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Run the CLI command and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    settings = Settings()
    if getattr(args, "recreate_collection", False):
        settings = settings.with_overrides(recreate_collection=True)

    configure_logging(settings.log_level)

    try:
        if args.command == "index":
            service = IndexingService(settings)
            file_name = args.file if not args.all else None
            inserted_rows = service.index(file_name=file_name)
            _write_stdout({"inserted_rows": inserted_rows})
            return 0

        if args.command == "validate":
            service = IndexingService(settings)
            file_name = args.file if not args.all else None
            selected_files = service.validate(file_name=file_name)
            _write_stdout({"selected_files": selected_files})
            return 0

        if args.command == "show-mapping-gaps":
            service = IndexingService(settings)
            gaps = service.show_mapping_gaps()
            _write_stdout({"mapping_gaps": gaps})
            return 0

        if args.command == "describe-collection":
            _write_stdout(
                {
                    "collection_name": settings.collection_name,
                    "qdrant_path": str(settings.qdrant_path),
                }
            )
            return 0
    except IndexerError as error:
        log_event(LOGGER, "indexer_error", error=str(error))
        _write_stderr({"error": str(error)})
        return 1
    except Exception as error:
        log_event(LOGGER, "unexpected_error", error=str(error))
        _write_stderr({"error": str(error)})
        return 1

    raise InputValidationError(f"Unsupported command '{args.command}'.")


if __name__ == "__main__":
    raise SystemExit(run())
