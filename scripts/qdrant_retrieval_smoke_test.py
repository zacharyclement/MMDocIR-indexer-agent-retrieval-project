from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from indexer.shared.config import Settings
from indexer.shared.errors import IndexerError

EXPECTED_PAYLOAD_FIELDS: tuple[str, ...] = (
    "doc_name",
    "domain",
    "page_number",
    "page_uid",
    "file_path",
    "source_sha256",
    "page_width",
    "page_height",
    "indexed_at",
    "run_id",
)


class SmokeTestError(IndexerError):
    """Raised when the retrieval smoke test detects invalid state or shape."""


class StepTimeoutError(SmokeTestError):
    """Raised when a smoke-test step exceeds its configured timeout."""


@dataclass(frozen=True)
class PointSummary:
    """Represents a validated page point or retrieval hit."""

    doc_name: str
    domain: str
    page_number: int
    page_uid: str
    file_path: str
    source_sha256: str
    page_width: int
    page_height: int
    indexed_at: str
    run_id: str
    patch_count: int | None = None
    embedding_dimension: int | None = None
    score: float | None = None


@dataclass
class StepTimer:
    """Tracks elapsed time for one visible smoke-test step."""

    step_name: str
    _started_at: float = field(init=False, repr=False)

    def __enter__(self) -> "StepTimer":
        self._started_at = time.perf_counter()
        emit_event("step_started", step=self.step_name)
        return self

    def progress(self, message: str, **payload: object) -> None:
        """Emit a progress update for the current step."""

        emit_event(
            "step_progress",
            step=self.step_name,
            elapsed_seconds=round(self.elapsed_seconds, 3),
            message=message,
            **payload,
        )

    @property
    def elapsed_seconds(self) -> float:
        """Return the elapsed seconds since this step started."""

        return time.perf_counter() - self._started_at

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
        if exc is None:
            emit_event(
                "step_completed",
                step=self.step_name,
                elapsed_seconds=round(self.elapsed_seconds, 3),
            )
            return False

        emit_event(
            "step_failed",
            step=self.step_name,
            elapsed_seconds=round(self.elapsed_seconds, 3),
            error=str(exc),
        )
        return False


@dataclass(frozen=True)
class ScriptConfig:
    """Holds runtime options for the retrieval smoke test."""

    qdrant_path: Path
    collection_name: str
    report_path: Path
    sample_size: int
    query_limit: int
    timeout_seconds: int | None


@dataclass(frozen=True)
class ReportEntry:
    """Represents one successful index-report record keyed by document name."""

    doc_name: str
    file_path: str
    domain: str
    file_hash: str


@dataclass(frozen=True)
class CollectionSummary:
    """Represents basic collection metadata surfaced by Qdrant."""

    status: str | None
    point_count: int | None
    vector_size: int | None


def deadline(timeout_seconds: int | None, operation_name: str) -> tuple[object, float] | None:
    """Start a process-level deadline for one Qdrant operation."""

    if timeout_seconds is None:
        return None

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return None

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _timeout_handler(signum: int, frame: object) -> None:
        raise StepTimeoutError(
            f"{operation_name} exceeded {timeout_seconds} seconds."
        )

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    return previous_handler, float(timeout_seconds)


def cancel_deadline(deadline_state: tuple[object, float] | None) -> None:
    """Cancel and restore a previously started deadline."""

    if deadline_state is None:
        return

    previous_handler, _ = deadline_state
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, previous_handler)


def run_with_deadline(
    timeout_seconds: int | None,
    operation_name: str,
    callback: Any,
) -> Any:
    """Execute one callback while enforcing the configured deadline."""

    deadline_state = deadline(timeout_seconds, operation_name)
    try:
        return callback()
    finally:
        cancel_deadline(deadline_state)


def emit_event(event: str, **payload: object) -> None:
    """Write one structured progress event to stdout."""

    event_payload = {"event": event, **payload}
    sys.stdout.write(json.dumps(event_payload, default=str) + "\n")
    sys.stdout.flush()


def positive_int(value: str) -> int:
    """Parse a positive integer command-line argument."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer command-line argument."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative integer.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the smoke-test script."""

    settings = Settings()
    parser = argparse.ArgumentParser(prog="qdrant_retrieval_smoke_test")
    parser.add_argument("--qdrant-path", type=Path, default=settings.qdrant_path)
    parser.add_argument("--collection-name", type=str, default=settings.collection_name)
    parser.add_argument("--report-path", type=Path, default=settings.report_path)
    parser.add_argument("--sample-size", type=positive_int, default=3)
    parser.add_argument("--query-limit", type=positive_int, default=5)
    parser.add_argument(
        "--timeout-seconds",
        type=non_negative_int,
        default=15,
        help="Per-Qdrant-call timeout. Set to 0 to disable timeouts.",
    )
    return parser


def build_config(args: argparse.Namespace) -> ScriptConfig:
    """Translate parsed CLI arguments into a validated script config."""

    timeout_seconds = args.timeout_seconds if args.timeout_seconds > 0 else None
    return ScriptConfig(
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
        report_path=args.report_path,
        sample_size=args.sample_size,
        query_limit=args.query_limit,
        timeout_seconds=timeout_seconds,
    )


def build_client(qdrant_path: Path) -> Any:
    """Create a local-mode Qdrant client for the configured path."""

    return QdrantClient(path=str(qdrant_path))


def load_success_reports(report_path: Path) -> dict[str, ReportEntry]:
    """Load successful index-report rows keyed by document name."""

    if not report_path.exists():
        return {}

    report_entries: dict[str, ReportEntry] = {}
    with report_path.open("r", encoding="utf-8") as file_handle:
        for line_number, raw_line in enumerate(file_handle, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            try:
                payload = json.loads(stripped_line)
            except json.JSONDecodeError as error:
                raise SmokeTestError(
                    f"Invalid JSON in report file '{report_path}' on line {line_number}: {error}"
                ) from error

            if payload.get("status") != "success":
                continue

            doc_name = require_non_empty_string(payload.get("doc_name"), "doc_name")
            domain = require_non_empty_string(payload.get("domain"), "domain")
            file_path = require_non_empty_string(payload.get("file_path"), "file_path")
            file_hash = require_non_empty_string(payload.get("file_hash"), "file_hash")
            report_entries[doc_name] = ReportEntry(
                doc_name=doc_name,
                file_path=file_path,
                domain=domain,
                file_hash=file_hash,
            )
    return report_entries


def require_non_empty_string(value: object, field_name: str) -> str:
    """Return a validated non-empty string field."""

    if not isinstance(value, str) or not value.strip():
        raise SmokeTestError(f"Field '{field_name}' must be a non-empty string.")
    return value


def require_positive_int(value: object, field_name: str) -> int:
    """Return a validated positive integer field."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SmokeTestError(f"Field '{field_name}' must be a positive integer.")
    return value


def require_score(value: object) -> float:
    """Return a validated numeric score from a retrieval hit."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SmokeTestError("Retrieval hit score must be numeric.")
    return float(value)


def extract_multivector(vector: object) -> list[list[float]]:
    """Normalize a Qdrant vector payload into a multivector list of floats."""

    if isinstance(vector, Mapping):
        if not vector:
            raise SmokeTestError("Returned vector mapping was empty.")
        first_value = next(iter(vector.values()))
        return extract_multivector(first_value)

    if not isinstance(vector, list):
        raise SmokeTestError("Returned vector was not a multivector list.")
    if not vector:
        raise SmokeTestError("Returned multivector was empty.")

    normalized_vectors: list[list[float]] = []
    expected_dimension: int | None = None
    for row in vector:
        if not isinstance(row, list) or not row:
            raise SmokeTestError("Each multivector row must be a non-empty list.")
        normalized_row = [normalize_numeric_value(entry) for entry in row]
        if expected_dimension is None:
            expected_dimension = len(normalized_row)
        elif len(normalized_row) != expected_dimension:
            raise SmokeTestError("Returned multivector rows did not share one dimension.")
        normalized_vectors.append(normalized_row)
    return normalized_vectors


def normalize_numeric_value(value: object) -> float:
    """Normalize an integer or float vector element into a float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SmokeTestError("Vector entries must be numeric.")
    return float(value)


def extract_collection_summary(collection_info: object) -> CollectionSummary:
    """Extract basic collection metadata from the Qdrant collection response."""

    status_value = getattr(collection_info, "status", None)
    status = str(status_value) if status_value is not None else None

    point_count_value = getattr(collection_info, "points_count", None)
    point_count = point_count_value if isinstance(point_count_value, int) else None

    vector_size: int | None = None
    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)

    if isinstance(vectors, Mapping):
        if vectors:
            first_vectors_value = next(iter(vectors.values()))
            vector_size = extract_vector_size(first_vectors_value)
    else:
        vector_size = extract_vector_size(vectors)

    return CollectionSummary(
        status=status,
        point_count=point_count,
        vector_size=vector_size,
    )


def extract_vector_size(vector_config: object) -> int | None:
    """Extract the configured embedding dimension from a vector config object."""

    size = getattr(vector_config, "size", None)
    if isinstance(size, int) and size > 0:
        return size
    return None


def build_point_summary(
    payload: Mapping[str, object],
    multivector: list[list[float]] | None,
    score: float | None,
) -> PointSummary:
    """Validate one point payload and convert it into a summary object."""

    missing_fields = [field_name for field_name in EXPECTED_PAYLOAD_FIELDS if field_name not in payload]
    if missing_fields:
        raise SmokeTestError(
            f"Returned payload is missing expected fields: {', '.join(sorted(missing_fields))}"
        )

    doc_name = require_non_empty_string(payload.get("doc_name"), "doc_name")
    domain = require_non_empty_string(payload.get("domain"), "domain")
    page_number = require_positive_int(payload.get("page_number"), "page_number")
    page_uid = require_non_empty_string(payload.get("page_uid"), "page_uid")
    file_path = require_non_empty_string(payload.get("file_path"), "file_path")
    source_sha256 = require_non_empty_string(payload.get("source_sha256"), "source_sha256")
    page_width = require_positive_int(payload.get("page_width"), "page_width")
    page_height = require_positive_int(payload.get("page_height"), "page_height")
    indexed_at = require_non_empty_string(payload.get("indexed_at"), "indexed_at")
    run_id = require_non_empty_string(payload.get("run_id"), "run_id")

    expected_page_uid = f"{doc_name}::page::{page_number}"
    if page_uid != expected_page_uid:
        raise SmokeTestError(
            f"Payload page_uid '{page_uid}' did not match expected '{expected_page_uid}'."
        )

    patch_count: int | None = None
    embedding_dimension: int | None = None
    if multivector is not None:
        patch_count = len(multivector)
        embedding_dimension = len(multivector[0])

    return PointSummary(
        doc_name=doc_name,
        domain=domain,
        page_number=page_number,
        page_uid=page_uid,
        file_path=file_path,
        source_sha256=source_sha256,
        page_width=page_width,
        page_height=page_height,
        indexed_at=indexed_at,
        run_id=run_id,
        patch_count=patch_count,
        embedding_dimension=embedding_dimension,
        score=score,
    )


def validate_against_report(
    point_summary: PointSummary,
    report_entries: Mapping[str, ReportEntry],
) -> None:
    """Cross-check one validated point against the index report when available."""

    report_entry = report_entries.get(point_summary.doc_name)
    if report_entry is None:
        return

    if point_summary.file_path != report_entry.file_path:
        raise SmokeTestError(
            f"Payload file_path for '{point_summary.doc_name}' did not match the report."
        )
    if point_summary.domain != report_entry.domain:
        raise SmokeTestError(
            f"Payload domain for '{point_summary.doc_name}' did not match the report."
        )
    if point_summary.source_sha256 != report_entry.file_hash:
        raise SmokeTestError(
            f"Payload source_sha256 for '{point_summary.doc_name}' did not match the report hash."
        )


def validate_vector_shape(point_summary: PointSummary, expected_dimension: int | None) -> None:
    """Validate multivector dimensions against collection metadata when available."""

    if point_summary.patch_count is None or point_summary.embedding_dimension is None:
        raise SmokeTestError("Expected a point summary with vector details attached.")
    if point_summary.patch_count <= 0:
        raise SmokeTestError("Returned point multivector contained zero patches.")
    if point_summary.embedding_dimension <= 0:
        raise SmokeTestError("Returned point multivector contained zero dimensions.")
    if expected_dimension is not None and point_summary.embedding_dimension != expected_dimension:
        raise SmokeTestError(
            "Returned multivector dimension did not match the collection configuration."
        )


def validate_sample_record(
    record: object,
    report_entries: Mapping[str, ReportEntry],
    expected_dimension: int | None,
) -> tuple[PointSummary, list[list[float]]]:
    """Validate one scrolled Qdrant record and return its summary and vector."""

    payload = getattr(record, "payload", None)
    if not isinstance(payload, Mapping):
        raise SmokeTestError("Scrolled record did not include a payload mapping.")

    multivector = extract_multivector(getattr(record, "vector", None))
    point_summary = build_point_summary(payload, multivector, score=None)
    validate_vector_shape(point_summary, expected_dimension)
    validate_against_report(point_summary, report_entries)
    return point_summary, multivector


def validate_query_hit(
    hit: object,
    report_entries: Mapping[str, ReportEntry],
) -> PointSummary:
    """Validate one retrieval hit returned by Qdrant."""

    payload = getattr(hit, "payload", None)
    if not isinstance(payload, Mapping):
        raise SmokeTestError("Retrieval hit did not include a payload mapping.")

    score = require_score(getattr(hit, "score", None))
    point_summary = build_point_summary(payload, multivector=None, score=score)
    validate_against_report(point_summary, report_entries)
    return point_summary


def run_smoke_test(config: ScriptConfig) -> dict[str, object]:
    """Execute the collection inspection and retrieval smoke test."""

    overall_started_at = time.perf_counter()

    with StepTimer("connect_qdrant") as step:
        step.progress(
            "opening local Qdrant client",
            qdrant_path=str(config.qdrant_path),
            collection_name=config.collection_name,
        )
        client = run_with_deadline(
            config.timeout_seconds,
            "opening local Qdrant client",
            lambda: build_client(config.qdrant_path),
        )

    with StepTimer("load_index_report") as step:
        step.progress("loading successful index report entries", report_path=str(config.report_path))
        report_entries = load_success_reports(config.report_path)
        step.progress(
            "loaded index report entries",
            success_record_count=len(report_entries),
        )

    with StepTimer("inspect_collection") as step:
        step.progress(
            "fetching collection metadata",
            collection_name=config.collection_name,
            timeout_seconds=config.timeout_seconds,
        )
        collection_info = run_with_deadline(
            config.timeout_seconds,
            "fetching collection metadata",
            lambda: client.get_collection(
                collection_name=config.collection_name,
            ),
        )
        collection_summary = extract_collection_summary(collection_info)
        step.progress(
            "collection metadata loaded",
            status=collection_summary.status,
            point_count=collection_summary.point_count,
            vector_size=collection_summary.vector_size,
        )

    with StepTimer("scroll_sample_points") as step:
        step.progress(
            "scrolling sample points",
            sample_size=config.sample_size,
            with_vectors=True,
            timeout_seconds=config.timeout_seconds,
        )
        records, next_offset = run_with_deadline(
            config.timeout_seconds,
            "scrolling sample points",
            lambda: client.scroll(
                collection_name=config.collection_name,
                limit=config.sample_size,
                with_payload=True,
                with_vectors=True,
            ),
        )
        if not records:
            raise SmokeTestError(
                f"Collection '{config.collection_name}' was reachable but returned zero records."
            )
        step.progress(
            "sample points received",
            returned_records=len(records),
            next_offset=next_offset,
        )

    with StepTimer("validate_sample_points") as step:
        sample_summaries: list[PointSummary] = []
        query_vector: list[list[float]] | None = None
        for index, record in enumerate(records, start=1):
            point_summary, multivector = validate_sample_record(
                record,
                report_entries=report_entries,
                expected_dimension=collection_summary.vector_size,
            )
            sample_summaries.append(point_summary)
            if query_vector is None:
                query_vector = multivector
            step.progress(
                "validated sample point",
                index=index,
                total=len(records),
                doc_name=point_summary.doc_name,
                page_number=point_summary.page_number,
                patch_count=point_summary.patch_count,
                embedding_dimension=point_summary.embedding_dimension,
            )
        if query_vector is None:
            raise SmokeTestError("Failed to capture a stored vector for retrieval testing.")

    with StepTimer("query_nearest_neighbors") as step:
        step.progress(
            "querying with one stored page multivector",
            query_limit=config.query_limit,
            query_patch_count=len(query_vector),
            query_embedding_dimension=len(query_vector[0]),
            timeout_seconds=config.timeout_seconds,
        )
        query_response = run_with_deadline(
            config.timeout_seconds,
            "querying nearest neighbors",
            lambda: client.query_points(
                collection_name=config.collection_name,
                query=query_vector,
                limit=config.query_limit,
                with_payload=True,
                with_vectors=False,
            ),
        )
        hits = getattr(query_response, "points", None)
        if not isinstance(hits, Sequence) or not hits:
            raise SmokeTestError("Nearest-neighbor query returned no points.")
        step.progress("nearest-neighbor query returned points", hit_count=len(hits))

    with StepTimer("validate_query_hits") as step:
        hit_summaries: list[PointSummary] = []
        for index, hit in enumerate(hits, start=1):
            point_summary = validate_query_hit(hit, report_entries=report_entries)
            hit_summaries.append(point_summary)
            step.progress(
                "validated retrieval hit",
                index=index,
                total=len(hits),
                doc_name=point_summary.doc_name,
                page_number=point_summary.page_number,
                score=point_summary.score,
            )

        expected_page_uid = sample_summaries[0].page_uid
        returned_page_uids = {hit.page_uid for hit in hit_summaries}
        if expected_page_uid not in returned_page_uids:
            raise SmokeTestError(
                "Querying with a stored vector did not return the sampled page among the top hits."
            )
        step.progress(
            "retrieval validated against sampled page",
            sampled_page_uid=expected_page_uid,
        )

    return {
        "collection_name": config.collection_name,
        "qdrant_path": str(config.qdrant_path),
        "report_path": str(config.report_path),
        "collection": asdict(collection_summary),
        "validated_sample_count": len(sample_summaries),
        "sample_points": [asdict(summary) for summary in sample_summaries],
        "retrieval_hit_count": len(hit_summaries),
        "retrieval_hits": [asdict(summary) for summary in hit_summaries],
        "elapsed_seconds": round(time.perf_counter() - overall_started_at, 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Qdrant retrieval smoke test and return an exit code."""

    overall_started_at = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)

    emit_event(
        "smoke_test_started",
        collection_name=config.collection_name,
        qdrant_path=str(config.qdrant_path),
        report_path=str(config.report_path),
        sample_size=config.sample_size,
        query_limit=config.query_limit,
        timeout_seconds=config.timeout_seconds,
    )

    try:
        summary = run_smoke_test(config)
    except Exception as error:
        emit_event(
            "smoke_test_failed",
            error=str(error),
            elapsed_seconds=round(time.perf_counter() - overall_started_at, 3),
        )
        return 1

    emit_event("smoke_test_completed", **summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
