"""Evaluation runner for the retrieval agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agent.agent_results_parser import RetrievalCitation, RetrievalToolCall
from app.agent.config import AppSettings
from app.agent.graph import DeepAgentChatService
from eval.judges import AnswerJudge, JudgeInput
from eval.metrics import (
    build_evaluation_summary,
    compute_hit_rate_at_k,
    compute_initial_recall_at_k,
    compute_rerank_ndcg_at_k,
    compute_rerank_recall_at_k,
)
from eval.schemas import (
    AnswerEvaluationRecord,
    CitationRecord,
    EvaluationQuestionResult,
    FlattenedQuestionRecord,
    RawDocumentRecord,
    RetrievalToolCallRecord,
)
from indexer.shared.errors import (
    DependencyUnavailableError,
    IndexingRuntimeError,
    InputValidationError,
)
from indexer.shared.logging_utils import configure_logging, get_logger, log_event

try:
    import fitz
except ImportError:
    fitz = None

LOGGER = get_logger(__name__)
EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = EVAL_ROOT / "data" / "dataset.jsonl"
DEFAULT_RESULTS_PATH = EVAL_ROOT / "artifacts" / "question_results.json"
DEFAULT_SUMMARY_PATH = EVAL_ROOT / "artifacts" / "summary.json"


def main() -> None:
    """Run the evaluation CLI."""

    args = _parse_args()
    settings = AppSettings()
    settings.apply_runtime_environment()
    configure_logging(settings.log_level)
    question_results = run_evaluation(
        dataset_path=args.dataset_path,
        results_path=args.results_path,
        summary_path=args.summary_path,
        k=args.k,
        judge_model_name=args.judge_model_name,
        settings=settings,
    )
    log_event(
        LOGGER,
        "evaluation_completed",
        question_count=len(question_results),
        results_path=str(args.results_path),
        summary_path=str(args.summary_path),
    )


def run_evaluation(
    dataset_path: Path,
    results_path: Path,
    summary_path: Path,
    k: int,
    judge_model_name: str,
    settings: AppSettings | None = None,
) -> list[EvaluationQuestionResult]:
    """Run the evaluation over a raw document-level JSONL dataset."""

    if k < 1:
        raise InputValidationError("Evaluation k must be at least 1.")

    resolved_settings = settings or AppSettings()
    raw_documents = load_raw_documents(dataset_path)
    flattened_questions = flatten_documents(raw_documents)
    configure_logging(resolved_settings.log_level)
    log_event(
        LOGGER,
        "flattened_eval_dataset",
        document_count=len(raw_documents),
        question_count=len(flattened_questions),
    )

    service = DeepAgentChatService(resolved_settings)
    judge = AnswerJudge(model_name=judge_model_name)
    question_results: list[EvaluationQuestionResult] = []
    for question_record in flattened_questions:
        try:
            question_results.append(
                evaluate_question(
                    service=service,
                    judge=judge,
                    question_record=question_record,
                    k=k,
                )
            )
        except Exception as error:
            log_event(
                LOGGER,
                "evaluation_question_failed",
                doc_name=question_record.doc_name,
                question_index=question_record.question_index,
                error=str(error),
            )
            question_results.append(
                build_failed_question_result(
                    question_record=question_record,
                    model_name=judge.model_name,
                    error_message=str(error),
                )
            )

    summary = build_evaluation_summary(question_results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            [result.model_dump(mode="json") for result in question_results],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return question_results


def load_raw_documents(dataset_path: Path) -> list[RawDocumentRecord]:
    """Load raw per-document JSONL rows from disk."""

    if not dataset_path.is_file():
        raise InputValidationError(
            f"Evaluation dataset file does not exist: {dataset_path}"
        )

    raw_documents: list[RawDocumentRecord] = []
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        raw_documents.append(RawDocumentRecord.model_validate_json(stripped_line))
    return raw_documents


def flatten_documents(
    raw_documents: list[RawDocumentRecord],
) -> list[FlattenedQuestionRecord]:
    """Flatten raw document rows into question-level evaluation examples."""

    flattened_questions: list[FlattenedQuestionRecord] = []
    for raw_document in raw_documents:
        for question_index, question_record in enumerate(raw_document.questions):
            flattened_questions.append(
                FlattenedQuestionRecord(
                    doc_name=raw_document.doc_name,
                    domain=raw_document.domain,
                    question_index=question_index,
                    question=question_record.question,
                    expected_answer=question_record.answer,
                    expected_pages=list(question_record.page_id),
                    question_type=question_record.question_type,
                    layout_mapping=list(question_record.layout_mapping),
                )
            )
    return flattened_questions


def evaluate_question(
    service: DeepAgentChatService,
    judge: AnswerJudge,
    question_record: FlattenedQuestionRecord,
    k: int,
) -> EvaluationQuestionResult:
    """Evaluate one question with answer and retrieval metrics."""

    resolved_domains = _resolve_domains(service=service, domain=question_record.domain)
    trace_result = service.chat_with_trace(
        message=question_record.question,
        thread_id=None,
        model_name=None,
        domains=resolved_domains,
        doc_names=[question_record.doc_name],
    )
    retrieval_query = question_record.question
    retrieval_domains = resolved_domains
    retrieval_doc_names: list[str] | None = [question_record.doc_name]
    if trace_result.retrieval_tool_calls:
        latest_tool_call = trace_result.retrieval_tool_calls[-1]
        if latest_tool_call.query:
            retrieval_query = latest_tool_call.query
        if latest_tool_call.domains:
            retrieval_domains = list(latest_tool_call.domains)
        if latest_tool_call.doc_names:
            retrieval_doc_names = list(latest_tool_call.doc_names)
    retrieval_preview = service.preview_retrieval(
        query=retrieval_query,
        domains=retrieval_domains,
        doc_names=retrieval_doc_names,
        limit=max(k, 1),
    )
    retrieved_contexts = extract_retrieved_contexts(
        citations=trace_result.citations,
        repo_root=Path(__file__).resolve().parents[1],
    )
    answer_evaluation = judge.score(
        JudgeInput(
            question=question_record.question,
            expected_answer=question_record.expected_answer,
            model_answer=trace_result.answer,
            contexts=retrieved_contexts,
        )
    )

    coarse_retrieved_pages = [
        candidate.page_number for candidate in retrieval_preview.candidates
    ]
    reranked_pages = [result.page_number for result in retrieval_preview.results]
    return EvaluationQuestionResult(
        doc_name=question_record.doc_name,
        domain=question_record.domain,
        question_index=question_record.question_index,
        question=question_record.question,
        expected_answer=question_record.expected_answer,
        expected_pages=list(question_record.expected_pages),
        question_type=question_record.question_type,
        model_name=trace_result.model_name,
        model_answer=trace_result.answer,
        retrieval_tool_call_count=len(trace_result.retrieval_tool_calls),
        retrieval_tool_calls=[
            build_retrieval_tool_call_record(tool_call)
            for tool_call in trace_result.retrieval_tool_calls
        ],
        final_citations=[
            build_citation_record(citation)
            for citation in trace_result.citations
        ],
        coarse_retrieved_pages=coarse_retrieved_pages,
        reranked_pages=reranked_pages,
        retrieved_contexts=retrieved_contexts,
        answer_evaluation=answer_evaluation,
        initial_recall_at_k=compute_initial_recall_at_k(
            expected_pages=question_record.expected_pages,
            retrieved_pages=coarse_retrieved_pages,
            k=k,
        ),
        rerank_ndcg_at_k=compute_rerank_ndcg_at_k(
            expected_pages=question_record.expected_pages,
            reranked_pages=reranked_pages,
            k=k,
        ),
        rerank_recall_at_k=compute_rerank_recall_at_k(
            expected_pages=question_record.expected_pages,
            reranked_pages=reranked_pages,
            k=k,
        ),
        hit_rate_at_k=compute_hit_rate_at_k(
            expected_pages=question_record.expected_pages,
            reranked_pages=reranked_pages,
            k=k,
        ),
    )


def build_failed_question_result(
    question_record: FlattenedQuestionRecord,
    model_name: str,
    error_message: str,
) -> EvaluationQuestionResult:
    """Build a persisted failed-question row instead of aborting the whole run."""

    return EvaluationQuestionResult(
        doc_name=question_record.doc_name,
        domain=question_record.domain,
        question_index=question_record.question_index,
        question=question_record.question,
        expected_answer=question_record.expected_answer,
        expected_pages=list(question_record.expected_pages),
        question_type=question_record.question_type,
        model_name=model_name,
        model_answer="",
        retrieval_tool_call_count=0,
        retrieval_tool_calls=[],
        final_citations=[],
        coarse_retrieved_pages=[],
        reranked_pages=[],
        retrieved_contexts=[],
        answer_evaluation=AnswerEvaluationRecord(evaluator_error=error_message),
        initial_recall_at_k=0.0,
        rerank_ndcg_at_k=0.0,
        rerank_recall_at_k=0.0,
        hit_rate_at_k=0.0,
    )


def extract_retrieved_contexts(
    citations: list[RetrievalCitation],
    repo_root: Path,
) -> list[str]:
    """Materialize text contexts for retrieved pages from their source PDFs."""

    grouped_citations: dict[str, list[int]] = {}
    for citation in citations:
        grouped_citations.setdefault(citation.file_path, []).append(
            citation.page_number
        )

    contexts: list[str] = []
    for file_path, page_numbers in grouped_citations.items():
        contexts.extend(
            _extract_pdf_page_texts(
                pdf_path=repo_root / file_path,
                page_numbers=page_numbers,
            )
        )
    return contexts


def build_citation_record(citation: RetrievalCitation) -> CitationRecord:
    """Convert one runtime citation into a persisted schema object."""

    return CitationRecord.model_validate(citation.__dict__)


def build_retrieval_tool_call_record(
    tool_call: RetrievalToolCall,
) -> RetrievalToolCallRecord:
    """Convert one runtime retrieval tool call into a persisted schema object."""

    return RetrievalToolCallRecord(
        query=tool_call.query,
        domains=tool_call.domains,
        doc_names=tool_call.doc_names,
        citations=[build_citation_record(citation) for citation in tool_call.citations],
    )


def _extract_pdf_page_texts(
    pdf_path: Path,
    page_numbers: list[int],
) -> list[str]:
    if fitz is None:
        raise DependencyUnavailableError(
            "PyMuPDF is required for evaluation context extraction. Install 'pymupdf'."
        )
    if not pdf_path.is_file():
        return []

    ordered_page_numbers = sorted(
        {page_number for page_number in page_numbers if page_number >= 0}
    )
    contexts: list[str] = []
    try:
        with fitz.open(pdf_path) as document:
            for page_number in ordered_page_numbers:
                if page_number >= len(document):
                    continue
                page = document.load_page(page_number)
                page_text = page.get_text("text").strip()
                if page_text:
                    contexts.append(page_text)
    except Exception as error:
        raise IndexingRuntimeError(
            f"Failed to extract evaluation context from '{pdf_path}': {error}"
        ) from error
    return contexts


def _resolve_domains(
    service: DeepAgentChatService,
    domain: str,
) -> list[str] | None:
    if domain in service.available_domains:
        return [domain]
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation for the retrieval agent."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--judge-model-name",
        default="anthropic:claude-sonnet-4-6",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
