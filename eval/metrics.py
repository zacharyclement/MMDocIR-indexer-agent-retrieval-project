"""Metric helpers for evaluation outputs."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, pstdev

from eval.schemas import (
    DocumentMetricsRow,
    EvaluationQuestionResult,
    EvaluationSummary,
    EvaluatorFeedbackRecord,
    EvaluatorSummaryRow,
    OverallMetricsRow,
)


def compute_initial_recall_at_k(
    expected_pages: list[int],
    retrieved_pages: list[int],
    k: int,
) -> float:
    """Compute coarse retrieval recall at k."""

    relevant_pages = set(expected_pages)
    if not relevant_pages:
        return 0.0
    retrieved_page_set = set(retrieved_pages[:k])
    return len(relevant_pages & retrieved_page_set) / len(relevant_pages)


def compute_rerank_recall_at_k(
    expected_pages: list[int],
    reranked_pages: list[int],
    k: int,
) -> float:
    """Compute reranked recall at k."""

    relevant_pages = set(expected_pages)
    if not relevant_pages:
        return 0.0
    reranked_page_set = set(reranked_pages[:k])
    return len(relevant_pages & reranked_page_set) / len(relevant_pages)


def compute_hit_rate_at_k(
    expected_pages: list[int],
    reranked_pages: list[int],
    k: int,
) -> float:
    """Compute hit rate at k using binary relevance."""

    relevant_pages = set(expected_pages)
    if not relevant_pages:
        return 0.0
    return 1.0 if relevant_pages & set(reranked_pages[:k]) else 0.0


def compute_rerank_ndcg_at_k(
    expected_pages: list[int],
    reranked_pages: list[int],
    k: int,
) -> float:
    """Compute rerank NDCG at k using binary page relevance."""

    relevant_pages = set(expected_pages)
    if not relevant_pages:
        return 0.0

    dcg = 0.0
    for index, page_number in enumerate(reranked_pages[:k]):
        relevance = 1.0 if page_number in relevant_pages else 0.0
        if relevance == 0.0:
            continue
        dcg += relevance / math.log2(index + 2)

    ideal_hits = min(len(relevant_pages), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def build_evaluation_summary(
    question_results: list[EvaluationQuestionResult],
) -> EvaluationSummary:
    """Aggregate question-level results into document and overall summary rows."""

    grouped_results: dict[
        tuple[str, str],
        list[EvaluationQuestionResult],
    ] = defaultdict(list)
    for result in question_results:
        grouped_results[(result.doc_name, result.domain)].append(result)

    document_rows = [
        _build_document_metrics_row(doc_name, domain, results)
        for (doc_name, domain), results in sorted(grouped_results.items())
    ]
    overall_row = _build_overall_metrics_row(question_results)
    return EvaluationSummary(
        document_metrics=document_rows,
        overall_metrics=overall_row,
    )


def _build_document_metrics_row(
    doc_name: str,
    domain: str,
    question_results: list[EvaluationQuestionResult],
) -> DocumentMetricsRow:
    return DocumentMetricsRow(
        doc_name=doc_name,
        domain=domain,
        question_count=len(question_results),
        answer_evaluator_summaries=_build_evaluator_summaries(question_results),
        initial_recall_at_k_mean=mean(
            result.initial_recall_at_k for result in question_results
        ),
        rerank_ndcg_at_k_mean=mean(
            result.rerank_ndcg_at_k for result in question_results
        ),
        rerank_recall_at_k_mean=mean(
            result.rerank_recall_at_k for result in question_results
        ),
        hit_rate_at_k_mean=mean(result.hit_rate_at_k for result in question_results),
        retrieval_tool_call_count_mean=mean(
            result.retrieval_tool_call_count for result in question_results
        ),
        retrieval_tool_call_count_total=sum(
            result.retrieval_tool_call_count for result in question_results
        ),
    )


def _build_overall_metrics_row(
    question_results: list[EvaluationQuestionResult],
) -> OverallMetricsRow:
    return OverallMetricsRow(
        total_documents=len({result.doc_name for result in question_results}),
        total_questions=len(question_results),
        answer_evaluator_summaries=_build_evaluator_summaries(question_results),
        initial_recall_at_k_mean=mean(
            result.initial_recall_at_k for result in question_results
        )
        if question_results
        else 0.0,
        rerank_ndcg_at_k_mean=mean(
            result.rerank_ndcg_at_k for result in question_results
        )
        if question_results
        else 0.0,
        rerank_recall_at_k_mean=mean(
            result.rerank_recall_at_k for result in question_results
        )
        if question_results
        else 0.0,
        hit_rate_at_k_mean=mean(result.hit_rate_at_k for result in question_results)
        if question_results
        else 0.0,
        retrieval_tool_call_count_mean=mean(
            result.retrieval_tool_call_count for result in question_results
        )
        if question_results
        else 0.0,
        retrieval_tool_call_count_total=sum(
            result.retrieval_tool_call_count for result in question_results
        ),
        zero_hit_question_count=sum(
            1 for result in question_results if result.hit_rate_at_k == 0.0
        ),
        failed_run_count=sum(
            1 for result in question_results if not result.model_answer.strip()
        ),
    )


def _build_evaluator_summaries(
    question_results: list[EvaluationQuestionResult],
) -> list[EvaluatorSummaryRow]:
    grouped_feedback: dict[
        tuple[str, str],
        list[EvaluatorFeedbackRecord],
    ] = defaultdict(list)
    for result in question_results:
        for feedback in result.answer_evaluation.feedback:
            grouped_feedback[(feedback.evaluator_key, feedback.source)].append(feedback)

    evaluator_summaries: list[EvaluatorSummaryRow] = []
    for (evaluator_key, source), feedback_rows in sorted(grouped_feedback.items()):
        bool_scores = [
            feedback.score
            for feedback in feedback_rows
            if isinstance(feedback.score, bool)
        ]
        numeric_scores = [
            feedback.score
            for feedback in feedback_rows
            if isinstance(feedback.score, float)
        ]
        scored_question_count = len(bool_scores) + len(numeric_scores)
        evaluator_summaries.append(
            EvaluatorSummaryRow(
                evaluator_key=evaluator_key,
                source=source,
                question_count=len(question_results),
                scored_question_count=scored_question_count,
                true_rate=_mean_nullable(
                    [1.0 if score else 0.0 for score in bool_scores]
                ),
                numeric_score_mean=_mean_nullable(numeric_scores),
                numeric_score_std=_std_nullable(numeric_scores),
            )
        )
    return evaluator_summaries


def _mean_nullable(values: list[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return mean(numeric_values)


def _std_nullable(values: list[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    if len(numeric_values) == 1:
        return 0.0
    return pstdev(numeric_values)
