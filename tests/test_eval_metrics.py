"""Tests for evaluation metric helpers."""

from __future__ import annotations

from eval.metrics import (
    build_evaluation_summary,
    compute_hit_rate_at_k,
    compute_initial_recall_at_k,
    compute_rerank_ndcg_at_k,
    compute_rerank_recall_at_k,
)
from eval.schemas import (
    AnswerEvaluationRecord,
    EvaluationQuestionResult,
    EvaluatorFeedbackRecord,
)


def test_retrieval_metric_helpers_compute_expected_values() -> None:
    assert compute_initial_recall_at_k([1, 2], [2, 5, 1], 2) == 0.5
    assert compute_rerank_recall_at_k([1, 2], [2, 5, 1], 3) == 1.0
    assert compute_hit_rate_at_k([1, 2], [9, 8, 2], 2) == 0.0
    ndcg_value = compute_rerank_ndcg_at_k([1, 2], [2, 5, 1], 3)
    assert 0.0 < ndcg_value <= 1.0


def test_build_evaluation_summary_aggregates_retrieval_tool_usage() -> None:
    question_results = [
        EvaluationQuestionResult(
            doc_name="doc-a.pdf",
            domain="Guidebook",
            question_index=0,
            question="Q1",
            expected_answer="A1",
            expected_pages=[1],
            question_type=("Table",),
            model_name="anthropic:claude-sonnet-4-6",
            model_answer="A1",
            retrieval_tool_call_count=1,
            retrieval_tool_calls=[],
            final_citations=[],
            coarse_retrieved_pages=[1],
            reranked_pages=[1],
            retrieved_contexts=["context"],
            answer_evaluation=AnswerEvaluationRecord(
                feedback=[
                    EvaluatorFeedbackRecord(
                        evaluator_key="correctness",
                        score=True,
                    ),
                    EvaluatorFeedbackRecord(
                        evaluator_key="groundedness",
                        score=0.8,
                    ),
                ]
            ),
            initial_recall_at_k=1.0,
            rerank_ndcg_at_k=1.0,
            rerank_recall_at_k=1.0,
            hit_rate_at_k=1.0,
        ),
        EvaluationQuestionResult(
            doc_name="doc-a.pdf",
            domain="Guidebook",
            question_index=1,
            question="Q2",
            expected_answer="A2",
            expected_pages=[2],
            question_type=("Chart",),
            model_name="anthropic:claude-sonnet-4-6",
            model_answer="A2",
            retrieval_tool_call_count=3,
            retrieval_tool_calls=[],
            final_citations=[],
            coarse_retrieved_pages=[8],
            reranked_pages=[8],
            retrieved_contexts=["context"],
            answer_evaluation=AnswerEvaluationRecord(
                feedback=[
                    EvaluatorFeedbackRecord(
                        evaluator_key="correctness",
                        score=False,
                    ),
                    EvaluatorFeedbackRecord(
                        evaluator_key="groundedness",
                        score=0.2,
                    ),
                ]
            ),
            initial_recall_at_k=0.0,
            rerank_ndcg_at_k=0.0,
            rerank_recall_at_k=0.0,
            hit_rate_at_k=0.0,
        ),
    ]

    summary = build_evaluation_summary(question_results)

    document_evaluator_summaries = (
        summary.document_metrics[0].answer_evaluator_summaries
    )
    overall_evaluator_summaries = summary.overall_metrics.answer_evaluator_summaries

    assert summary.document_metrics[0].retrieval_tool_call_count_total == 4
    assert summary.document_metrics[0].retrieval_tool_call_count_mean == 2.0
    assert document_evaluator_summaries[0].evaluator_key == "correctness"
    assert document_evaluator_summaries[0].true_rate == 0.5
    assert summary.overall_metrics.retrieval_tool_call_count_total == 4
    assert summary.overall_metrics.retrieval_tool_call_count_mean == 2.0
    assert overall_evaluator_summaries[1].evaluator_key == "groundedness"
    assert overall_evaluator_summaries[1].numeric_score_mean == 0.5
    assert summary.overall_metrics.zero_hit_question_count == 1
