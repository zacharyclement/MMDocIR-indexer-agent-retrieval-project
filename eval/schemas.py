"""Typed models for the evaluation workflow."""

from __future__ import annotations

import ast
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LayoutMapping(BaseModel):
    """Represents one annotated layout region for a question."""

    model_config = ConfigDict(extra="forbid")

    page: int
    page_size: tuple[float, float]
    bbox: tuple[float, float, float, float]


class RawQuestionRecord(BaseModel):
    """Represents one question annotation nested under a raw document row."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    question: str = Field(alias="Q", min_length=1)
    answer: str = Field(alias="A", min_length=1)
    page_id: list[int] = Field(min_length=1)
    question_type: tuple[str, ...] = Field(alias="type")
    layout_mapping: list[LayoutMapping] = Field(default_factory=list)

    @field_validator("question_type", mode="before")
    @classmethod
    def validate_question_type(cls, value: object) -> tuple[str, ...]:
        """Normalize raw question type values into a string tuple."""

        if isinstance(value, list):
            normalized = [
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            ]
            return tuple(normalized)
        if isinstance(value, str):
            stripped_value = value.strip()
            if not stripped_value:
                return ()
            try:
                parsed_value = ast.literal_eval(stripped_value)
            except (ValueError, SyntaxError):
                return (stripped_value,)
            if isinstance(parsed_value, list):
                normalized = [
                    item.strip()
                    for item in parsed_value
                    if isinstance(item, str) and item.strip()
                ]
                return tuple(normalized)
            return (stripped_value,)
        return ()


class RawDocumentRecord(BaseModel):
    """Represents one raw document-level JSONL row."""

    model_config = ConfigDict(extra="forbid")

    doc_name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    page_indices: list[int] = Field(default_factory=list)
    layout_indices: list[int] = Field(default_factory=list)
    questions: list[RawQuestionRecord] = Field(default_factory=list)


class FlattenedQuestionRecord(BaseModel):
    """Represents one processed question-level evaluation example."""

    model_config = ConfigDict(extra="forbid")

    doc_name: str
    domain: str
    question_index: int
    question: str
    expected_answer: str
    expected_pages: list[int]
    question_type: tuple[str, ...]
    layout_mapping: list[LayoutMapping]


class CitationRecord(BaseModel):
    """Represents one retrieved citation captured during evaluation."""

    model_config = ConfigDict(extra="forbid")

    doc_name: str
    domain: str
    page_number: int
    page_uid: str
    file_path: str
    page_image_path: str
    coarse_score: float
    rerank_score: float


class RetrievalToolCallRecord(BaseModel):
    """Represents one retrieval tool invocation captured from the agent trace."""

    model_config = ConfigDict(extra="forbid")

    query: str
    domains: tuple[str, ...]
    doc_names: tuple[str, ...]
    citations: list[CitationRecord]


class EvaluatorFeedbackRecord(BaseModel):
    """Represents one persisted evaluator feedback row."""

    model_config = ConfigDict(extra="forbid")

    evaluator_key: str
    source: Literal["openevals"] = "openevals"
    score: bool | float | None = None
    comment: str | None = None
    raw_result: str | None = None


class AnswerEvaluationRecord(BaseModel):
    """Represents all answer evaluator outputs for one question."""

    model_config = ConfigDict(extra="forbid")

    feedback: list[EvaluatorFeedbackRecord] = Field(default_factory=list)
    evaluator_error: str | None = None


class EvaluatorSummaryRow(BaseModel):
    """Represents aggregate summary statistics for one evaluator key."""

    model_config = ConfigDict(extra="forbid")

    evaluator_key: str
    source: Literal["openevals"]
    question_count: int
    scored_question_count: int
    true_rate: float | None = None
    numeric_score_mean: float | None = None
    numeric_score_std: float | None = None


class EvaluationQuestionResult(BaseModel):
    """Represents one fully evaluated question row."""

    model_config = ConfigDict(extra="forbid")

    doc_name: str
    domain: str
    question_index: int
    question: str
    expected_answer: str
    expected_pages: list[int]
    question_type: tuple[str, ...]
    model_name: str
    model_answer: str
    retrieval_tool_call_count: int
    retrieval_tool_calls: list[RetrievalToolCallRecord]
    final_citations: list[CitationRecord]
    coarse_retrieved_pages: list[int]
    reranked_pages: list[int]
    retrieved_contexts: list[str]
    answer_evaluation: AnswerEvaluationRecord
    initial_recall_at_k: float
    rerank_ndcg_at_k: float
    rerank_recall_at_k: float
    hit_rate_at_k: float


class DocumentMetricsRow(BaseModel):
    """Represents one document-level aggregate row for reporting."""

    model_config = ConfigDict(extra="forbid")

    doc_name: str
    domain: str
    question_count: int
    answer_evaluator_summaries: list[EvaluatorSummaryRow]
    initial_recall_at_k_mean: float
    rerank_ndcg_at_k_mean: float
    rerank_recall_at_k_mean: float
    hit_rate_at_k_mean: float
    retrieval_tool_call_count_mean: float
    retrieval_tool_call_count_total: int


class OverallMetricsRow(BaseModel):
    """Represents one corpus-level aggregate row for reporting."""

    model_config = ConfigDict(extra="forbid")

    total_documents: int
    total_questions: int
    answer_evaluator_summaries: list[EvaluatorSummaryRow]
    initial_recall_at_k_mean: float
    rerank_ndcg_at_k_mean: float
    rerank_recall_at_k_mean: float
    hit_rate_at_k_mean: float
    retrieval_tool_call_count_mean: float
    retrieval_tool_call_count_total: int
    zero_hit_question_count: int
    failed_run_count: int


class EvaluationSummary(BaseModel):
    """Represents persisted aggregate outputs for one evaluation run."""

    model_config = ConfigDict(extra="forbid")

    document_metrics: list[DocumentMetricsRow]
    overall_metrics: OverallMetricsRow
