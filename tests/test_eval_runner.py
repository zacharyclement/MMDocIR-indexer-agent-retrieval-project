"""Tests for evaluation dataset loading and flattening."""

from __future__ import annotations

import json
from pathlib import Path

from eval.runner import (
    build_failed_question_result,
    flatten_documents,
    load_raw_documents,
)


def test_load_raw_documents_and_flatten_documents(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_name": "doc-a.pdf",
                        "domain": "Guidebook",
                        "page_indices": [],
                        "layout_indices": [],
                        "questions": [
                            {
                                "Q": "What is A?",
                                "A": "Alpha",
                                "page_id": [1],
                                "type": '["Table"]',
                                "layout_mapping": [],
                            },
                            {
                                "Q": "What is B?",
                                "A": "Beta",
                                "page_id": [2, 3],
                                "type": ["Chart"],
                                "layout_mapping": [],
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "doc_name": "doc-b.pdf",
                        "domain": "Academic paper",
                        "page_indices": [],
                        "layout_indices": [],
                        "questions": [
                            {
                                "Q": "What is C?",
                                "A": "Gamma",
                                "page_id": [4],
                                "type": ["Pure-text (Plain-text)"],
                                "layout_mapping": [],
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    raw_documents = load_raw_documents(dataset_path)
    flattened_questions = flatten_documents(raw_documents)

    assert len(raw_documents) == 2
    assert len(flattened_questions) == 3
    assert flattened_questions[0].doc_name == "doc-a.pdf"
    assert flattened_questions[0].question_index == 0
    assert flattened_questions[0].question_type == ("Table",)
    assert flattened_questions[1].expected_pages == [2, 3]
    assert flattened_questions[2].doc_name == "doc-b.pdf"


def test_build_failed_question_result_sets_zeroed_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "doc_name": "doc-a.pdf",
                "domain": "Guidebook",
                "page_indices": [],
                "layout_indices": [],
                "questions": [
                    {
                        "Q": "What is A?",
                        "A": "Alpha",
                        "page_id": [1],
                        "type": ["Table"],
                        "layout_mapping": [],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    question_record = flatten_documents(load_raw_documents(dataset_path))[0]
    failed_result = build_failed_question_result(
        question_record=question_record,
        model_name="anthropic:claude-sonnet-4-6",
        error_message="boom",
    )

    assert failed_result.model_answer == ""
    assert failed_result.retrieval_tool_call_count == 0
    assert failed_result.hit_rate_at_k == 0.0
    assert failed_result.answer_evaluation.evaluator_error == "boom"
