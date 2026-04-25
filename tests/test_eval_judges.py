"""Tests for OpenEvals answer judge helpers."""

from __future__ import annotations

from eval.judges import (
    JudgeInput,
    _normalize_comment,
    _normalize_score,
    _run_answer_evaluator,
)


class _FakeEvaluator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "key": "groundedness",
            "score": 0.75,
            "comment": "  supported by context  ",
        }


def test_run_answer_evaluator_passes_expected_prompt_variables() -> None:
    evaluator = _FakeEvaluator()

    feedback = _run_answer_evaluator(
        evaluator_key="groundedness",
        evaluator=evaluator,
        judge_input=JudgeInput(
            question="What happened?",
            expected_answer="The expected answer.",
            model_answer="The model answer.",
            contexts=["Context one.", "Context two."],
        ),
    )

    assert evaluator.calls == [
        {
            "inputs": "What happened?",
            "outputs": "The model answer.",
            "context": "Context one.\n\nContext two.",
        }
    ]
    assert feedback.evaluator_key == "groundedness"
    assert feedback.score == 0.75
    assert feedback.comment == "supported by context"
    assert feedback.raw_result is not None


def test_run_answer_evaluator_passes_reference_output_for_correctness() -> None:
    evaluator = _FakeEvaluator()

    _run_answer_evaluator(
        evaluator_key="correctness",
        evaluator=evaluator,
        judge_input=JudgeInput(
            question="What happened?",
            expected_answer="The expected answer.",
            model_answer="The model answer.",
            contexts=[],
        ),
    )

    assert evaluator.calls[0]["reference_outputs"] == "The expected answer."


def test_normalize_score_and_comment_handle_supported_types() -> None:
    assert _normalize_score(True) is True
    assert _normalize_score(0.5) == 0.5
    assert _normalize_score("bad") is None
    assert _normalize_comment("  note  ") == "note"
    assert _normalize_comment(None) is None
