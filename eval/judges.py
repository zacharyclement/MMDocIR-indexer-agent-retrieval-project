"""Judge helpers for answer and context evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.agent.llms import DEFAULT_MODEL_NAME, normalize_model_name
from eval.schemas import AnswerEvaluationRecord, EvaluatorFeedbackRecord
from indexer.shared.errors import IndexingRuntimeError

DEFAULT_JUDGE_MODEL_NAME = DEFAULT_MODEL_NAME
DEFAULT_EVALUATOR_SPECS = (
    ("correctness", "CORRECTNESS_PROMPT"),
    ("helpfulness", "RAG_HELPFULNESS_PROMPT"),
    ("groundedness", "RAG_GROUNDEDNESS_PROMPT"),
)


@dataclass(frozen=True)
class JudgeInput:
    """Represents one judge invocation payload."""

    question: str
    expected_answer: str
    model_answer: str
    contexts: list[str]


class AnswerJudge:
    """Evaluates answer quality with prebuilt OpenEvals evaluators."""

    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL_NAME) -> None:
        self._model_name = normalize_model_name(model_name)
        try:
            self._evaluators = _build_answer_evaluators(self._model_name)
        except ImportError as error:
            raise IndexingRuntimeError(
                "OpenEvals scoring requires the optional evaluation dependencies to be "
                "installed in the active environment."
            ) from error

    @property
    def model_name(self) -> str:
        """Return the normalized judge model name."""

        return self._model_name

    def score(self, judge_input: JudgeInput) -> AnswerEvaluationRecord:
        """Score one evaluated question."""

        try:
            return AnswerEvaluationRecord(
                feedback=[
                    _run_answer_evaluator(
                        evaluator_key=evaluator_key,
                        evaluator=evaluator,
                        judge_input=judge_input,
                    )
                    for evaluator_key, evaluator in self._evaluators.items()
                ]
            )
        except ImportError as error:
            raise IndexingRuntimeError(
                "OpenEvals scoring requires the optional evaluation dependencies to be "
                "installed in the active environment."
            ) from error
        except Exception as error:
            raise IndexingRuntimeError(
                "Failed to score evaluation example with OpenEvals using judge model "
                f"'{self._model_name}': {error}"
            ) from error


def _build_answer_evaluators(model_name: str) -> dict[str, object]:
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import (
        CORRECTNESS_PROMPT,
        RAG_GROUNDEDNESS_PROMPT,
        RAG_HELPFULNESS_PROMPT,
    )

    prompt_by_name = {
        "CORRECTNESS_PROMPT": CORRECTNESS_PROMPT,
        "RAG_HELPFULNESS_PROMPT": RAG_HELPFULNESS_PROMPT,
        "RAG_GROUNDEDNESS_PROMPT": RAG_GROUNDEDNESS_PROMPT,
    }
    evaluators: dict[str, object] = {}
    for evaluator_key, prompt_name in DEFAULT_EVALUATOR_SPECS:
        evaluators[evaluator_key] = create_llm_as_judge(
            prompt=prompt_by_name[prompt_name],
            feedback_key=evaluator_key,
            model=model_name,
        )
    return evaluators


def _run_answer_evaluator(
    evaluator_key: str,
    evaluator: object,
    judge_input: JudgeInput,
) -> EvaluatorFeedbackRecord:
    if not callable(evaluator):
        raise IndexingRuntimeError(
            f"Evaluator '{evaluator_key}' is not callable."
        )

    evaluator_kwargs = {
        "inputs": judge_input.question,
        "outputs": judge_input.model_answer,
    }
    if evaluator_key == "correctness":
        evaluator_kwargs["reference_outputs"] = judge_input.expected_answer
    if evaluator_key == "groundedness":
        evaluator_kwargs["context"] = "\n\n".join(judge_input.contexts)

    result = evaluator(**evaluator_kwargs)
    if not isinstance(result, dict):
        raise IndexingRuntimeError(
            f"Evaluator '{evaluator_key}' returned an unexpected result: {result!r}"
        )
    return EvaluatorFeedbackRecord(
        evaluator_key=str(result.get("key", evaluator_key)),
        source="openevals",
        score=_normalize_score(result.get("score")),
        comment=_normalize_comment(result.get("comment")),
        raw_result=json.dumps(result, default=str, sort_keys=True),
    )


def _normalize_score(value: object) -> bool | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_comment(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped_value = value.strip()
        return stripped_value or None
    return str(value)
