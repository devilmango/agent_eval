"""Deterministic scenario scoring and aggregate metric calculation."""

from __future__ import annotations

from collections.abc import Iterable

from agent_eval.models import AgentResult, EvaluationReport, Scenario, ScenarioScore
from agent_eval.results import ResultError


def _normalized(values: Iterable[str]) -> set[str]:
    return {value.strip().casefold() for value in values}


def score_scenario(scenario: Scenario, result: AgentResult) -> ScenarioScore:
    intent_correct = scenario.intent.strip().casefold() == result.intent.strip().casefold()
    asked = _normalized(result.asked)
    required = _normalized(scenario.must_ask)
    allowed = _normalized(scenario.allowed_questions)
    missing = tuple(question for question in scenario.must_ask if question.strip().casefold() not in asked)
    unnecessary = tuple(
        question
        for question in result.asked
        if question.strip().casefold() not in required
        and question.strip().casefold() not in allowed
    )

    tool_correct = None
    if scenario.expected_tools is not None:
        tool_correct = _normalized(scenario.expected_tools) == _normalized(result.tools)

    actual_actions = _normalized(result.actions)
    violations = tuple(
        prohibited
        for prohibited in scenario.must_not
        if prohibited.strip().casefold() in actual_actions
    )
    passed = (
        intent_correct
        and not missing
        and not violations
        and not unnecessary
        and tool_correct is not False
    )
    return ScenarioScore(
        scenario=scenario.name,
        passed=passed,
        expected_intent=scenario.intent,
        actual_intent=result.intent,
        intent_correct=intent_correct,
        missing_questions=missing,
        unnecessary_questions=unnecessary,
        expected_tools=scenario.expected_tools,
        actual_tools=result.tools,
        tool_correct=tool_correct,
        policy_violations=violations,
        notes=result.notes,
    )


def evaluate(
    scenarios: list[Scenario], results: dict[str, AgentResult], *, strict: bool = False
) -> EvaluationReport:
    expected_names = {scenario.name for scenario in scenarios}
    unknown = sorted(results.keys() - expected_names)
    missing_results = sorted(expected_names - results.keys())
    if unknown:
        raise ResultError(f"Results contain unknown scenario(s): {', '.join(unknown)}")
    if missing_results:
        message = f"Results are missing scenario(s): {', '.join(missing_results)}"
        if strict:
            raise ResultError(message)

    scores: list[ScenarioScore] = []
    for scenario in scenarios:
        result = results.get(scenario.name)
        if result is None:
            scores.append(
                ScenarioScore(
                    scenario=scenario.name,
                    passed=False,
                    expected_intent=scenario.intent,
                    actual_intent="(missing result)",
                    intent_correct=False,
                    missing_questions=scenario.must_ask,
                    expected_tools=scenario.expected_tools,
                    tool_correct=False if scenario.expected_tools is not None else None,
                    notes="No agent result was provided.",
                )
            )
        else:
            scores.append(score_scenario(scenario, result))

    total = len(scores)
    passed = sum(score.passed for score in scores)
    intent_correct = sum(score.intent_correct for score in scores)
    tool_scores = [score.tool_correct for score in scores if score.tool_correct is not None]
    policy_checks = sum(len(scenario.must_not) for scenario in scenarios)
    violations = sum(len(score.policy_violations) for score in scores)
    asked_count = sum(len(result.asked) for result in results.values())
    unnecessary_count = sum(len(score.unnecessary_questions) for score in scores)

    return EvaluationReport(
        total=total,
        passed=passed,
        failed=total - passed,
        intent_accuracy=intent_correct / total if total else 0.0,
        tool_accuracy=(sum(tool_scores) / len(tool_scores)) if tool_scores else None,
        policy_compliance=(
            (policy_checks - violations) / policy_checks if policy_checks else 1.0
        ),
        unnecessary_question_rate=(
            unnecessary_count / asked_count if asked_count else 0.0
        ),
        scores=tuple(scores),
    )

