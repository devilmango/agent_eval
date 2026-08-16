"""Deterministic scenario scoring and aggregate metric calculation."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean

from agent_eval.graders import AnswerGrader, SemanticJudge, grade_answer, load_grader_registry, load_judge_registry
from agent_eval.models import AgentResult, EvaluationReport, GradeResult, Scenario, ScenarioScore, ToolCallExpectation
from agent_eval.results import ResultError


def _normalized(values: Iterable[str]) -> set[str]:
    return {value.strip().casefold() for value in values}


def _lookup(arguments: dict, path: str) -> tuple[bool, object]:
    value: object = arguments
    for segment in path.split("."):
        if not isinstance(value, dict) or segment not in value:
            return False, None
        value = value[segment]
    return True, value


def _matches(actual_arguments: dict, path: str, matcher: object) -> bool:
    found, actual = _lookup(actual_arguments, path)
    if isinstance(matcher, dict):
        operator, expected = next(iter(matcher.items()))
        if operator == "exists":
            return found is expected
        if not found:
            return False
        if operator == "equals":
            return actual == expected
        if operator == "not_equals":
            return actual != expected
        if operator == "one_of":
            return actual in expected
        if operator == "contains":
            try:
                return expected in actual
            except TypeError:
                return False
    return found and actual == matcher


def _check_tool_calls(
    expected: tuple[ToolCallExpectation, ...], actual: tuple
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if len(expected) != len(actual):
        mismatches.append(f"expected {len(expected)} tool call(s), got {len(actual)}")
    for index, expected_call in enumerate(expected):
        if index >= len(actual):
            break
        actual_call = actual[index]
        if expected_call.name.strip().casefold() != actual_call.name.strip().casefold():
            mismatches.append(
                f"tool call {index + 1}: expected {expected_call.name!r}, got {actual_call.name!r}"
            )
            continue
        for argument, matcher in expected_call.arguments.items():
            if not _matches(actual_call.arguments, argument, matcher):
                mismatches.append(
                    f"tool call {index + 1} ({actual_call.name}) argument {argument!r} did not match"
                )
    return tuple(mismatches)


def _missing_policy_prerequisites(result: AgentResult, action: str, required: tuple[str, ...]) -> list[str]:
    if not required:
        return []
    action_key = action.strip().casefold()
    action_positions = [
        index
        for index, event in enumerate(result.events)
        if event.event_type == "action" and event.name.strip().casefold() == action_key
    ]
    if not action_positions:
        # A conditional policy requires an ordered event trace; final question
        # and action sets cannot prove that a prerequisite came first.
        return list(required)
    action_position = action_positions[0]
    missing: list[str] = []
    for question in required:
        question_key = question.strip().casefold()
        if not any(
            event.event_type == "question"
            and event.name.strip().casefold() == question_key
            and index < action_position
            for index, event in enumerate(result.events)
        ):
            missing.append(question)
    return missing


def _check_turn_expectations(scenario: Scenario, result: AgentResult) -> tuple[str, ...]:
    if not scenario.turns:
        return ()
    violations: list[str] = []
    if len(result.turns) != len(scenario.turns):
        return (f"expected {len(scenario.turns)} turn result(s), got {len(result.turns)}",)
    for index, (expected, actual) in enumerate(zip(scenario.turns, result.turns), start=1):
        prefix = f"turn {index}: "
        if expected.intent and expected.intent.strip().casefold() != actual.intent.strip().casefold():
            violations.append(prefix + f"intent {actual.intent!r}, expected {expected.intent!r}")
        asked = _normalized(actual.asked)
        required = _normalized(expected.must_ask)
        missing = [name for name in expected.must_ask if name.strip().casefold() not in asked]
        unnecessary = [name for name in actual.asked
                       if name.strip().casefold() not in required
                       and name.strip().casefold() not in _normalized(expected.allowed_questions)]
        if missing:
            violations.append(prefix + "missing questions: " + ", ".join(missing))
        if unnecessary:
            violations.append(prefix + "unnecessary questions: " + ", ".join(unnecessary))
        if expected.expected_tools is not None and _normalized(expected.expected_tools) != _normalized(actual.tools):
            violations.append(prefix + "tool set differs from expected")
        if expected.expected_tool_calls is not None:
            mismatches = _check_tool_calls(expected.expected_tool_calls, actual.tool_calls)
            violations.extend(prefix + mismatch for mismatch in mismatches)
        turn_result = AgentResult(
            scenario=result.scenario, intent=actual.intent, asked=actual.asked,
            tools=actual.tools, actions=actual.actions, tool_calls=actual.tool_calls,
            events=actual.events, answer=actual.answer,
        )
        actions = _normalized(actual.actions)
        for rule in expected.must_not:
            missing_prerequisites = _missing_policy_prerequisites(turn_result, rule.action, rule.unless_asked)
            if rule.action.strip().casefold() in actions and (not rule.unless_asked or missing_prerequisites):
                violations.append(prefix + f"prohibited action: {rule.action}")
    return tuple(violations)


def score_scenario(
    scenario: Scenario,
    result: AgentResult,
    grader_registry: dict[str, AnswerGrader] | None = None,
    judge_registry: dict[str, SemanticJudge] | None = None,
) -> ScenarioScore:
    intent_correct = scenario.intent.strip().casefold() == result.intent.strip().casefold()
    asked = _normalized(result.asked)
    turn_required = tuple(question for turn in scenario.turns for question in turn.must_ask)
    turn_allowed = tuple(question for turn in scenario.turns for question in turn.allowed_questions)
    required = _normalized((*scenario.must_ask, *turn_required))
    allowed = _normalized((*scenario.allowed_questions, *turn_allowed))
    missing = tuple(
        question
        for question in scenario.must_ask
        if question.strip().casefold() not in asked
    )
    unnecessary = tuple(
        question
        for question in result.asked
        if question.strip().casefold() not in required
        and question.strip().casefold() not in allowed
    )

    tool_correct = None
    tool_mismatches: tuple[str, ...] = ()
    if scenario.expected_tools is not None or scenario.expected_tool_calls is not None:
        mismatches: list[str] = []
        if scenario.expected_tools is not None and _normalized(scenario.expected_tools) != _normalized(result.tools):
            mismatches.append("tool set differs from expected")
        if scenario.expected_tool_calls is not None:
            mismatches.extend(_check_tool_calls(scenario.expected_tool_calls, result.tool_calls))
        tool_mismatches = tuple(mismatches)
        tool_correct = not mismatches

    actual_actions = _normalized(result.actions)
    violations: list[str] = []
    for rule in scenario.must_not:
        missing_prerequisites = _missing_policy_prerequisites(
            result, rule.action, rule.unless_asked
        )
        if rule.action.strip().casefold() in actual_actions and (
            not rule.unless_asked or missing_prerequisites
        ):
            suffix = f" (unless asked: {', '.join(missing_prerequisites)})" if missing_prerequisites else ""
            violations.append(rule.action + suffix)

    answer_grades = grade_answer(
        scenario,
        result,
        grader_registry if grader_registry is not None else load_grader_registry(),
        judge_registry,
    ) if scenario.answer_graders else ()
    turn_violations = _check_turn_expectations(scenario, result)

    passed = (
        intent_correct
        and not missing
        and not violations
        and not unnecessary
        and tool_correct is not False
        and all(grade.passed for grade in answer_grades)
        and not turn_violations
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
        tool_call_mismatches=tool_mismatches,
        tool_correct=tool_correct,
        policy_violations=tuple(violations),
        answer_grades=answer_grades,
        turn_violations=turn_violations,
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
    if missing_results and strict:
        raise ResultError(f"Results are missing scenario(s): {', '.join(missing_results)}")

    has_graders = any(scenario.answer_graders for scenario in scenarios)
    grader_registry = load_grader_registry() if has_graders else {}
    judge_registry = (
        load_judge_registry()
        if any(spec.get("type") == "llm_judge" for scenario in scenarios for spec in scenario.answer_graders)
        else {}
    )

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
                    tool_correct=(
                        False
                        if scenario.expected_tools is not None or scenario.expected_tool_calls is not None
                        else None
                    ),
                    tool_call_mismatches=("No agent result was provided.",)
                    if scenario.expected_tool_calls is not None
                    else (),
                    notes="No agent result was provided.",
                    answer_grades=tuple(
                        GradeResult(
                            grader=spec["type"],
                            score=0.0,
                            passed=False,
                            feedback="No agent result was provided.",
                            rubric=dict(spec),
                        )
                        for spec in scenario.answer_graders
                    ),
                )
            )
        else:
            scores.append(score_scenario(scenario, result, grader_registry, judge_registry))

    total = len(scores)
    passed = sum(score.passed for score in scores)
    intent_correct = sum(score.intent_correct for score in scores)
    tool_scores = [score.tool_correct for score in scores if score.tool_correct is not None]
    policy_checks = sum(len(scenario.must_not) for scenario in scenarios)
    violations = sum(len(score.policy_violations) for score in scores)
    asked_count = sum(len(result.asked) for result in results.values())
    unnecessary_count = sum(len(score.unnecessary_questions) for score in scores)
    answer_scores = [grade.score for score in scores for grade in score.answer_grades]

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
        answer_quality_score=mean(answer_scores) if answer_scores else None,
        scores=tuple(scores),
    )
