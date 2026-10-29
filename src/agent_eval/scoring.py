"""Deterministic scenario scoring and aggregate metric calculation."""

from __future__ import annotations

from collections.abc import Iterable
import re
from statistics import mean
from pathlib import Path

from agent_eval.graders import (AnswerGrader, SemanticJudge, grade_answer, load_grader_registry,
                                load_judge_registry, public_grader_rubric)
from agent_eval.models import (AgentResult, AgentTurnResult, EvaluationReport, FailureDiagnostic, GradeResult,
                               Scenario, ScenarioScore, ScenarioTurn, ToolCallExpectation)
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


def _scenario_diagnostics(
    scenario: Scenario, result: AgentResult, *, intent_correct: bool,
    missing_questions: tuple[str, ...], unnecessary_questions: tuple[str, ...],
    policy_violations: tuple[str, ...],
    answer_grades: tuple[GradeResult, ...],
) -> tuple[FailureDiagnostic, ...]:
    diagnostics: list[FailureDiagnostic] = []
    observed_turns = result.turns
    if scenario.turns:
        for index, expected in enumerate(scenario.turns, start=1):
            actual = observed_turns[index - 1] if index <= len(observed_turns) else None
            if actual is None:
                diagnostics.append(FailureDiagnostic(
                    "turn", f"expected turn {index} was not returned", turn_index=index,
                    expected=expected.input, actual=None,
                ))
                continue
            if expected.intent and expected.intent.casefold() != actual.intent.casefold():
                diagnostics.append(FailureDiagnostic(
                    "intent", f"turn {index} intent differed", turn_index=index,
                    expected=expected.intent, actual=actual.intent,
                ))
            actual_questions = _normalized(actual.asked)
            for question in expected.must_ask:
                if question.casefold() not in actual_questions:
                    diagnostics.append(FailureDiagnostic(
                        "question", f"turn {index} omitted required question {question!r}",
                        turn_index=index, expected=question, actual=list(actual.asked),
                    ))
            unexpected = [question for question in actual.asked
                          if question.casefold() not in _normalized(expected.must_ask)
                          and question.casefold() not in _normalized(expected.allowed_questions)]
            for question in unexpected:
                diagnostics.append(FailureDiagnostic(
                    "question", f"turn {index} asked unexpected question {question!r}",
                    turn_index=index, expected=list(expected.must_ask), actual=question,
                ))
            if expected.expected_tools is not None and _normalized(expected.expected_tools) != _normalized(actual.tools):
                diagnostics.append(FailureDiagnostic(
                    "tool", f"turn {index} tool set differed", turn_index=index,
                    expected=list(expected.expected_tools), actual=list(actual.tools),
                ))
            diagnostics.extend(_tool_call_diagnostics(
                expected.expected_tool_calls, actual.tool_calls, turn_index=index,
            ))
            turn_result = AgentResult(
                scenario=scenario.name, intent=actual.intent, actions=actual.actions,
                events=actual.events,
            )
            for rule in expected.must_not:
                missing = _missing_policy_prerequisites(turn_result, rule.action, rule.unless_asked)
                if rule.action.casefold() in _normalized(actual.actions) and (
                    not rule.unless_asked or missing
                ):
                    event_index = next((i + 1 for i, event in enumerate(actual.events)
                                        if event.event_type == "action"
                                        and event.name.casefold() == rule.action.casefold()), None)
                    diagnostics.append(FailureDiagnostic(
                        "policy", f"turn {index} performed prohibited action {rule.action!r}",
                        turn_index=index, event_index=event_index,
                        expected=f"confirmation before {rule.action}" if rule.unless_asked else "no action",
                        actual=rule.action,
                    ))
        if len(observed_turns) > len(scenario.turns):
            diagnostics.append(FailureDiagnostic(
                "turn", f"received {len(observed_turns) - len(scenario.turns)} unexpected extra turn(s)",
                turn_index=len(scenario.turns) + 1,
                expected=len(scenario.turns), actual=len(observed_turns),
            ))

    result_turn = len(observed_turns) if observed_turns else None
    if not intent_correct:
        diagnostics.append(FailureDiagnostic(
            "intent", "final intent differed", turn_index=result_turn,
            expected=scenario.intent, actual=result.intent,
        ))
    for question in missing_questions:
        diagnostics.append(FailureDiagnostic(
            "question", f"required question {question!r} was not asked", turn_index=result_turn,
            expected=question, actual=list(result.asked),
        ))
    for question in unnecessary_questions:
        diagnostics.append(FailureDiagnostic(
            "question", f"unexpected question {question!r} was asked", turn_index=result_turn,
            expected=list((*scenario.must_ask, *scenario.allowed_questions)), actual=question,
        ))
    if scenario.expected_tools is not None and _normalized(scenario.expected_tools) != _normalized(result.tools):
        diagnostics.append(FailureDiagnostic(
            "tool", "tool set differed", turn_index=result_turn,
            expected=list(scenario.expected_tools), actual=list(result.tools),
        ))
    diagnostics.extend(_tool_call_diagnostics(
        scenario.expected_tool_calls, result.tool_calls, turn_index=result_turn,
    ))
    for violation in policy_violations:
        action = violation.split(" (unless asked:", 1)[0]
        event_index = next((i + 1 for i, event in enumerate(result.events)
                            if event.event_type == "action" and event.name.casefold() == action.casefold()), None)
        diagnostics.append(FailureDiagnostic(
            "policy", f"prohibited action {action!r} was performed", turn_index=result_turn,
            event_index=event_index, expected="prerequisite before action" if "unless asked" in violation else "no action",
            actual=action,
        ))
    for grade in answer_grades:
        if grade.passed or grade.metadata.get("skipped") is True:
            continue
        turn_index = result_turn
        if grade.grader == "evidence_entailment":
            claims = grade.metadata.get("claims", [])
            for index, claim in enumerate(claims, start=1):
                if claim.get("passed"):
                    continue
                citations = tuple(claim.get("citations", []))
                diagnostics.append(FailureDiagnostic(
                    "evidence", f"claim {index} was not entailed: {claim.get('feedback', '')}",
                    turn_index=turn_index, citation_ids=citations,
                    evidence_sources=citations, expected="claim supported by cited evidence",
                    actual=claim.get("claim"),
                ))
            for error in grade.metadata.get("citation_errors", []):
                citations = tuple(re.findall(r"source ['\"]([^'\"]+)", error))
                diagnostics.append(FailureDiagnostic(
                    "evidence", error, turn_index=turn_index, citation_ids=citations,
                    evidence_sources=citations,
                ))
        else:
            citations = tuple(grade.metadata.get("citations", []))
            known = _evidence_source_ids(grade.rubric)
            diagnostics.append(FailureDiagnostic(
                "evidence" if grade.grader == "evidence_grounded" else "answer_grader",
                f"{grade.grader} grader failed: {grade.feedback or 'failed'}",
                turn_index=turn_index, citation_ids=citations,
                evidence_sources=tuple(item for item in citations if item in known),
                expected="answer satisfies grader", actual=grade.feedback,
            ))
    return tuple(diagnostics)


def _tool_call_diagnostics(expected: tuple[ToolCallExpectation, ...] | None,
                           actual: tuple, *, turn_index: int | None) -> list[FailureDiagnostic]:
    if expected is None:
        return []
    diagnostics: list[FailureDiagnostic] = []
    for index in range(max(len(expected), len(actual))):
        wanted = expected[index] if index < len(expected) else None
        observed = actual[index] if index < len(actual) else None
        expected_value = None if wanted is None else {"name": wanted.name, "arguments": wanted.arguments}
        actual_value = None if observed is None else {"name": observed.name, "arguments": observed.arguments}
        mismatch = (wanted is None or observed is None
                    or wanted.name.casefold() != observed.name.casefold()
                    or any(not _matches(observed.arguments, path, matcher)
                           for path, matcher in wanted.arguments.items()))
        if mismatch:
            diagnostics.append(FailureDiagnostic(
                "tool_call", f"tool call {index + 1} differed from expectation",
                turn_index=turn_index, tool_call_index=index + 1,
                expected=expected_value, actual=actual_value,
            ))
    return diagnostics


def _evidence_source_ids(rubric: dict) -> set[str]:
    evidence = rubric.get("evidence", [])
    if isinstance(evidence, dict):
        return {str(item) for item in evidence}
    return {item["id"] for item in evidence if isinstance(item, dict) and isinstance(item.get("id"), str)}


def _trace_timeline(scenario: Scenario, result: AgentResult) -> tuple[dict, ...]:
    """Build a stable per-turn sequence from the ordered trace fields we capture."""
    timeline: list[dict] = []
    observed_turns = result.turns or (AgentTurnResult(
        intent=result.intent, asked=result.asked, tools=result.tools, actions=result.actions,
        tool_calls=result.tool_calls, events=result.events, answer=result.answer,
        notes=result.notes, metadata=result.metadata,
    ),)
    prompts = scenario.turns or (ScenarioTurn(input=scenario.input),)
    for index, prompt in enumerate(prompts, start=1):
        actual = observed_turns[index - 1] if index <= len(observed_turns) else None

        def add(kind: str, expected=None, observed=None, matched=None):
            timeline.append({
                "sequence": len(timeline) + 1,
                "turn_index": index,
                "kind": kind,
                "expected": expected,
                "observed": observed,
                "matched": matched,
                "first_divergence": False,
            })

        add("user_input", None, prompt.input, None)
        expected_intent = (prompt.intent if scenario.turns else
                           scenario.intent if index == len(prompts) else None)
        if expected_intent is None and scenario.turns and index == len(prompts):
            expected_intent = scenario.intent
        add("intent", expected_intent, actual.intent if actual else None,
            expected_intent.strip().casefold() == actual.intent.strip().casefold()
            if expected_intent is not None and actual else False if expected_intent is not None else None)

        required = prompt.must_ask if scenario.turns else scenario.must_ask
        allowed = set(prompt.allowed_questions if scenario.turns else scenario.allowed_questions)
        asked = actual.asked if actual else ()
        for question in required:
            observed = next((item for item in asked if item.casefold() == question.casefold()), None)
            add("question", question, observed, observed is not None)
        required_folded = {item.casefold() for item in required}
        for question in asked:
            expected = question if question.casefold() in required_folded else None
            matched = True if expected is not None or question in allowed else False
            if expected is None:
                add("question", expected, question, matched)

        expected_tools = prompt.expected_tools
        if expected_tools is None and (not scenario.turns or index == len(prompts)):
            expected_tools = scenario.expected_tools
        if expected_tools is not None:
            for tool in expected_tools:
                observed = next((item for item in actual.tools if item.casefold() == tool.casefold()), None) if actual else None
                add("tool_selection", tool, observed, observed is not None)
            expected_folded = {item.casefold() for item in expected_tools}
            if actual:
                for tool in actual.tools:
                    if tool.casefold() not in expected_folded:
                        add("tool_selection", None, tool, False)

        expected_calls = prompt.expected_tool_calls
        if expected_calls is None and (not scenario.turns or index == len(prompts)):
            expected_calls = scenario.expected_tool_calls
        observed_calls = actual.tool_calls if actual else ()
        if expected_calls is not None:
            for call_index in range(max(len(expected_calls), len(observed_calls))):
                expected_call = expected_calls[call_index] if call_index < len(expected_calls) else None
                observed_call = observed_calls[call_index] if call_index < len(observed_calls) else None
                expected_value = ({"name": expected_call.name, "arguments": expected_call.arguments}
                                  if expected_call else None)
                observed_value = ({"name": observed_call.name, "arguments": observed_call.arguments}
                                  if observed_call else None)
                matched = bool(expected_call and observed_call
                               and expected_call.name.casefold() == observed_call.name.casefold()
                               and all(_matches(observed_call.arguments, key, value)
                                       for key, value in expected_call.arguments.items()))
                add("tool_call", expected_value, observed_value, matched)
        else:
            for observed_call in observed_calls:
                add("tool_call", None, {"name": observed_call.name,
                                         "arguments": observed_call.arguments}, None)

        rules = list(scenario.must_not)
        if scenario.turns:
            rules.extend(prompt.must_not)
        forbidden = {rule.action.casefold() for rule in rules}
        turn_result = AgentResult(
            scenario=scenario.name, intent=actual.intent if actual else "",
            actions=actual.actions if actual else (), events=actual.events if actual else (),
        )

        def violates_policy(action: str) -> bool:
            return any(
                rule.action.casefold() == action.casefold()
                and (not rule.unless_asked or _missing_policy_prerequisites(
                    turn_result, rule.action, rule.unless_asked,
                ))
                for rule in rules
            )

        event_rows = actual.events if actual else ()
        if event_rows:
            for event in event_rows:
                if event.event_type == "action":
                    add("policy_event", {"must_not": sorted(forbidden)}, event.name,
                        not violates_policy(event.name))
                else:
                    add("trace_event", None, {"type": event.event_type, "name": event.name}, None)
        elif actual:
            for action in actual.actions:
                add("policy_event", {"must_not": sorted(forbidden)}, action,
                    not violates_policy(action))

        answer = actual.answer if actual else None
        add("answer", None, answer, None)
        if answer:
            evidence_ids = {
                source
                for grader in scenario.answer_graders
                for source in _evidence_source_ids(grader)
            }
            for citation in re.findall(r"\[([A-Za-z0-9_.:-]+)\]", answer):
                add("citation", sorted(evidence_ids) if evidence_ids else None, citation,
                    citation in evidence_ids if evidence_ids else None)

    first = next((item for item in timeline if item["matched"] is False), None)
    if first is not None:
        first["first_divergence"] = True
    return tuple(timeline)


def score_scenario(
    scenario: Scenario,
    result: AgentResult,
    grader_registry: dict[str, AnswerGrader] | None = None,
    judge_registry: dict[str, SemanticJudge] | None = None,
    *, judge_cache: Path | None = None, judge_repetitions: int = 1,
    skip_semantic_judges: bool = False,
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
        judge_cache=judge_cache,
        judge_repetitions=judge_repetitions,
        skip_semantic_judges=skip_semantic_judges,
    ) if scenario.answer_graders else ()
    turn_violations = _check_turn_expectations(scenario, result)

    passed = (
        intent_correct
        and not missing
        and not violations
        and not unnecessary
        and tool_correct is not False
        and all(grade.passed or grade.metadata.get("skipped") is True for grade in answer_grades)
        and not turn_violations
    )
    diagnostics = _scenario_diagnostics(
        scenario, result, intent_correct=intent_correct, missing_questions=missing,
        unnecessary_questions=unnecessary, policy_violations=tuple(violations),
        answer_grades=answer_grades,
    )
    if not passed and not diagnostics:
        diagnostics = (FailureDiagnostic("scenario", "scenario did not pass its declared expectations"),)
    timeline = _trace_timeline(scenario, result)
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
        diagnostics=diagnostics,
        trace_timeline=timeline,
    )


def evaluate(
    scenarios: list[Scenario], results: dict[str, AgentResult], *, strict: bool = False,
    judge_cache: Path | None = None, judge_repetitions: int = 1,
    skip_semantic_judges: bool = False,
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
        if not skip_semantic_judges
        and any(spec.get("type") in {"llm_judge", "evidence_entailment"}
                for scenario in scenarios for spec in scenario.answer_graders)
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
                    diagnostics=(FailureDiagnostic("result", "No agent result was provided."),),
                    answer_grades=tuple(
                        GradeResult(
                            grader=spec["type"],
                            score=0.0,
                            passed=False,
                            feedback="No agent result was provided.",
                            rubric=public_grader_rubric(spec),
                        )
                        for spec in scenario.answer_graders
                    ),
                )
            )
        else:
            scores.append(score_scenario(scenario, result, grader_registry, judge_registry,
                                         judge_cache=judge_cache, judge_repetitions=judge_repetitions,
                                         skip_semantic_judges=skip_semantic_judges))

    total = len(scores)
    passed = sum(score.passed for score in scores)
    intent_correct = sum(score.intent_correct for score in scores)
    tool_scores = [score.tool_correct for score in scores if score.tool_correct is not None]
    policy_checks = sum(len(scenario.must_not) for scenario in scenarios)
    violations = sum(len(score.policy_violations) for score in scores)
    asked_count = sum(len(result.asked) for result in results.values())
    unnecessary_count = sum(len(score.unnecessary_questions) for score in scores)
    answer_scores = [grade.score for score in scores for grade in score.answer_grades
                     if grade.metadata.get("skipped") is not True]

    metadata_rows = [result.metadata for result in results.values() if result.metadata]
    execution: dict = {"runs_with_metadata": len(metadata_rows)}
    for field in ("latency_ms", "input_tokens", "output_tokens", "cost_usd"):
        values = [row[field] for row in metadata_rows if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)]
        if values:
            execution[field] = {"total": sum(values), "mean": mean(values), "min": min(values), "max": max(values)}
    for field in ("model", "prompt_version"):
        values = sorted({str(row[field]) for row in metadata_rows if row.get(field) is not None})
        if values:
            execution[field + "s"] = values
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
        execution=execution,
    )
