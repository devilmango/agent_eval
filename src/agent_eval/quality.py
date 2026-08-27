"""Scenario-set coverage summaries and deterministic trace mutation checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any
from pathlib import Path
import json

from agent_eval.models import AgentResult, Scenario, ToolCall, TraceEvent
from agent_eval.scoring import score_scenario


def scenario_suggestions(scenarios: list[Scenario], *, limit: int = 20) -> dict[str, Any]:
    """Return deterministic, review-only candidates; no new oracle labels are invented."""
    coverage = scenario_coverage(scenarios)
    checks = coverage["checks"]
    gaps = [name for name in ("required_question_scenarios", "tool_scenarios", "policy_scenarios", "answer_graded_scenarios", "multi_turn_scenarios") if checks[name] == 0]
    candidates = []
    for scenario in scenarios:
        user_text = next((value for value in scenario.input.values() if isinstance(value, str)), None)
        if user_text:
            candidates.append({"scenario": scenario.name, "kind": "paraphrase_review", "input_variant": "Please help me with this request: " + user_text,
                              "oracle": "Review expected behavior before adopting; labels are intentionally not inferred."})
        for gap in gaps:
            candidates.append({"scenario": scenario.name, "kind": "coverage_gap", "dimension": gap,
                               "candidate": f"Add a reviewed scenario that asserts {gap.replace('_', ' ')} behavior.",
                               "oracle": "Human-authored expected behavior required."})
    return {"coverage": coverage, "review_required": True, "candidates": candidates[:limit]}


def judge_agreement(paths: list[Path]) -> dict[str, Any]:
    """Compare per-scenario judge outcomes in saved JSON evaluation reports."""
    if len(paths) < 2:
        raise ValueError("judge-agreement requires at least two report files")
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read evaluation report {path}: {exc}") from exc
        rows = data.get("scenarios") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected an Agent Eval JSON report with scenarios")
        for row in rows:
            if not isinstance(row, dict):
                continue
            grades = row.get("answer_grades", [])
            judges = [grade for grade in grades if isinstance(grade, dict) and grade.get("metadata", {}).get("judge")]
            if judges:
                by_scenario.setdefault(str(row.get("scenario")), []).extend(judges)
    outcomes = []
    for scenario, grades in sorted(by_scenario.items()):
        decisions = [bool(grade.get("passed")) for grade in grades]
        scores = [float(grade["score"]) for grade in grades if isinstance(grade.get("score"), (int, float))]
        outcomes.append({"scenario": scenario, "judgments": len(grades),
                         "agreement": max(decisions.count(True), decisions.count(False)) / len(decisions),
                         "score_range": max(scores) - min(scores) if scores else None,
                         "judges": sorted({grade["metadata"].get("judge", "unknown") for grade in grades})})
    decisions = [item["agreement"] for item in outcomes]
    return {"reports": [str(path) for path in paths], "scenario_count": len(outcomes),
            "mean_majority_agreement": sum(decisions) / len(decisions) if decisions else None,
            "scenarios": outcomes}


def failure_analysis(scenarios: list[Scenario], report: Any,
                     results: dict[str, AgentResult]) -> dict[str, Any]:
    """Cluster failures and summarize pass rates by intent, tag, model, and prompt."""
    scenario_by_name = {scenario.name: scenario for scenario in scenarios}
    clusters: dict[str, set[str]] = {}
    slices: dict[str, dict[str, set[str]]] = {}

    def add_slice(dimension: str, value: str, name: str) -> None:
        slices.setdefault(dimension, {}).setdefault(value, set()).add(name)

    for score in report.scores:
        scenario = scenario_by_name[score.scenario]
        result = results.get(score.scenario)
        if not score.passed:
            reasons: list[str] = []
            if result is None:
                reasons.append("missing_result")
            if not score.intent_correct:
                reasons.append("wrong_intent")
            reasons.extend(f"missing_question:{name}" for name in score.missing_questions)
            reasons.extend(f"unnecessary_question:{name}" for name in score.unnecessary_questions)
            if score.tool_correct is False:
                reasons.extend("tool_mismatch:" + detail for detail in (score.tool_call_mismatches or ("tool selection",)))
            reasons.extend("policy_violation:" + detail for detail in score.policy_violations)
            reasons.extend("answer_grader:" + grade.grader for grade in score.answer_grades if not grade.passed)
            reasons.extend("turn_failure:" + detail for detail in score.turn_violations)
            if not reasons:
                reasons.append("other_failure")
            for reason in reasons:
                clusters.setdefault(reason, set()).add(score.scenario)
        add_slice("intent", scenario.intent, scenario.name)
        for tag in scenario.tags:
            add_slice("tag", tag, scenario.name)
        if result:
            for dimension in ("model", "prompt_version"):
                value = result.metadata.get(dimension)
                if value is not None:
                    add_slice(dimension, str(value), scenario.name)

    score_by_name = {score.scenario: score for score in report.scores}
    cluster_rows = [{"reason": reason, "scenario_count": len(names), "scenarios": sorted(names)}
                    for reason, names in sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))]
    slice_rows: dict[str, list[dict[str, Any]]] = {}
    for dimension, groups in sorted(slices.items()):
        rows = []
        for value, names in sorted(groups.items()):
            passed = sum(score_by_name[name].passed for name in names)
            rows.append({"value": value, "scenario_count": len(names), "passed": passed,
                         "failed": len(names) - passed,
                         "pass_rate": passed / len(names) if names else 0.0,
                         "failed_scenarios": sorted(name for name in names if not score_by_name[name].passed)})
        slice_rows[dimension] = rows
    return {"scenario_count": report.total, "failed_scenarios": report.failed,
            "failure_clusters": cluster_rows, "slices": slice_rows}


def text_failure_analysis(analysis: dict[str, Any]) -> str:
    lines = [f"Failure analysis: {analysis['failed_scenarios']}/{analysis['scenario_count']} scenarios failed", "", "Failure clusters:"]
    if analysis["failure_clusters"]:
        for row in analysis["failure_clusters"]:
            lines.append(f"  {row['reason']}: {row['scenario_count']} ({', '.join(row['scenarios'])})")
    else:
        lines.append("  none")
    for dimension, rows in analysis["slices"].items():
        lines.extend(["", dimension.replace("_", " ").capitalize() + " slices:"])
        lines.extend(f"  {row['value']}: {row['passed']}/{row['scenario_count']} passed ({row['pass_rate'] * 100:.1f}%)" for row in rows)
    return "\n".join(lines)


def scenario_coverage(scenarios: list[Scenario]) -> dict[str, Any]:
    """Summarize which behavior dimensions are explicitly represented."""
    intents = Counter(scenario.intent for scenario in scenarios)
    intents.update(turn.intent for scenario in scenarios for turn in scenario.turns if turn.intent)
    tags = Counter(tag for scenario in scenarios for tag in scenario.tags)
    tools = Counter(tool for scenario in scenarios
                    for tool in ((scenario.expected_tools or ()) + tuple(
                        call.name for call in (scenario.expected_tool_calls or ())
                    )))
    tools.update(tool for scenario in scenarios for turn in scenario.turns
                 for tool in ((turn.expected_tools or ()) + tuple(
                     call.name for call in (turn.expected_tool_calls or ())
                 )))
    policy_actions = Counter(rule.action for scenario in scenarios for rule in scenario.must_not)
    policy_actions.update(rule.action for scenario in scenarios for turn in scenario.turns for rule in turn.must_not)
    return {
        "scenario_count": len(scenarios),
        "intent_labels": dict(sorted(intents.items())),
        "tool_labels": dict(sorted(tools.items())),
        "policy_actions": dict(sorted(policy_actions.items())),
        "tags": dict(sorted(tags.items())),
        "checks": {
            "required_question_scenarios": sum(bool(s.must_ask or any(t.must_ask for t in s.turns)) for s in scenarios),
            "tool_scenarios": sum(s.expected_tools is not None or s.expected_tool_calls is not None
                                   or any(t.expected_tools is not None or t.expected_tool_calls is not None for t in s.turns)
                                   for s in scenarios),
            "policy_scenarios": sum(bool(s.must_not or any(t.must_not for t in s.turns)) for s in scenarios),
            "answer_graded_scenarios": sum(bool(s.answer_graders) for s in scenarios),
            "multi_turn_scenarios": sum(bool(s.turns) for s in scenarios),
        },
    }


def _mutations(scenario: Scenario, result: AgentResult) -> list[tuple[str, AgentResult]]:
    mutations = [("intent", replace(result, intent="__agent_eval_mutated_intent__"))]
    mutations.append(("unnecessary_question", replace(
        result, asked=result.asked + ("__agent_eval_mutated_question__",)
    )))
    required = next((turn.must_ask[0] for turn in scenario.turns if turn.must_ask),
                    scenario.must_ask[0] if scenario.must_ask else None)
    if required:
        key = required.casefold()
        changed_turns = tuple(replace(turn, asked=tuple(q for q in turn.asked if q.casefold() != key))
                              for turn in result.turns)
        mutations.append(("required_question", replace(
            result, asked=tuple(question for question in result.asked if question.casefold() != key),
            turns=changed_turns,
        )))
    if scenario.expected_tools is not None:
        changed_turns = tuple(replace(turn, tools=("__agent_eval_mutated_tool__",)) for turn in result.turns)
        mutations.append(("tool_selection", replace(result, tools=("__agent_eval_mutated_tool__",), turns=changed_turns)))
    elif any(turn.expected_tools is not None for turn in scenario.turns):
        index = next(i for i, turn in enumerate(scenario.turns) if turn.expected_tools is not None)
        changed_turns = list(result.turns)
        if index < len(changed_turns):
            changed_turns[index] = replace(changed_turns[index], tools=("__agent_eval_mutated_tool__",))
            mutations.append(("tool_selection", replace(result, tools=("__agent_eval_mutated_tool__",), turns=tuple(changed_turns))))
    if scenario.expected_tool_calls is not None:
        mutations.append(("tool_call", replace(result, tool_calls=result.tool_calls + (ToolCall("__agent_eval_mutated_tool__"),))))
    elif any(turn.expected_tool_calls is not None for turn in scenario.turns):
        index = next(i for i, turn in enumerate(scenario.turns) if turn.expected_tool_calls is not None)
        changed_turns = list(result.turns)
        if index < len(changed_turns):
            changed_turns[index] = replace(
                changed_turns[index],
                tool_calls=changed_turns[index].tool_calls + (ToolCall("__agent_eval_mutated_tool__"),),
            )
            mutations.append(("tool_call", replace(
                result,
                tool_calls=result.tool_calls + (ToolCall("__agent_eval_mutated_tool__"),),
                turns=tuple(changed_turns),
            )))
    policy_rules = list(scenario.must_not) + [rule for turn in scenario.turns for rule in turn.must_not]
    for index, rule in enumerate(policy_rules, start=1):
        event = TraceEvent("action", rule.action)
        changed_turns = result.turns
        for turn_index, expected_turn in enumerate(scenario.turns):
            if rule in expected_turn.must_not and turn_index < len(result.turns):
                updated = list(result.turns)
                updated[turn_index] = replace(
                    updated[turn_index],
                    actions=updated[turn_index].actions + (rule.action,),
                    events=(event,) + updated[turn_index].events,
                )
                changed_turns = tuple(updated)
                break
        mutations.append((f"policy_action_{index}", replace(
            result,
            actions=result.actions + (rule.action,),
            events=(event,) + result.events,
            turns=changed_turns,
        )))
    return mutations


def mutation_report(
    scenarios: list[Scenario], results: dict[str, AgentResult]
) -> dict[str, Any]:
    """Apply representative regressions to passing traces and report if scoring catches them."""
    unknown = sorted(results.keys() - {scenario.name for scenario in scenarios})
    if unknown:
        raise ValueError("Results contain unknown scenario(s): " + ", ".join(unknown))
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        result = results.get(scenario.name)
        if result is None:
            rows.append({"scenario": scenario.name, "status": "missing_result", "mutations": []})
            continue
        # Do not invoke optional semantic graders during a mutation audit; this
        # measures the deterministic scenario assertions themselves.
        deterministic = replace(scenario, answer_graders=())
        if not score_scenario(deterministic, result).passed:
            rows.append({"scenario": scenario.name, "status": "baseline_failing", "mutations": []})
            continue
        attempts = []
        for mutation, changed in _mutations(deterministic, result):
            killed = not score_scenario(deterministic, changed).passed
            attempts.append({"mutation": mutation, "killed": killed})
        rows.append({"scenario": scenario.name, "status": "checked", "mutations": attempts})
    checked = [mutation for row in rows for mutation in row["mutations"]]
    killed = sum(mutation["killed"] for mutation in checked)
    return {
        "scenario_count": len(scenarios),
        "checked_scenario_count": sum(row["status"] == "checked" for row in rows),
        "mutation_count": len(checked),
        "killed": killed,
        "survived": len(checked) - killed,
        "mutation_score": killed / len(checked) if checked else None,
        "scenarios": rows,
    }


def text_coverage(report: dict[str, Any]) -> str:
    checks = report["checks"]
    lines = [f"Scenario coverage: {report['scenario_count']} scenarios", ""]
    lines.append("Declared checks:")
    lines.extend(f"  {label.replace('_', ' ')}: {count}/{report['scenario_count']} scenarios"
                 for label, count in checks.items())
    for label, key in (("Intent labels", "intent_labels"), ("Tool labels", "tool_labels"),
                       ("Policy actions", "policy_actions"), ("Tags", "tags")):
        lines.append("")
        lines.append(label + ":")
        values = report[key]
        lines.extend(f"  {name}: {count}" for name, count in values.items())
        if not values:
            lines.append("  none")
    return "\n".join(lines)


def text_mutation_report(report: dict[str, Any]) -> str:
    score = "n/a" if report["mutation_score"] is None else f"{report['mutation_score'] * 100:.1f}%"
    lines = [
        f"Mutation audit: {report['mutation_count']} mutations across {report['scenario_count']} scenarios",
        f"Killed: {report['killed']}",
        f"Survived: {report['survived']}",
        f"Mutation score: {score}",
    ]
    for scenario in report["scenarios"]:
        if scenario["status"] != "checked":
            lines.append(f"  {scenario['scenario']}: {scenario['status']}")
        for mutation in scenario["mutations"]:
            if not mutation["killed"]:
                lines.append(f"  SURVIVED {scenario['scenario']}: {mutation['mutation']}")
    return "\n".join(lines)
