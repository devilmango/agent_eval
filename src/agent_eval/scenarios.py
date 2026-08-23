"""Load and validate YAML evaluation scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_eval.models import PolicyRule, Scenario, ScenarioTurn, ToolCallExpectation


class ScenarioError(ValueError):
    """Raised when a scenario is missing required or well-formed fields."""


def _string_list(value: Any, field_name: str, source: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ScenarioError(f"{source}: '{field_name}' must be a list of strings")
    return tuple(value)


def _tool_call_expectations(value: Any, source: Path) -> tuple[ToolCallExpectation, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ScenarioError(f"{source}: 'expected.tool_calls' must be a list")
    calls: list[ToolCallExpectation] = []
    for index, item in enumerate(value):
        label = f"expected.tool_calls[{index}]"
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ScenarioError(f"{source}: '{label}.name' must be a string")
        arguments = item.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ScenarioError(f"{source}: '{label}.arguments' must be a mapping")
        for argument, matcher in arguments.items():
            if not isinstance(argument, str):
                raise ScenarioError(f"{source}: '{label}' argument names must be strings")
            if isinstance(matcher, dict):
                if len(matcher) != 1 or next(iter(matcher), None) not in {
                    "equals", "not_equals", "one_of", "contains", "exists"
                }:
                    raise ScenarioError(
                        f"{source}: '{label}.arguments.{argument}' must use one supported matcher"
                    )
                operator, expected = next(iter(matcher.items()))
                if operator == "one_of" and not isinstance(expected, list):
                    raise ScenarioError(
                        f"{source}: '{label}.arguments.{argument}.one_of' must be a list"
                    )
                if operator == "exists" and not isinstance(expected, bool):
                    raise ScenarioError(
                        f"{source}: '{label}.arguments.{argument}.exists' must be a boolean"
                    )
        calls.append(ToolCallExpectation(name=item["name"], arguments=arguments))
    return tuple(calls)


def _policy_rules(value: Any, source: Path) -> tuple[PolicyRule, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ScenarioError(f"{source}: 'must_not' must be a list")
    rules: list[PolicyRule] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            rules.append(PolicyRule(action=item))
            continue
        label = f"must_not[{index}]"
        if not isinstance(item, dict) or not isinstance(item.get("action"), str):
            raise ScenarioError(f"{source}: '{label}' must be a string or mapping with 'action'")
        unless = item.get("unless", {})
        if not isinstance(unless, dict):
            raise ScenarioError(f"{source}: '{label}.unless' must be a mapping")
        unexpected = set(unless) - {"asked"}
        if unexpected:
            raise ScenarioError(f"{source}: unsupported '{label}.unless' field(s): {', '.join(sorted(unexpected))}")
        required = unless.get("asked", [])
        if isinstance(required, str):
            required = [required]
        rules.append(
            PolicyRule(
                action=item["action"],
                unless_asked=_string_list(required, f"{label}.unless.asked", source),
            )
        )
    return tuple(rules)


def load_scenario(path: Path) -> Scenario:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScenarioError(f"{path}: could not read YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise ScenarioError(f"{path}: scenario must be a YAML mapping")

    name = document.get("scenario")
    user_input = document.get("input")
    raw_turns = document.get("turns")
    expected = document.get("expected", {})
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(f"{path}: 'scenario' must be a non-empty string")
    if not isinstance(expected, dict) or not isinstance(expected.get("intent"), str):
        raise ScenarioError(f"{path}: 'expected.intent' must be a string")

    turns: list[ScenarioTurn] = []
    if raw_turns is not None:
        if user_input is not None:
            raise ScenarioError(f"{path}: use either 'input' or 'turns', not both")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ScenarioError(f"{path}: 'turns' must be a non-empty list")
        for index, raw_turn in enumerate(raw_turns):
            label = f"turns[{index}]"
            if not isinstance(raw_turn, dict) or not isinstance(raw_turn.get("input"), dict) or not raw_turn["input"]:
                raise ScenarioError(f"{path}: '{label}.input' must be a non-empty mapping")
            turn_expected = raw_turn.get("expected", {})
            if not isinstance(turn_expected, dict):
                raise ScenarioError(f"{path}: '{label}.expected' must be a mapping")
            turn_intent = turn_expected.get("intent")
            if turn_intent is not None and not isinstance(turn_intent, str):
                raise ScenarioError(f"{path}: '{label}.expected.intent' must be a string")
            turn_tools = turn_expected.get("tools")
            if turn_tools is not None:
                turn_tools = _string_list(turn_tools, f"{label}.expected.tools", path)
            turns.append(ScenarioTurn(
                input=raw_turn["input"],
                intent=turn_intent,
                must_ask=_string_list(turn_expected.get("must_ask"), f"{label}.expected.must_ask", path),
                expected_tools=turn_tools,
                must_not=_policy_rules(raw_turn.get("must_not"), path),
                allowed_questions=_string_list(raw_turn.get("allowed_questions"), f"{label}.allowed_questions", path),
                expected_tool_calls=_tool_call_expectations(turn_expected.get("tool_calls"), path),
            ))
        user_input = turns[0].input
    elif not isinstance(user_input, dict) or not user_input:
        raise ScenarioError(f"{path}: 'input' must be a non-empty mapping when 'turns' is omitted")

    expected_tools = expected.get("tools")
    if expected_tools is not None:
        expected_tools = _string_list(expected_tools, "expected.tools", path)

    expected_tool_calls = _tool_call_expectations(expected.get("tool_calls"), path)
    answer_graders = expected.get("answer_graders", [])
    if not isinstance(answer_graders, list) or any(
        not isinstance(grader, dict) or not isinstance(grader.get("type"), str)
        for grader in answer_graders
    ):
        raise ScenarioError(f"{path}: 'expected.answer_graders' must be a list of mappings with a type")
    tags = _string_list(document.get("tags"), "tags", path)
    if len({tag.casefold() for tag in tags}) != len(tags):
        raise ScenarioError(f"{path}: 'tags' must not contain duplicates")
    fixtures = document.get("tool_fixtures", [])
    if not isinstance(fixtures, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("name"), str)
        or ("response" not in item and "error" not in item)
        for item in fixtures
    ):
        raise ScenarioError(f"{path}: 'tool_fixtures' must be a list of mappings with name and response")
    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture.get("arguments", {}), dict):
            raise ScenarioError(f"{path}: 'tool_fixtures[{index}].arguments' must be a mapping")

    return Scenario(
        name=name,
        input=user_input,
        intent=expected["intent"],
        must_ask=_string_list(expected.get("must_ask"), "expected.must_ask", path),
        expected_tools=expected_tools,
        must_not=_policy_rules(document.get("must_not"), path),
        allowed_questions=_string_list(
            document.get("allowed_questions"), "allowed_questions", path
        ),
        expected_tool_calls=expected_tool_calls,
        answer_graders=tuple(answer_graders),
        tags=tags,
        source=str(path),
        turns=tuple(turns),
        tool_fixtures=tuple(fixtures),
    )


def load_scenarios(path: Path) -> list[Scenario]:
    """Load one scenario file or all YAML files under a directory."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}
        )
    else:
        raise ScenarioError(f"Scenario path does not exist: {path}")

    if not files:
        raise ScenarioError(f"No .yaml or .yml scenario files found in {path}")

    scenarios = [load_scenario(file) for file in files]
    names = [scenario.name for scenario in scenarios]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ScenarioError(f"Duplicate scenario name(s): {', '.join(duplicates)}")
    return scenarios


def filter_scenarios(
    scenarios: list[Scenario], *, include_tags: tuple[str, ...] = (), exclude_tags: tuple[str, ...] = ()
) -> list[Scenario]:
    """Select scenarios matching any included tag and no excluded tag."""
    included = {tag.casefold() for tag in include_tags}
    excluded = {tag.casefold() for tag in exclude_tags}
    selected = [
        scenario
        for scenario in scenarios
        if (not included or included.intersection(tag.casefold() for tag in scenario.tags))
        and not excluded.intersection(tag.casefold() for tag in scenario.tags)
    ]
    if not selected:
        details = []
        if included:
            details.append("tags=" + ",".join(sorted(included)))
        if excluded:
            details.append("excluded=" + ",".join(sorted(excluded)))
        raise ScenarioError("No scenarios match the requested filters" + (": " + "; ".join(details) if details else ""))
    return selected
