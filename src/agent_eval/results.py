"""Read JSON Lines output captured from an agent under evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_eval.models import AgentResult, AgentTurnResult, ToolCall, TraceEvent


class ResultError(ValueError):
    """Raised when an agent result file is invalid or incomplete."""


def _list_of_strings(value: Any, field: str, line: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ResultError(f"line {line}: '{field}' must be a list of strings")
    return tuple(value)


def _tool_calls(value: Any, line: int) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ResultError(f"line {line}: 'tool_calls' must be a list")
    calls: list[ToolCall] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ResultError(f"line {line}: 'tool_calls[{index}].name' must be a string")
        arguments = item.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ResultError(f"line {line}: 'tool_calls[{index}].arguments' must be an object")
        calls.append(ToolCall(name=item["name"], arguments=arguments))
    return tuple(calls)


def _events(value: Any, line: int) -> tuple[TraceEvent, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ResultError(f"line {line}: 'events' must be a list")
    events: list[TraceEvent] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ResultError(f"line {line}: 'events[{index}]' must be an object")
        event_type = item.get("type")
        name = item.get("name")
        if event_type not in {"question", "action"} or not isinstance(name, str):
            raise ResultError(
                f"line {line}: 'events[{index}]' needs type 'question' or 'action' and a string name"
            )
        events.append(TraceEvent(event_type, name))
    return tuple(events)


def _turn_results(value: Any, line: int) -> tuple[AgentTurnResult, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise ResultError(f"line {line}: 'turns' must be a non-empty list")
    turns: list[AgentTurnResult] = []
    for index, item in enumerate(value):
        label = f"turns[{index}]"
        if not isinstance(item, dict) or not isinstance(item.get("intent"), str) or not item["intent"]:
            raise ResultError(f"line {line}: '{label}.intent' must be a non-empty string")
        calls = _tool_calls(item.get("tool_calls"), line)
        tools = _list_of_strings(item.get("tools"), f"{label}.tools", line)
        if "tools" not in item and calls:
            tools = tuple(call.name for call in calls)
        turns.append(AgentTurnResult(
            intent=item["intent"],
            asked=_list_of_strings(item.get("asked"), f"{label}.asked", line),
            tools=tools,
            actions=_list_of_strings(item.get("actions"), f"{label}.actions", line),
            tool_calls=calls,
            events=_events(item.get("events"), line),
            answer=item.get("answer"),
            notes=item.get("notes", ""),
        ))
        if turns[-1].answer is not None and not isinstance(turns[-1].answer, str):
            raise ResultError(f"line {line}: '{label}.answer' must be a string")
        if not isinstance(turns[-1].notes, str):
            raise ResultError(f"line {line}: '{label}.notes' must be a string")
    return tuple(turns)


def load_results(path: Path) -> dict[str, AgentResult]:
    results: dict[str, AgentResult] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ResultError(f"Could not read results file {path}: {exc}") from exc

    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ResultError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ResultError(f"line {line_number}: each result must be a JSON object")
        scenario = item.get("scenario")
        turns = _turn_results(item.get("turns"), line_number)
        intent = item.get("intent", turns[-1].intent if turns else None)
        if not isinstance(scenario, str) or not scenario:
            raise ResultError(f"line {line_number}: 'scenario' must be a non-empty string")
        if not isinstance(intent, str) or not intent:
            raise ResultError(f"line {line_number}: 'intent' must be a non-empty string (or provide turns)")
        if scenario in results:
            raise ResultError(f"line {line_number}: duplicate result for '{scenario}'")
        notes = item.get("notes", "")
        if not isinstance(notes, str):
            raise ResultError(f"line {line_number}: 'notes' must be a string")
        asked = _list_of_strings(item.get("asked"), "asked", line_number)
        tools = _list_of_strings(item.get("tools"), "tools", line_number)
        actions = _list_of_strings(item.get("actions"), "actions", line_number)
        calls = _tool_calls(item.get("tool_calls"), line_number)
        events = _events(item.get("events"), line_number)
        if turns:
            if "asked" not in item:
                asked = tuple(value for turn in turns for value in turn.asked)
            if "tools" not in item:
                tools = tuple(value for turn in turns for value in turn.tools)
            if "actions" not in item:
                actions = tuple(value for turn in turns for value in turn.actions)
            if "tool_calls" not in item:
                calls = tuple(value for turn in turns for value in turn.tool_calls)
            if "events" not in item:
                events = tuple(value for turn in turns for value in turn.events)
        if "tools" not in item and calls:
            tools = tuple(call.name for call in calls)
        answer = item.get("answer", turns[-1].answer if turns else None)
        if answer is not None and not isinstance(answer, str):
            raise ResultError(f"line {line_number}: 'answer' must be a string")
        results[scenario] = AgentResult(
            scenario=scenario,
            intent=intent,
            asked=asked,
            tools=tools,
            actions=actions,
            tool_calls=calls,
            events=events,
            answer=answer,
            notes=notes,
            turns=turns,
        )
    return results
