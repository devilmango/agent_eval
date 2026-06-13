"""Read JSON Lines output captured from an agent under evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_eval.models import AgentResult, ToolCall, TraceEvent


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
        intent = item.get("intent")
        if not isinstance(scenario, str) or not scenario:
            raise ResultError(f"line {line_number}: 'scenario' must be a non-empty string")
        if not isinstance(intent, str) or not intent:
            raise ResultError(f"line {line_number}: 'intent' must be a non-empty string")
        if scenario in results:
            raise ResultError(f"line {line_number}: duplicate result for '{scenario}'")
        notes = item.get("notes", "")
        if not isinstance(notes, str):
            raise ResultError(f"line {line_number}: 'notes' must be a string")
        calls = _tool_calls(item.get("tool_calls"), line_number)
        tools = _list_of_strings(item.get("tools"), "tools", line_number)
        if "tools" not in item and calls:
            tools = tuple(call.name for call in calls)
        results[scenario] = AgentResult(
            scenario=scenario,
            intent=intent,
            asked=_list_of_strings(item.get("asked"), "asked", line_number),
            tools=tools,
            actions=_list_of_strings(item.get("actions"), "actions", line_number),
            tool_calls=calls,
            events=_events(item.get("events"), line_number),
            notes=notes,
        )
    return results
