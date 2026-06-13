"""Adapters for converting framework traces into Agent Eval's result format.

Framework SDKs evolve quickly, so the core package accepts small adapter
implementations instead of importing agent frameworks as dependencies. The
built-in path mapping adapter handles JSON traces from any framework.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

import yaml

from agent_eval.models import AgentResult, ToolCall, TraceEvent


class AdapterError(ValueError):
    """Raised when a trace cannot be normalized by an adapter."""


class TraceAdapter(Protocol):
    """Normalize one framework-specific trace record into an agent result."""

    def adapt(self, record: dict[str, Any]) -> AgentResult:
        """Convert a raw trace object into the canonical result model."""


_PATH_TOKEN = re.compile(r"[^.\[\]]+|\[\*\]|\[\d+\]")


def _resolve_path(record: Any, path: str) -> Any:
    path = path.strip()
    if path.startswith("$"):
        path = path[1:].lstrip(".")
    tokens = _PATH_TOKEN.findall(path)
    residue = re.sub(r"\[\*\]|\[\d+\]|[A-Za-z0-9_-]+|\.", "", path)
    if not tokens or residue:
        raise AdapterError(f"Invalid field path: {path!r}")

    values = [record]
    for token in tokens:
        next_values: list[Any] = []
        for value in values:
            if token == "[*]":
                if isinstance(value, list):
                    next_values.extend(value)
            elif token.startswith("["):
                index = int(token[1:-1])
                if isinstance(value, list) and index < len(value):
                    next_values.append(value[index])
            elif isinstance(value, dict) and token in value:
                next_values.append(value[token])
        values = next_values
    if not values:
        return None
    return values if len(values) > 1 or "[*]" in path else values[0]


def _as_strings(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise AdapterError(f"Mapped field '{field}' must resolve to a string or list of strings")
    return tuple(value)


class PathMappingAdapter:
    """Map dotted JSON paths from an agent trace to canonical result fields.

    Mapping paths may contain ``[*]`` to extract values from each element of a
    list, for example ``trace.tool_calls[*].name``.
    """

    ALLOWED_FIELDS = {
        "scenario", "intent", "asked", "tools", "actions", "tool_calls", "events", "notes"
    }

    def __init__(self, fields: dict[str, str]):
        unknown = sorted(set(fields) - self.ALLOWED_FIELDS)
        if unknown:
            raise AdapterError(f"Unknown result field(s) in mapping: {', '.join(unknown)}")
        for required in ("scenario", "intent"):
            if not isinstance(fields.get(required), str) or not fields[required]:
                raise AdapterError(f"Adapter mapping requires a path for '{required}'")
        if any(not isinstance(path, str) for path in fields.values()):
            raise AdapterError("Every adapter field mapping must be a string path")
        self.fields = fields

    @classmethod
    def from_yaml(cls, path: Path) -> "PathMappingAdapter":
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise AdapterError(f"Could not read adapter mapping {path}: {exc}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("fields"), dict):
            raise AdapterError(f"{path}: adapter mapping must contain a 'fields' mapping")
        return cls(document["fields"])

    def adapt(self, record: dict[str, Any]) -> AgentResult:
        values = {key: _resolve_path(record, path) for key, path in self.fields.items()}
        scenario = values.get("scenario")
        intent = values.get("intent")
        if not isinstance(scenario, str) or not scenario:
            raise AdapterError("Mapped 'scenario' path did not resolve to a non-empty string")
        if not isinstance(intent, str) or not intent:
            raise AdapterError(f"trace for scenario '{scenario}' has no mapped intent string")

        calls_value = values.get("tool_calls")
        calls: list[ToolCall] = []
        if calls_value is not None:
            if not isinstance(calls_value, list):
                raise AdapterError(f"trace for scenario '{scenario}': tool_calls must resolve to a list")
            for index, call in enumerate(calls_value):
                if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                    raise AdapterError(
                        f"trace for scenario '{scenario}': tool_calls[{index}] needs a name"
                    )
                arguments = call.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise AdapterError(
                        f"trace for scenario '{scenario}': tool_calls[{index}].arguments must be an object"
                    )
                calls.append(ToolCall(call["name"], arguments))

        tools = _as_strings(values.get("tools"), "tools")
        if "tools" not in values and calls:
            tools = tuple(call.name for call in calls)
        notes = values.get("notes", "")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            raise AdapterError(f"trace for scenario '{scenario}': notes must resolve to a string")
        events_value = values.get("events")
        events: list[TraceEvent] = []
        if events_value is not None:
            if not isinstance(events_value, list):
                raise AdapterError(f"trace for scenario '{scenario}': events must resolve to a list")
            for index, event in enumerate(events_value):
                if (
                    not isinstance(event, dict)
                    or event.get("type") not in {"question", "action"}
                    or not isinstance(event.get("name"), str)
                ):
                    raise AdapterError(
                        f"trace for scenario '{scenario}': events[{index}] needs type and name"
                    )
                events.append(TraceEvent(event["type"], event["name"]))
        return AgentResult(
            scenario=scenario,
            intent=intent,
            asked=_as_strings(values.get("asked"), "asked"),
            tools=tools,
            actions=_as_strings(values.get("actions"), "actions"),
            tool_calls=tuple(calls),
            events=tuple(events),
            notes=notes,
        )


def result_as_dict(result: AgentResult) -> dict[str, Any]:
    """Serialize a canonical result to the public JSONL schema."""
    return {
        "scenario": result.scenario,
        "intent": result.intent,
        "asked": list(result.asked),
        "tools": list(result.tools),
        "actions": list(result.actions),
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments} for call in result.tool_calls
        ],
        "events": [
            {"type": event.event_type, "name": event.name} for event in result.events
        ],
        "notes": result.notes,
    }
