"""Adapters for converting framework traces into Agent Eval's result format.

Framework SDKs evolve quickly, so the core package accepts small adapter
implementations instead of importing agent frameworks as dependencies. The
built-in path mapping adapter handles JSON traces from any framework.
"""

from __future__ import annotations

import re
import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol

import yaml

from agent_eval.models import AgentResult, ToolCall, TraceEvent


class AdapterError(ValueError):
    """Raised when a trace cannot be normalized by an adapter."""


def load_adapter_plugin(name: str) -> "TraceAdapter":
    """Load an optional adapter from the ``agent_eval.adapters`` entry-point group."""
    builtins = {"openai-agents": OpenAIAgentsAdapter, "langchain": LangChainEventsAdapter}
    if name in builtins:
        return builtins[name]()
    points = entry_points()
    plugins = points.select(group="agent_eval.adapters") if hasattr(points, "select") else points.get("agent_eval.adapters", [])
    matches = [point for point in plugins if point.name == name]
    if not matches:
        available = ", ".join(sorted(point.name for point in plugins)) or "none installed"
        raise AdapterError(f"unknown adapter '{name}' (available: {available})")
    if len(matches) > 1:
        raise AdapterError(f"multiple installed adapters are named '{name}'")
    try:
        adapter = matches[0].load()
        if isinstance(adapter, type):
            adapter = adapter()
        elif callable(adapter) and not hasattr(adapter, "adapt"):
            adapter = adapter()
    except Exception as exc:
        raise AdapterError(f"could not load adapter '{name}': {exc}") from exc
    if not hasattr(adapter, "adapt"):
        raise AdapterError(f"adapter '{name}' must provide adapt(record)")
    return adapter


class TraceAdapter(Protocol):
    """Normalize one framework-specific trace record into an agent result."""

    def adapt(self, record: dict[str, Any]) -> AgentResult:
        """Convert a raw trace object into the canonical result model."""


class OpenAIAgentsAdapter:
    """Normalize OpenAI Agents SDK exported span records (without importing its SDK)."""

    def adapt(self, record: dict[str, Any]) -> AgentResult:
        data = record.get("span_data", record)
        attrs = record.get("attributes", {})
        name = data.get("name") or data.get("tool_name") or ""
        scenario = attrs.get("agent_eval.scenario") or record.get("scenario")
        intent = attrs.get("agent_eval.intent") or record.get("intent")
        if not scenario or not intent:
            raise AdapterError("OpenAI Agents span requires agent_eval.scenario and agent_eval.intent attributes")
        call = []
        if name and ("tool" in str(record.get("span_type", "")).lower() or data.get("type") in {"function", "function_call"} or "arguments" in data):
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            call = [ToolCall(str(name), arguments if isinstance(arguments, dict) else {})]
        answer = attrs.get("agent.output") or data.get("output")
        return AgentResult(scenario=str(scenario), intent=str(intent),
                           tools=tuple(item.name for item in call), tool_calls=tuple(call),
                           answer=answer if isinstance(answer, str) else None,
                           metadata={"framework": "openai-agents", "model": attrs.get("gen_ai.request.model")})


class LangChainEventsAdapter:
    """Normalize LangChain/LangSmith run records and ``astream_events`` events."""

    def adapt(self, record: dict[str, Any]) -> AgentResult:
        metadata = record.get("metadata", {})
        extra = record.get("extra", {})
        metadata = {**(extra.get("metadata", {}) if isinstance(extra, dict) else {}),
                    **(metadata if isinstance(metadata, dict) else {})}
        scenario = metadata.get("agent_eval.scenario") or record.get("scenario")
        intent = metadata.get("agent_eval.intent") or record.get("intent")
        if not scenario or not intent:
            raise AdapterError("LangChain trace requires agent_eval.scenario and agent_eval.intent metadata")
        events = record.get("events", [record])
        if not isinstance(events, list):
            events = [record]
        calls: list[ToolCall] = []
        answer = record.get("answer")
        for event in events:
            if not isinstance(event, dict):
                continue
            event_name = event.get("event", "")
            data = event.get("data", {})
            if event_name in {"on_tool_start", "on_tool_end"} or event.get("type") == "tool":
                tool_name = event.get("name") or event.get("name", "")
                args = data.get("input", {}) if isinstance(data, dict) else {}
                if event_name == "on_tool_start" and tool_name:
                    calls.append(ToolCall(str(tool_name), args if isinstance(args, dict) else {}))
            if event_name in {"on_chain_end", "on_chat_model_end"} and isinstance(data, dict):
                output = data.get("output")
                if isinstance(output, str):
                    answer = output
        return AgentResult(scenario=str(scenario), intent=str(intent),
                           tools=tuple(call.name for call in calls), tool_calls=tuple(calls),
                           answer=answer if isinstance(answer, str) else None,
                           metadata={"framework": "langchain", "model": metadata.get("ls_model_name")})


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
        "scenario", "intent", "asked", "tools", "actions", "tool_calls", "events", "answer", "notes"
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
        answer = values.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise AdapterError(f"trace for scenario '{scenario}': answer must resolve to a string")
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
            answer=answer,
            notes=notes,
        )


def result_as_dict(result: AgentResult) -> dict[str, Any]:
    """Serialize a canonical result to the public JSONL schema."""
    output = {
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
        "answer": result.answer,
        "metadata": result.metadata,
    }
    if result.turns:
        output["turns"] = [
            {
                "intent": turn.intent,
                "asked": list(turn.asked),
                "tools": list(turn.tools),
                "actions": list(turn.actions),
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in turn.tool_calls
                ],
                "events": [{"type": event.event_type, "name": event.name} for event in turn.events],
                "answer": turn.answer,
                "notes": turn.notes,
                "metadata": turn.metadata,
            }
            for turn in result.turns
        ]
    return output
