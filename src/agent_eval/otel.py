"""Read agent runs from an OpenTelemetry OTLP/JSON trace export."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent_eval.adapters import AdapterError
from agent_eval.models import AgentResult, ToolCall, TraceEvent


def _decode_any_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        raw = value["intValue"]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "arrayValue" in value:
        return [_decode_any_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return _attributes(value["kvlistValue"].get("values", []))
    if "bytesValue" in value:
        return value["bytesValue"]
    return value


def _attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {key: _decode_any_value(item) for key, item in value.items()}
    if not isinstance(value, list):
        return {}
    output: dict[str, Any] = {}
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("key"), str):
            output[item["key"]] = _decode_any_value(item.get("value"))
    return output


def _as_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _span_operation(attributes: dict[str, Any], span_name: str) -> str:
    operation = attributes.get("gen_ai.operation.name")
    return str(operation if operation is not None else span_name).casefold()


def _time(span: dict[str, Any]) -> int:
    try:
        return int(span.get("startTimeUnixNano", 0))
    except (TypeError, ValueError):
        return 0


def _extract_text(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return value
            extracted = _extract_text(parsed)
            if extracted is not None:
                return extracted
        return value
    if isinstance(value, list):
        chunks = [text for item in value if (text := _extract_text(item))]
        return "\n".join(chunks) if chunks else None
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("output_text"), str):
            return value["output_text"]
        if "content" in value:
            return _extract_text(value["content"])
        # A message may wrap content in a provider-specific body field.
        for key in ("message", "messages", "parts"):
            if key in value:
                extracted = _extract_text(value[key])
                if extracted:
                    return extracted
    return None


class OpenTelemetryAdapter:
    """Convert OTLP/JSON agent spans into canonical Agent Eval results.

    The adapter uses ``gen_ai.operation.name`` for agent and tool spans. A
    deployment-specific intent and scenario identifier are read from span or
    resource attributes, defaulting to ``agent_eval.intent`` and
    ``agent_eval.scenario``. Content-bearing GenAI attributes are ignored unless
    ``include_sensitive_attributes`` is enabled.
    """

    def __init__(
        self,
        *,
        scenario_attribute: str = "agent_eval.scenario",
        intent_attribute: str = "agent_eval.intent",
        include_sensitive_attributes: bool = False,
    ):
        self.scenario_attribute = scenario_attribute
        self.intent_attribute = intent_attribute
        self.include_sensitive_attributes = include_sensitive_attributes

    def adapt_document(self, document: Any) -> list[AgentResult]:
        if not isinstance(document, dict) or not isinstance(document.get("resourceSpans"), list):
            raise AdapterError("OTLP JSON must contain a 'resourceSpans' array")
        spans_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for resource_span in document["resourceSpans"]:
            if not isinstance(resource_span, dict):
                continue
            resource_attributes = _attributes(resource_span.get("resource", {}).get("attributes", []))
            for scope in resource_span.get("scopeSpans", []):
                if not isinstance(scope, dict):
                    continue
                for raw_span in scope.get("spans", []):
                    if not isinstance(raw_span, dict):
                        continue
                    trace_id = raw_span.get("traceId")
                    if not isinstance(trace_id, str):
                        continue
                    span = dict(raw_span)
                    span["_attributes"] = {**resource_attributes, **_attributes(raw_span.get("attributes", []))}
                    spans_by_trace[trace_id].append(span)

        results: list[AgentResult] = []
        for trace_id, spans in spans_by_trace.items():
            by_id = {span.get("spanId"): span for span in spans if isinstance(span.get("spanId"), str)}
            roots = [
                span
                for span in spans
                if not span.get("parentSpanId")
                and _span_operation(span["_attributes"], str(span.get("name", ""))) == "invoke_agent"
            ]
            if not roots:
                # Some exporters omit the parent ID on the root but still mark
                # the agent invocation with the semantic operation attribute.
                roots = [
                    span for span in spans
                    if _span_operation(span["_attributes"], str(span.get("name", ""))) == "invoke_agent"
                ]
            for root in roots:
                members = self._descendants(root, by_id)
                attributes = root["_attributes"]
                scenario = attributes.get(self.scenario_attribute)
                intent = attributes.get(self.intent_attribute)
                if not isinstance(scenario, str) or not scenario:
                    raise AdapterError(
                        f"trace {trace_id}: agent span is missing '{self.scenario_attribute}'"
                    )
                if not isinstance(intent, str) or not intent:
                    raise AdapterError(
                        f"trace {trace_id}: agent span is missing '{self.intent_attribute}'"
                    )
                results.append(self._make_result(scenario, intent, members))
        if not results:
            raise AdapterError("No OpenTelemetry 'invoke_agent' spans were found")
        return results

    @staticmethod
    def _descendants(root: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        root_id = root.get("spanId")
        selected: list[dict[str, Any]] = [root]
        parents = {root_id}
        changed = True
        while changed:
            changed = False
            for span in by_id.values():
                span_id = span.get("spanId")
                if span_id not in parents and span.get("parentSpanId") in parents:
                    selected.append(span)
                    parents.add(span_id)
                    changed = True
        return sorted(selected, key=_time)

    def _make_result(self, scenario: str, intent: str, spans: list[dict[str, Any]]) -> AgentResult:
        tools: list[str] = []
        tool_calls: list[ToolCall] = []
        timestamped_events: list[tuple[int, TraceEvent]] = []
        answer: str | None = None
        for span in spans:
            attributes = span["_attributes"]
            operation = _span_operation(attributes, str(span.get("name", "")))
            if operation == "execute_tool":
                tool_name = attributes.get("gen_ai.tool.name", span.get("name"))
                if isinstance(tool_name, str) and tool_name:
                    tools.append(tool_name)
                    raw_arguments = attributes.get("gen_ai.tool.call.arguments", {})
                    arguments = _as_object(raw_arguments) if self.include_sensitive_attributes else {}
                    tool_calls.append(ToolCall(tool_name, arguments))

            event_type = attributes.get("agent_eval.event.type")
            event_name = attributes.get("agent_eval.event.name")
            if event_type in {"question", "action"} and isinstance(event_name, str):
                timestamped_events.append((_time(span), TraceEvent(event_type, event_name)))

            for raw_event in span.get("events", []):
                if not isinstance(raw_event, dict):
                    continue
                event_attributes = _attributes(raw_event.get("attributes", []))
                event_type = event_attributes.get("agent_eval.event.type")
                event_name = event_attributes.get("agent_eval.event.name")
                if event_type in {"question", "action"} and isinstance(event_name, str):
                    try:
                        timestamp = int(raw_event.get("timeUnixNano", _time(span)))
                    except (TypeError, ValueError):
                        timestamp = _time(span)
                    timestamped_events.append((timestamp, TraceEvent(event_type, event_name)))

            if self.include_sensitive_attributes:
                output = attributes.get("gen_ai.output.messages")
                if output is None:
                    output = attributes.get("agent_eval.answer")
                extracted = _extract_text(output)
                if extracted:
                    answer = extracted

        events = [event for _, event in sorted(timestamped_events, key=lambda item: item[0])]
        asked = tuple(dict.fromkeys(event.name for event in events if event.event_type == "question"))
        actions = tuple(dict.fromkeys(event.name for event in events if event.event_type == "action"))
        return AgentResult(
            scenario=scenario,
            intent=intent,
            asked=asked,
            tools=tuple(tools),
            actions=actions,
            tool_calls=tuple(tool_calls),
            events=tuple(events),
            answer=answer,
        )


def load_otel_document(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AdapterError(f"Could not read OTLP trace export {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AdapterError(f"Invalid OTLP JSON in {path}: {exc.msg}") from exc
