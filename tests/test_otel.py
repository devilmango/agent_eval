import unittest

from agent_eval.otel import OpenTelemetryAdapter


def attribute(key, value):
    return {"key": key, "value": {"stringValue": value}}


def fixture():
    return {
        "resourceSpans": [{"scopeSpans": [{"spans": [
            {
                "traceId": "trace-1", "spanId": "root", "name": "agent",
                "startTimeUnixNano": "100",
                "attributes": [attribute("gen_ai.operation.name", "invoke_agent"),
                               attribute("agent_eval.scenario", "booking"),
                               attribute("agent_eval.intent", "BOOK"),
                               attribute("agent_eval.event.type", "question"),
                               attribute("agent_eval.event.name", "identity")],
            },
            {
                "traceId": "trace-1", "spanId": "tool", "parentSpanId": "root", "name": "lookup",
                "startTimeUnixNano": "200",
                "attributes": [attribute("gen_ai.operation.name", "execute_tool"),
                               attribute("gen_ai.tool.name", "lookup"),
                               attribute("gen_ai.tool.call.arguments", '{"patient_id":"demo"}')],
            },
            {
                "traceId": "trace-1", "spanId": "model", "parentSpanId": "root", "name": "chat",
                "startTimeUnixNano": "300",
                "attributes": [attribute("gen_ai.output.messages", '[{"role":"assistant","content":"Booked."}]')],
            },
        ]}]}],
    }


class OpenTelemetryAdapterTests(unittest.TestCase):
    def test_adapts_trace_without_capturing_content_by_default(self):
        result = OpenTelemetryAdapter().adapt_document(fixture())[0]
        self.assertEqual(result.scenario, "booking")
        self.assertEqual(result.tools, ("lookup",))
        self.assertEqual(result.asked, ("identity",))
        self.assertEqual(result.tool_calls[0].arguments, {})
        self.assertIsNone(result.answer)

    def test_sensitive_capture_includes_tool_arguments_and_child_answer(self):
        result = OpenTelemetryAdapter(include_sensitive_attributes=True).adapt_document(fixture())[0]
        self.assertEqual(result.tool_calls[0].arguments, {"patient_id": "demo"})
        self.assertEqual(result.answer, "Booked.")

    def test_requires_scenario_and_intent_attributes(self):
        document = fixture()
        document["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"] = [
            attribute("gen_ai.operation.name", "invoke_agent")
        ]
        with self.assertRaisesRegex(ValueError, "agent_eval.scenario"):
            OpenTelemetryAdapter().adapt_document(document)

