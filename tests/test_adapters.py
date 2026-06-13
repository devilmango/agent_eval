import unittest

from agent_eval.adapters import PathMappingAdapter


class PathMappingAdapterTests(unittest.TestCase):
    def test_maps_nested_framework_trace_and_tool_calls(self):
        adapter = PathMappingAdapter(
            {
                "scenario": "metadata.scenario_id",
                "intent": "agent.final.intent",
                "asked": "agent.final.questions[*]",
                "tools": "trace.tool_calls[*].name",
                "tool_calls": "trace.tool_calls",
                "actions": "trace.actions",
                "events": "trace.events",
            }
        )
        result = adapter.adapt(
            {
                "metadata": {"scenario_id": "booking"},
                "agent": {"final": {"intent": "BOOK", "questions": ["date"]}},
                "trace": {
                    "tool_calls": [
                        {"name": "lookup", "arguments": {"id": "p-1"}},
                        {"name": "reserve", "arguments": {"slot": "s-3"}},
                    ],
                    "actions": [],
                    "events": [{"type": "question", "name": "date"}],
                },
            }
        )
        self.assertEqual(result.scenario, "booking")
        self.assertEqual(result.asked, ("date",))
        self.assertEqual(result.tools, ("lookup", "reserve"))
        self.assertEqual([call.name for call in result.tool_calls], ["lookup", "reserve"])
        self.assertEqual(result.events[0].event_type, "question")

    def test_requires_scenario_and_intent_mappings(self):
        with self.assertRaisesRegex(ValueError, "requires a path for 'intent'"):
            PathMappingAdapter({"scenario": "scenario"})
