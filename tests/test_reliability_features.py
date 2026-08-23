import json
import tempfile
import unittest
from pathlib import Path

from agent_eval.adapters import load_adapter_plugin
from agent_eval.models import AgentResult, Scenario
from agent_eval.quality import judge_agreement, scenario_suggestions
from agent_eval.results import load_results
from agent_eval.runner import execute_scenarios
from agent_eval.tools import ToolFixtureError, ToolSandbox


class ReliabilityFeatureTests(unittest.TestCase):
    def test_builtin_framework_adapters(self):
        openai = load_adapter_plugin("openai-agents").adapt({
            "scenario": "lookup", "intent": "LOOKUP", "span_type": "function",
            "span_data": {"name": "search", "arguments": '{"q":"hours"}'},
        })
        self.assertEqual(openai.tool_calls[0].arguments, {"q": "hours"})
        langchain = load_adapter_plugin("langchain").adapt({
            "scenario": "lookup", "intent": "LOOKUP",
            "events": [{"event": "on_tool_start", "name": "search", "data": {"input": {"q": "hours"}}}],
        })
        self.assertEqual(langchain.tools, ("search",))

    def test_fixture_sandbox_is_deny_by_default(self):
        scenario = Scenario(name="s", input={"user": "hi"}, intent="GREET",
                            tool_fixtures=({"name": "lookup", "arguments": {"id": 7}, "response": {"ok": True}},))
        sandbox = ToolSandbox(scenario)
        self.assertEqual(sandbox.call("lookup", {"id": 7}), {"ok": True})
        with self.assertRaises(ToolFixtureError):
            sandbox.call("dangerous", {})

    def test_result_metadata_round_trips(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.jsonl"
            path.write_text(json.dumps({"scenario": "s", "intent": "GREET", "metadata": {"model": "test", "cost_usd": 0.2}}) + "\n")
            self.assertEqual(load_results(path)["s"].metadata["cost_usd"], 0.2)

    def test_suggestions_and_judge_agreement(self):
        scenario = Scenario(name="s", input={"user": "hello"}, intent="GREET")
        report = scenario_suggestions([scenario])
        self.assertTrue(report["review_required"])
        self.assertTrue(report["candidates"])
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for index, passed in enumerate((True, False)):
                path = Path(temp) / f"{index}.json"
                path.write_text(json.dumps({"scenarios": [{"scenario": "s", "answer_grades": [
                    {"score": 0.8 if passed else 0.4, "passed": passed,
                     "metadata": {"judge": f"judge{index}"}}
                ]}]}))
                paths.append(path)
            audit = judge_agreement(paths)
            self.assertEqual(audit["scenario_count"], 1)
            self.assertEqual(audit["scenarios"][0]["agreement"], 0.5)


if __name__ == "__main__":
    unittest.main()
