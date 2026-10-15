import json
import io
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_eval.models import (
    AgentResult, AgentTurnResult, PolicyRule, Scenario, ScenarioTurn,
    ToolCall, ToolCallExpectation,
)
from agent_eval.cli import main
from agent_eval.report import html_report, json_report, junit_report, sarif_report, text_report
from agent_eval.runner import execute_scenarios
from agent_eval.scoring import evaluate, score_scenario


class CountingRunner:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run_turn(self, scenario, user_input, history):
        with self.lock:
            self.calls.append(scenario.name)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
        return AgentTurnResult(intent=scenario.intent, answer="ok")


class DiagnosticRunner:
    version = "test-v1"

    def run_turn(self, scenario, user_input, history):
        return AgentTurnResult(intent=scenario.intent, answer="ok")


class ExecutionAndDiagnosticsTests(unittest.TestCase):
    def test_run_agent_cli_reports_parallel_cache_execution_stats(self):
        class Runner:
            version = "test-v1"

            def run_turn(self, scenario, user_input, history):
                return AgentTurnResult(intent=scenario.intent)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenarios = root / "scenario.yaml"
            scenarios.write_text(
                "scenario: sample\ninput:\n  user: hi\nexpected:\n  intent: OK\n",
                encoding="utf-8",
            )
            cache = root / "cache.json"
            for expected_executed, expected_reused in ((1, 0), (0, 1)):
                report = root / f"report-{expected_reused}.json"
                results = root / f"results-{expected_reused}.jsonl"
                output = io.StringIO()
                with patch("agent_eval.cli.load_runner", return_value=Runner()):
                    with redirect_stdout(output):
                        status = main([
                            "run-agent", str(scenarios), "--runner", "fixture",
                            "--results", str(results), "--workers", "2",
                            "--cache", str(cache), "--report", str(report), "--format", "json",
                        ])
                self.assertEqual(status, 0)
                data = json.loads(report.read_text(encoding="utf-8"))
                stats = data["execution"]["agent_execution"]
                self.assertEqual(stats["executed"], expected_executed)
                self.assertEqual(stats["reused"], expected_reused)

    def test_parallel_execution_preserves_scenario_order(self):
        scenarios = [Scenario(name=name, input={"user": name}, intent="OK")
                     for name in ("first", "second", "third", "fourth")]
        runner = CountingRunner()
        results = execute_scenarios(scenarios, runner, max_workers=4)
        self.assertEqual([result.scenario for result in results],
                         [scenario.name for scenario in scenarios])
        self.assertGreater(runner.max_active, 1)

    def test_checkpoint_resume_and_incremental_cache_reuse_matching_scenarios(self):
        first = Scenario(name="first", input={"user": "one"}, intent="OK")
        second = Scenario(name="second", input={"user": "two"}, intent="OK")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            cache = Path(directory) / "cache.json"
            runner = CountingRunner()
            stats = {}
            execute_scenarios([first], runner, checkpoint=checkpoint, cache=cache,
                              runner_key="fixture-v1", stats=stats)
            self.assertEqual(stats["executed"], 1)
            results = execute_scenarios([first, second], runner, checkpoint=checkpoint,
                                        cache=cache, runner_key="fixture-v1", stats=stats)
            self.assertEqual([result.scenario for result in results], ["first", "second"])
            self.assertEqual(stats["reused"], 1)
            self.assertEqual(stats["executed"], 1)
            self.assertEqual(runner.calls.count("first"), 1)
            self.assertEqual(runner.calls.count("second"), 1)

    def test_cache_fingerprint_includes_scenario_and_runner_version(self):
        scenario = Scenario(name="case", input={"user": "original"}, intent="OK")
        changed = Scenario(name="case", input={"user": "changed"}, intent="OK")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            runner = CountingRunner()
            execute_scenarios([scenario], runner, cache=cache, runner_key="runner-v1")
            execute_scenarios([changed], runner, cache=cache, runner_key="runner-v1")
            execute_scenarios([changed], runner, cache=cache, runner_key="runner-v2")
        self.assertEqual(runner.calls, ["case", "case", "case"])

    def test_score_reports_first_turn_divergence_and_tool_call_location(self):
        scenario = Scenario(
            name="reschedule", input={"user": "Move the visit"}, intent="RESCHEDULE",
            turns=(
                ScenarioTurn(input={"user": "Move the visit"}, intent="IDENTIFY",
                             must_ask=("patient_identity",)),
                ScenarioTurn(input={"user": "Tomorrow"}, intent="RESCHEDULE",
                             expected_tool_calls=(ToolCallExpectation("reschedule", {"date": "tomorrow"}),)),
            ),
        )
        result = AgentResult(
            scenario="reschedule", intent="OTHER", answer="Canceled. [unknown-source]",
            turns=(
                AgentTurnResult(intent="IDENTIFY", asked=("patient_identity",)),
                AgentTurnResult(intent="OTHER", tools=("cancel",),
                                tool_calls=(ToolCall("cancel"),), answer="Canceled. [unknown-source]"),
            ),
        )
        score = score_scenario(scenario, result)
        data = score.as_dict()
        self.assertEqual(data["first_divergence"]["turn_index"], 2)
        self.assertTrue(any(item["tool_call_index"] == 1 for item in data["diagnostics"]))

    def test_evidence_failure_diagnostic_names_citation_and_text_report_shows_first_divergence(self):
        scenario = Scenario(
            name="hours", input={"user": "When do you close?"}, intent="HOURS",
            answer_graders=({"type": "evidence_grounded", "evidence": [
                {"id": "hours", "text": "The clinic closes at 5 PM."},
            ]},),
        )
        report = evaluate([scenario], {"hours": AgentResult(
            "hours", "HOURS", answer="We close at 6 PM. [bad-source]")})
        diagnostics = report.as_dict()["scenarios"][0]["diagnostics"]
        evidence = next(item for item in diagnostics if item["category"] == "evidence")
        self.assertIn("bad-source", evidence["citation_ids"])
        self.assertIn("First divergence", text_report(report))
        self.assertIn("bad-source", json_report(report))
        self.assertIn("bad-source", html_report(report))
        self.assertIn("bad-source", junit_report(report))
        self.assertIn("bad-source", sarif_report(report))


if __name__ == "__main__":
    unittest.main()
