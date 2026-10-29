import threading
import time
import unittest
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_eval.distributed import (
    create_shard_manifest, merge_shard_results, select_shard,
)
from agent_eval.models import (
    AgentResult, AgentTurnResult, PolicyRule, Scenario, ScenarioTurn, ToolCall,
    ToolCallExpectation, TraceEvent,
)
from agent_eval.report import html_report
from agent_eval.runner import ScenarioTimeoutError, execute_scenarios
from agent_eval.scoring import evaluate, score_scenario
from agent_eval.cli import main


class ControlledRunner:
    provider = "fixture-provider"

    def __init__(self, active=None):
        self.active = active

    def run_turn(self, scenario, user_input, history, execution_control=None):
        if self.active is not None:
            with self.active[0]:
                self.active[1][0] += 1
                self.active[1][1] = max(self.active[1][1], self.active[1][0])
            time.sleep(0.01)
            with self.active[0]:
                self.active[1][0] -= 1
        if user_input.get("user") == "wait":
            time.sleep(0.04)
            if execution_control is not None:
                execution_control.raise_if_cancelled()
        return AgentTurnResult(intent=scenario.intent)


class StopAfterFirstRunner:
    provider = "fixture-provider"

    def __init__(self, stop):
        self.stop = stop
        self.calls = 0

    def run_turn(self, scenario, user_input, history, execution_control=None):
        self.calls += 1
        self.stop.set()
        return AgentTurnResult(intent=scenario.intent)


class ExecutionScaleAndTraceTests(unittest.TestCase):
    def test_provider_limit_caps_parallel_calls_for_runners_with_same_provider(self):
        scenarios = [Scenario(name=f"s{index}", input={"user": "go"}, intent="OK")
                     for index in range(5)]
        active = [threading.Lock(), [0, 0]]
        results = execute_scenarios(
            scenarios, ControlledRunner(active), max_workers=5,
            provider_key="fixture-provider",
            runner_factory=lambda: ControlledRunner(active),
            provider_limits={"fixture-provider": 2},
        )
        self.assertEqual(len(results), 5)
        self.assertEqual(active[1][1], 2)

    def test_scenario_timeout_fails_slow_cooperative_runner(self):
        scenario = Scenario(name="slow", input={"user": "wait"}, intent="OK")
        with self.assertRaisesRegex(ScenarioTimeoutError, "slow.*timed out"):
            execute_scenarios([scenario], ControlledRunner(), scenario_timeout=0.005)

    def test_cancellation_stops_queue_and_returns_completed_scenarios(self):
        scenarios = [Scenario(name=f"s{index}", input={"user": "go"}, intent="OK")
                     for index in range(3)]
        stop = threading.Event()
        stats = {}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            results = execute_scenarios(
                scenarios, StopAfterFirstRunner(stop), max_workers=1,
                cancel_event=stop, stats=stats, checkpoint=checkpoint,
            )
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        self.assertEqual([result.scenario for result in results], ["s0"])
        self.assertEqual(stats["cancelled"], 2)
        self.assertIn("s0", saved["results"])

    def test_shards_cover_suite_once_and_merge_checks_manifest_and_duplicates(self):
        scenarios = [Scenario(name=name, input={"user": name}, intent="OK")
                     for name in ("alpha", "beta", "gamma", "delta")]
        manifest = create_shard_manifest(scenarios, shard_count=2)
        shard0 = select_shard(scenarios, manifest, shard_index=0)
        shard1 = select_shard(scenarios, manifest, shard_index=1)
        self.assertEqual({item.name for item in (*shard0, *shard1)},
                         {item.name for item in scenarios})
        self.assertFalse({item.name for item in shard0} & {item.name for item in shard1})
        results0 = [AgentResult(item.name, "OK") for item in shard0]
        results1 = [AgentResult(item.name, "OK") for item in shard1]
        merged = merge_shard_results(scenarios, manifest, [(0, results0), (1, results1)])
        self.assertEqual([item.scenario for item in merged], [item.name for item in scenarios])
        with self.assertRaisesRegex(ValueError, "duplicate shard"):
            merge_shard_results(scenarios, manifest, [(0, results0), (0, results0), (1, results1)])
        changed = [Scenario(name="alpha", input={"user": "changed"}, intent="OK"), *scenarios[1:]]
        with self.assertRaisesRegex(ValueError, "scenario set does not match"):
            select_shard(changed, manifest, shard_index=0)

    def test_merge_rejects_missing_or_wrong_scenarios(self):
        scenarios = [Scenario(name=name, input={"user": name}, intent="OK")
                     for name in ("alpha", "beta")]
        manifest = create_shard_manifest(scenarios, shard_count=2)
        with self.assertRaisesRegex(ValueError, "missing scenario"):
            merge_shard_results(scenarios, manifest, [(0, [])])
        with self.assertRaisesRegex(ValueError, "unexpected scenario"):
            merge_shard_results(scenarios, manifest, [
                (0, [AgentResult("foreign", "OK")]),
                (1, [AgentResult("beta", "OK")]),
            ])

    def test_shard_and_merge_cli_round_trip_two_worker_result_files(self):
        class Runner:
            provider = "fixture-provider"

            def run_turn(self, scenario, user_input, history):
                return AgentTurnResult(intent=scenario.intent)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario_dir = root / "suite"
            scenario_dir.mkdir()
            for name in ("alpha", "beta"):
                (scenario_dir / f"{name}.yaml").write_text(
                    f"scenario: {name}\ninput:\n  user: hi\nexpected:\n  intent: OK\n",
                    encoding="utf-8",
                )
            manifest = root / "manifest.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([
                    "shard", str(scenario_dir), "--count", "2", "--output", str(manifest),
                ]), 0)
            result_paths = []
            for index in range(2):
                result_path = root / f"shard-{index}.jsonl"
                result_paths.append(result_path)
                with patch("agent_eval.cli.load_runner", return_value=Runner()):
                    with redirect_stdout(io.StringIO()):
                        status = main([
                            "run-agent", str(scenario_dir), "--runner", "fixture",
                            "--results", str(result_path), "--shard-manifest", str(manifest),
                            "--shard-index", str(index),
                        ])
                self.assertEqual(status, 0)
            merged = root / "merged.jsonl"
            with redirect_stdout(io.StringIO()):
                status = main([
                    "merge-results", str(scenario_dir), "--manifest", str(manifest),
                    "--shard-result", f"0={result_paths[0]}",
                    "--shard-result", f"1={result_paths[1]}", "--output", str(merged),
                ])
            self.assertEqual(status, 0)
            self.assertEqual([json.loads(line)["scenario"] for line in merged.read_text().splitlines()],
                             ["alpha", "beta"])

    def test_run_agent_timeout_writes_earlier_completed_results(self):
        class Runner:
            def run_turn(self, scenario, user_input, history):
                if user_input["user"] == "slow":
                    time.sleep(0.03)
                return AgentTurnResult(intent=scenario.intent)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            suite.mkdir()
            (suite / "a_fast.yaml").write_text(
                "scenario: fast\ninput:\n  user: fast\nexpected:\n  intent: OK\n", encoding="utf-8")
            (suite / "z_slow.yaml").write_text(
                "scenario: slow\ninput:\n  user: slow\nexpected:\n  intent: OK\n", encoding="utf-8")
            results = root / "partial.jsonl"
            with patch("agent_eval.cli.load_runner", return_value=Runner()):
                with redirect_stdout(io.StringIO()):
                    status = main([
                        "run-agent", str(suite), "--runner", "fixture",
                        "--results", str(results), "--scenario-timeout", "0.005",
                    ])
            self.assertEqual(status, 1)
            self.assertEqual([json.loads(line)["scenario"] for line in results.read_text().splitlines()],
                             ["fast"])

    def test_run_agent_cancellation_writes_completed_results_and_returns_interrupt_status(self):
        class Runner:
            def run_turn(self, scenario, user_input, history, execution_control=None):
                execution_control.cancel_event.set()
                return AgentTurnResult(intent=scenario.intent)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            suite.mkdir()
            for name in ("alpha", "beta"):
                (suite / f"{name}.yaml").write_text(
                    f"scenario: {name}\ninput:\n  user: hi\nexpected:\n  intent: OK\n",
                    encoding="utf-8",
                )
            results = root / "partial.jsonl"
            with patch("agent_eval.cli.load_runner", return_value=Runner()):
                with redirect_stdout(io.StringIO()):
                    status = main([
                        "run-agent", str(suite), "--runner", "fixture",
                        "--results", str(results),
                    ])
            self.assertEqual(status, 130)
            self.assertEqual([json.loads(line)["scenario"] for line in results.read_text().splitlines()],
                             ["alpha"])

    def test_score_exposes_ordered_trace_with_first_divergence(self):
        scenario = Scenario(
            name="transfer", input={"user": "Send 10 dollars"}, intent="TRANSFER",
            turns=(ScenarioTurn(
                input={"user": "Send 10 dollars"}, intent="TRANSFER",
                must_ask=("recipient_confirmation",),
                expected_tool_calls=(ToolCallExpectation("transfer", {"amount": 10}),),
                must_not=(PolicyRule("transfer"),),
            ),),
        )
        result = AgentResult(
            scenario="transfer", intent="TRANSFER", answer="Sent. [receipt]",
            turns=(AgentTurnResult(
                intent="TRANSFER", asked=(), tool_calls=(ToolCall("transfer", {"amount": 10}),),
                actions=("transfer",), events=(TraceEvent("action", "transfer"),),
                answer="Sent. [receipt]",
            ),),
        )
        score = score_scenario(scenario, result)
        timeline = score.as_dict()["trace_timeline"]
        self.assertEqual([item["kind"] for item in timeline], [
            "user_input", "intent", "question", "tool_call", "policy_event", "answer", "citation",
        ])
        self.assertEqual(timeline[0]["observed"], {"user": "Send 10 dollars"})
        self.assertTrue(timeline[2]["first_divergence"])
        self.assertEqual(timeline[2]["expected"], "recipient_confirmation")
        self.assertIn("trace timeline", html_report(evaluate([scenario], {"transfer": result})).lower())

    def test_timeline_uses_aggregate_trace_when_saved_result_has_no_turns(self):
        scenario = Scenario(name="hours", input={"user": "hours?"}, intent="HOURS",
                            expected_tools=("lookup",))
        result = AgentResult("hours", "OTHER", tools=("lookup",), answer="Closed at 5. [hours]")
        timeline = score_scenario(scenario, result).as_dict()["trace_timeline"]
        self.assertEqual(timeline[1]["observed"], "OTHER")
        self.assertEqual(timeline[2]["observed"], "lookup")
        self.assertEqual(timeline[-1]["observed"], "hours")
        self.assertTrue(timeline[1]["first_divergence"])


if __name__ == "__main__":
    unittest.main()
