import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_eval.cache_tools import explain_cache, inspect_cache, prune_cache
from agent_eval.minimize import minimize_failure
from agent_eval.models import AgentResult, AgentTurnResult, PolicyRule, Scenario, ScenarioTurn, TraceEvent
from agent_eval.runner import _scenario_fingerprint, execute_scenarios
from agent_eval.cli import main
import io
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch


class CacheMaintenanceTests(unittest.TestCase):
    def test_inspect_and_explain_cache_hits_and_invalidation(self):
        scenario = Scenario(name="visit", input={"user": "move"}, intent="RESCHEDULE")
        state = {
            "format": "agent-eval-execution-state-v1",
            "runner_key": "fixture@1",
            "results": {
                "visit": {"fingerprint": _scenario_fingerprint(scenario), "updated_at": "2026-10-01T00:00:00+00:00", "result": {"scenario": "visit"}},
                "old": {"fingerprint": "stale", "updated_at": "2020-01-01T00:00:00+00:00", "result": {"scenario": "old"}},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            inspected = inspect_cache(path)
            explained = explain_cache(path, [scenario], runner_key="fixture@1")
            changed = Scenario(name="visit", input={"user": "changed"}, intent="RESCHEDULE")
            invalidated = explain_cache(path, [changed], runner_key="fixture@1")
            wrong_runner = explain_cache(path, [scenario], runner_key="fixture@2")
        self.assertEqual(inspected["entry_count"], 2)
        self.assertEqual(explained["scenarios"][0]["status"], "hit")
        self.assertTrue(explained["compatible"])
        self.assertEqual(invalidated["scenarios"][0]["reason"], "scenario fingerprint changed")
        self.assertFalse(wrong_runner["compatible"])

    def test_prune_removes_only_expired_entries_and_preserves_legacy_unknown_age(self):
        old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        state = {"format": "agent-eval-execution-state-v1", "runner_key": "fixture@1", "results": {
            "old": {"updated_at": old, "fingerprint": "a"},
            "fresh": {"updated_at": datetime.now(timezone.utc).isoformat(), "fingerprint": "b"},
            "legacy": {"fingerprint": "c"},
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            report = prune_cache(path, older_than_days=30)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["removed"], ["old"])
        self.assertEqual(set(saved["results"]), {"fresh", "legacy"})
        self.assertEqual(report["remaining_count"], 2)

    def test_execution_cache_entries_record_completion_time(self):
        class Runner:
            def run_turn(self, scenario, user_input, history):
                return AgentTurnResult(intent=scenario.intent)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            execute_scenarios([Scenario("visit", {"user": "go"}, "DONE")], Runner(), cache=path)
            entry = json.loads(path.read_text(encoding="utf-8"))["results"]["visit"]
        self.assertIsNotNone(datetime.fromisoformat(entry["updated_at"]))

    def test_cache_cli_check_and_minimize_cli_write_machine_readable_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenarios = root / "scenarios"
            scenarios.mkdir()
            (scenarios / "visit.yaml").write_text(
                "scenario: visit\ninput:\n  user: reschedule\nexpected:\n  intent: DONE\n",
                encoding="utf-8",
            )
            result_path = root / "results.jsonl"
            result_path.write_text(json.dumps({"scenario": "visit", "intent": "WRONG"}) + "\n", encoding="utf-8")
            report_path = root / "minimized.json"
            cache_path = root / "cache.json"
            cache_path.write_text(json.dumps({"format": "agent-eval-execution-state-v1",
                                              "runner_key": "fixture@unspecified", "results": {}}), encoding="utf-8")
            cache_report = root / "cache-check.json"
            with redirect_stdout(io.StringIO()):
                cache_status = main(["cache", "check", str(cache_path), str(scenarios),
                                     "--runner-key", "fixture@unspecified", "--format", "json",
                                     "--output", str(cache_report)])
            with redirect_stdout(io.StringIO()):
                status = main(["minimize", str(scenarios), "--responses", str(result_path),
                               "--scenario", "visit", "--output", str(report_path)])
            self.assertEqual(cache_status, 0)
            self.assertTrue(json.loads(cache_report.read_text(encoding="utf-8"))["compatible"])
            class Runner:
                def run_turn(self, scenario, user_input, history):
                    return AgentTurnResult(intent=scenario.intent)

            with patch("agent_eval.cli.load_runner", return_value=Runner()):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    run_status = main(["run-agent", str(scenarios), "--runner", "fixture",
                                       "--cache", str(cache_path), "--check-cache",
                                       "--results", str(root / "cached-run.jsonl")])
            self.assertEqual(run_status, 0)
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(report_path.read_text(encoding="utf-8"))["reproduction_verified"])


class FailureMinimizationTests(unittest.TestCase):
    def test_replay_minimization_records_reproduction_provenance(self):
        scenario = Scenario(name="visit", input={"user": "reschedule"}, intent="RESCHEDULE")
        baseline = AgentResult(scenario="visit", intent="CANCEL")

        def replay(candidate):
            return AgentResult(scenario=candidate.name, intent="CANCEL",
                               metadata={"model": "fixture-model", "prompt_version": "prompt-v3"})

        report = minimize_failure(
            scenario, baseline, replay=replay,
            provenance={"runner": "fixture", "runner_version": "1.4", "seed": 13,
                        "evaluation_config": {"scenario_timeout_seconds": 5}},
        )
        provenance = report["provenance"]
        self.assertEqual(provenance["runner"], "fixture")
        self.assertEqual(provenance["model"], "fixture-model")
        self.assertEqual(provenance["prompt_version"], "prompt-v3")
        self.assertEqual(provenance["seed"], 13)
        self.assertEqual(len(provenance["evaluation_config_fingerprint"]), 64)
        self.assertEqual(len(provenance["manifest_fingerprint"]), 64)

    def test_minimizer_keeps_only_turn_needed_for_original_failure(self):
        scenario = Scenario(name="visit", input={"user": "first"}, intent="DONE", turns=(
            ScenarioTurn(input={"user": "first"}, intent="ASK"),
            ScenarioTurn(input={"user": "second"}, intent="BOOK"),
            ScenarioTurn(input={"user": "third"}, intent="DONE"),
        ))
        result = AgentResult(
            scenario="visit", intent="DONE", turns=(
                AgentTurnResult(intent="ASK"),
                AgentTurnResult(intent="WRONG"),
                AgentTurnResult(intent="DONE"),
            ),
        )
        minimized = minimize_failure(scenario, result)
        self.assertTrue(minimized["reproduction_verified"])
        self.assertEqual(minimized["kept_turns"], [2])
        self.assertLess(minimized["minimized_size"]["turns"], 3)

    def test_minimizer_reduces_events_but_preserves_policy_failure(self):
        scenario = Scenario(name="cancel", input={"user": "cancel"}, intent="DONE", must_not=(PolicyRule("cancel"),))
        result = AgentResult(
            scenario="cancel", intent="DONE", actions=("cancel",),
            events=(TraceEvent("question", "identity"), TraceEvent("action", "cancel"), TraceEvent("question", "extra")),
            turns=(AgentTurnResult(intent="DONE", asked=("identity", "extra"), actions=("cancel",),
                                   events=(TraceEvent("question", "identity"), TraceEvent("action", "cancel"), TraceEvent("question", "extra"))),),
        )
        minimized = minimize_failure(scenario, result)
        self.assertTrue(minimized["reproduction_verified"])
        self.assertEqual(minimized["minimized_size"]["events"], 1)
        self.assertEqual(minimized["result"]["events"], [{"type": "action", "name": "cancel"}])

    def test_minimizer_handles_aggregate_only_legacy_trace(self):
        scenario = Scenario(name="cancel", input={"user": "cancel"}, intent="DONE",
                            must_not=(PolicyRule("cancel"),))
        result = AgentResult(scenario="cancel", intent="DONE", asked=("identity",),
                             actions=("cancel",), events=(
                                 TraceEvent("question", "identity"), TraceEvent("action", "cancel"),
                             ))
        minimized = minimize_failure(scenario, result)
        self.assertTrue(minimized["reproduction_verified"])
        self.assertEqual(minimized["original_size"]["events"], 2)
        self.assertEqual(minimized["minimized_size"]["events"], 1)


if __name__ == "__main__":
    unittest.main()
