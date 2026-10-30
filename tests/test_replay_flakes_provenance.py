import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_eval.investigation import investigate_failed_scenarios
from agent_eval.minimize import minimize_failure
from agent_eval.models import AgentResult, AgentTurnResult, Scenario, ScenarioTurn
from agent_eval.provenance import build_result_provenance, compare_result_provenance
from agent_eval.runner import execute_scenarios


class ReplayMinimizationTests(unittest.TestCase):
    def test_replays_reduced_turns_and_input_until_failure_is_minimal(self):
        scenario = Scenario(
            name="reschedule", input={"user": "hello"}, intent="RESCHEDULE",
            turns=(
                ScenarioTurn(input={"user": "hello there"}, intent="OK"),
                ScenarioTurn(input={"user": "Could you please reschedule my appointment tomorrow?"}, intent="RESCHEDULE"),
            ),
        )

        class Runner:
            def run_turn(self, scenario, user_input, history):
                text = user_input.get("user", "")
                return AgentTurnResult(intent="WRONG" if "reschedule" in text else ("RESCHEDULE" if not history else "OK"))

        runner = Runner()
        original = execute_scenarios([scenario], runner)[0]
        calls = []

        def replay(candidate):
            calls.append(candidate)
            return execute_scenarios([candidate], runner)[0]

        minimized = minimize_failure(scenario, original, replay=replay)
        reduced = minimized["minimized_scenario"]["turns"]
        self.assertTrue(minimized["replay"]["verified"])
        self.assertGreater(minimized["replay"]["attempts"], 1)
        self.assertEqual(len(reduced), 1)
        self.assertEqual(reduced[0]["input"]["user"], "reschedule")
        self.assertGreaterEqual(len(calls), 2)


class FlakeInvestigationTests(unittest.TestCase):
    def test_reruns_only_baseline_failures_and_classifies_intermitence(self):
        scenario = Scenario(name="booking", input={"user": "book"}, intent="BOOK")
        baseline = {"booking": AgentResult("booking", "OTHER")}

        class SeedRunner:
            def run_turn(self, scenario, user_input, history, execution_control=None):
                intent = "BOOK" if execution_control.seed % 2 == 0 else "OTHER"
                return AgentTurnResult(intent=intent)

        report = investigate_failed_scenarios(
            [scenario], baseline, SeedRunner(), repetitions=4, seed=10,
        )
        row = report["scenarios"][0]
        self.assertEqual(row["classification"], "intermittent")
        self.assertEqual(row["seeds"], [10, 11, 12, 13])
        self.assertEqual((row["passed_runs"], row["failed_runs"]), (2, 2))

    def test_classifies_failure_that_does_not_recur(self):
        scenario = Scenario(name="booking", input={"user": "book"}, intent="BOOK")
        report = investigate_failed_scenarios(
            [scenario], {"booking": AgentResult("booking", "OTHER")},
            type("AlwaysPass", (), {"run_turn": lambda self, scenario, user_input, history: AgentTurnResult(intent="BOOK")})(),
            repetitions=2, seed=1,
        )
        self.assertEqual(report["scenarios"][0]["classification"], "unreproduced")

    def test_investigate_cli_writes_reproducibility_classification(self):
        from contextlib import redirect_stdout
        import io
        from agent_eval.cli import main

        class SeedRunner:
            def run_turn(self, scenario, user_input, history, execution_control=None):
                return AgentTurnResult(intent="BOOK" if execution_control.seed % 2 == 0 else "OTHER")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            suite.mkdir()
            (suite / "booking.yaml").write_text(
                "scenario: booking\ninput:\n  user: book\nexpected:\n  intent: BOOK\n", encoding="utf-8")
            baseline = root / "baseline.jsonl"
            baseline.write_text('{"scenario":"booking","intent":"OTHER"}\n', encoding="utf-8")
            output = root / "investigation.json"
            with patch("agent_eval.cli.load_runner", return_value=SeedRunner()), redirect_stdout(io.StringIO()):
                status = main(["investigate", str(suite), "--responses", str(baseline), "--runner", "fixture",
                               "--repetitions", "3", "--seed", "20", "--format", "json", "--output", str(output)])
            self.assertEqual(status, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["scenarios"][0]["classification"], "intermittent")
        self.assertEqual(report["scenarios"][0]["seeds"], [20, 21, 22])

    def test_seeded_execution_exposes_seed_and_does_not_reuse_another_seed_cache(self):
        scenario = Scenario(name="booking", input={"user": "book"}, intent="BOOK")

        class SeedRunner:
            def __init__(self):
                self.seeds = []

            def run_turn(self, scenario, user_input, history, execution_control=None):
                self.seeds.append(execution_control.seed)
                return AgentTurnResult(intent="BOOK")

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            runner = SeedRunner()
            execute_scenarios([scenario], runner, cache=cache, runner_key="runner@1", seed=3)
            execute_scenarios([scenario], runner, cache=cache, runner_key="runner@1", seed=4)
            execute_scenarios([scenario], runner, cache=cache, runner_key="runner@1", seed=4)
        self.assertEqual(runner.seeds, [3, 4])


class ProvenanceTests(unittest.TestCase):
    def test_provenance_fingerprints_are_stable_and_comparison_flags_protocol_drift(self):
        scenario = Scenario(name="booking", input={"user": "book"}, intent="BOOK",
                            tool_fixtures=({"name": "lookup", "response": {"ok": True}},))
        result = AgentResult("booking", "BOOK", metadata={"model": "model-a", "prompt_version": "p1"})
        first = build_result_provenance(scenario, result, runner_key="runner@1", evaluation_config={"seed": 7})
        repeat = build_result_provenance(scenario, result, runner_key="runner@1", evaluation_config={"seed": 7})
        changed = build_result_provenance(scenario, result, runner_key="runner@1", evaluation_config={"seed": 8})
        self.assertEqual(first, repeat)
        self.assertNotEqual(first["evaluation_config_fingerprint"], changed["evaluation_config_fingerprint"])
        report = compare_result_provenance(
            {"booking": AgentResult("booking", "BOOK", metadata={"provenance": first})},
            {"booking": AgentResult("booking", "BOOK", metadata={"provenance": changed})},
        )
        self.assertFalse(report["compatible"])
        self.assertIn("evaluation_config_fingerprint", report["incompatible_fields"])

    def test_run_agent_writes_per_result_provenance_manifest(self):
        from contextlib import redirect_stdout
        import io
        from agent_eval.cli import main

        class Runner:
            version = "fixture-v1"

            def run_turn(self, scenario, user_input, history):
                return AgentTurnResult(intent=scenario.intent, metadata={"model": "fixture-model", "prompt_version": "p1"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            suite.mkdir()
            (suite / "booking.yaml").write_text(
                "scenario: booking\ninput:\n  user: book\nexpected:\n  intent: BOOK\n", encoding="utf-8")
            results = root / "results.jsonl"
            manifest = root / "provenance.json"
            bundle = root / "bundle"
            with patch("agent_eval.cli.load_runner", return_value=Runner()), redirect_stdout(io.StringIO()):
                status = main(["run-agent", str(suite), "--runner", "fixture", "--results", str(results),
                               "--seed", "4", "--provenance-manifest", str(manifest),
                               "--bundle-dir", str(bundle)])
            row = json.loads(results.read_text(encoding="utf-8"))
            self.assertEqual(status, 0)
            self.assertIn("scenario_fingerprint", row["metadata"]["provenance"])
            self.assertTrue(json.loads(manifest.read_text(encoding="utf-8"))["scenarios"]["booking"])
            self.assertTrue((bundle / "provenance-manifest.json").is_file())

    def test_compare_strict_blocks_incompatible_provenance(self):
        from agent_eval.adapters import result_as_dict
        from agent_eval.cli import main
        from contextlib import redirect_stdout
        import io

        scenario = Scenario(name="booking", input={"user": "book"}, intent="BOOK")
        result = AgentResult("booking", "BOOK")
        left = build_result_provenance(scenario, result, runner_key="runner@1", evaluation_config={"seed": 1})
        right = build_result_provenance(scenario, result, runner_key="runner@1", evaluation_config={"seed": 2})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            suite.mkdir()
            (suite / "booking.yaml").write_text(
                "scenario: booking\ninput:\n  user: book\nexpected:\n  intent: BOOK\n", encoding="utf-8")
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            for path, provenance in ((baseline, left), (candidate, right)):
                result_row = AgentResult("booking", "BOOK", metadata={"provenance": provenance})
                path.write_text(json.dumps(result_as_dict(result_row)) + "\n", encoding="utf-8")
            report_output = io.StringIO()
            with redirect_stdout(report_output):
                status = main(["compare", str(suite), "--baseline", str(baseline), "--candidate", str(candidate),
                               "--provenance-policy", "strict", "--format", "json"])
            self.assertEqual(status, 1)
            report = json.loads(report_output.getvalue())
        self.assertFalse(report["provenance"]["compatible"])


if __name__ == "__main__":
    unittest.main()
