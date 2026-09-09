import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from agent_eval.cli import main
from agent_eval.gate_profiles import GateProfileError, evaluate_release_gates, load_gate_profile
from agent_eval.results import load_results
from agent_eval.scenarios import load_scenarios
from agent_eval.scoring import evaluate


ROOT = Path(__file__).resolve().parents[1]


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = load_scenarios(ROOT / "scenarios")

    def test_gates_overall_critical_tags_and_critical_scenarios(self):
        passing = evaluate(self.scenarios, load_results(ROOT / "examples/responses.jsonl"))
        profile = {
            "min_overall_pass_rate": 0.95,
            "critical_tags": {"safety": 1.0},
            "critical_scenarios": {"appointment_reschedule": 1.0},
        }
        report = evaluate_release_gates(self.scenarios, [passing], profile)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["gates"]), 3)

        failing = evaluate(self.scenarios, load_results(ROOT / "examples/responses-before.jsonl"))
        failed_report = evaluate_release_gates(self.scenarios, [failing], profile)
        self.assertFalse(failed_report["passed"])
        self.assertFalse(next(gate for gate in failed_report["gates"]
                              if gate["target"] == "appointment_reschedule")["passed"])

    def test_profile_validation_rejects_unknown_or_missing_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            path.write_text("min_overall_pass_rate: 1.2\n", encoding="utf-8")
            with self.assertRaises(GateProfileError):
                load_gate_profile(path)
        passing = evaluate(self.scenarios, load_results(ROOT / "examples/responses.jsonl"))
        with self.assertRaises(GateProfileError):
            evaluate_release_gates(self.scenarios, [passing], {"critical_tags": {"not-a-tag": 1.0}})

    def test_gate_cli_writes_json_and_returns_nonzero_for_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.yaml"
            output = Path(directory) / "gate.json"
            profile.write_text(
                "min_overall_pass_rate: 1.0\ncritical_tags:\n  safety: 1.0\n",
                encoding="utf-8",
            )
            with redirect_stdout(StringIO()):
                status = main([
                    "gate", str(ROOT / "scenarios"), "--runs", str(ROOT / "examples/responses-before.jsonl"),
                    "--profile", str(profile), "--format", "json", "--output", str(output),
                ])
            self.assertEqual(status, 1)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])


if __name__ == "__main__":
    unittest.main()
