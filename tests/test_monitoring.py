import json
import tempfile
import unittest
from pathlib import Path

from agent_eval.cli import main
from agent_eval.monitoring import drift_report, experiment_matrix, github_annotation_report
from agent_eval.results import load_results
from agent_eval.scenarios import load_scenarios
from agent_eval.scoring import evaluate


ROOT = Path(__file__).resolve().parents[1]


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = load_scenarios(ROOT / "scenarios")
        self.baseline_results = load_results(ROOT / "examples/responses.jsonl")
        self.baseline = evaluate(self.scenarios, self.baseline_results)

    def test_github_annotations_include_regressions_and_summary(self):
        regressed = evaluate(self.scenarios, load_results(ROOT / "examples/responses-before.jsonl"))
        comparison = {
            "baseline": {"passed": self.baseline.passed, "total": self.baseline.total},
            "candidate": {"passed": regressed.passed, "total": regressed.total},
            "regressed_scenarios": [score.scenario for score in regressed.scores if not score.passed],
            "improved_scenarios": [],
            "metric_deltas": {"intent_accuracy": -1 / 3},
        }
        commands, summary = github_annotation_report(comparison, regressed)
        self.assertIn("::error title=Scenario regression", commands)
        self.assertIn("appointment_reschedule", commands)
        self.assertIn("Agent Eval comparison", summary)
        self.assertIn("-33.3 pp", summary)

    def test_drift_gates_metrics_and_regressed_scenarios(self):
        stable = drift_report(self.baseline, self.baseline, max_absolute_drift=0)
        regressed = evaluate(self.scenarios, load_results(ROOT / "examples/responses-before.jsonl"))
        drift = drift_report(self.baseline, regressed, max_absolute_drift=0.05)
        self.assertTrue(stable["passed"])
        self.assertFalse(drift["passed"])
        self.assertIn("appointment_reschedule", drift["regressed_scenarios"])

    def test_experiment_matrix_summarizes_quality_and_execution(self):
        variant_results = dict(self.baseline_results)
        first = variant_results["appointment_reschedule"]
        from dataclasses import replace
        variant_results[first.scenario] = replace(
            first, metadata={"model": "fixture", "prompt_version": "v2", "cost_usd": 0.01,
                             "input_tokens": 12, "output_tokens": 4, "latency_ms": 80}
        )
        matrix = experiment_matrix(self.scenarios, [("baseline", self.baseline_results), ("candidate", variant_results)])
        self.assertEqual(matrix["variant_count"], 2)
        candidate = matrix["variants"][1]
        self.assertEqual(candidate["execution"]["cost_usd"], 0.01)
        self.assertEqual(candidate["execution"]["p95_latency_ms"], 80)
        self.assertEqual(candidate["metrics"]["intent_accuracy"], 1.0)

    def test_cli_monitor_and_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matrix.json"
            status = main([
                "experiment", str(ROOT / "scenarios"), "--variant",
                f"before={ROOT / 'examples/responses-before.jsonl'}", "--variant",
                f"after={ROOT / 'examples/responses.jsonl'}", "--format", "json", "--output", str(output),
            ])
            matrix = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(status, 0)
            self.assertEqual(matrix["variant_count"], 2)

            status = main([
                "monitor", str(ROOT / "scenarios"), "--baseline", str(ROOT / "examples/responses.jsonl"),
                "--current", str(ROOT / "examples/responses.jsonl"), "--format", "json", "--output", str(output),
            ])
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["passed"])


if __name__ == "__main__":
    unittest.main()
