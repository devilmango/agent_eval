import unittest

from agent_eval.models import AgentResult, EvaluationReport, Scenario, ScenarioScore
from agent_eval.repeated import analyze_repeated_runs, wilson_interval
from agent_eval.report import repeated_html_report
from agent_eval.scoring import evaluate


def make_report(passed: bool) -> EvaluationReport:
    return EvaluationReport(
        total=1,
        passed=int(passed),
        failed=int(not passed),
        intent_accuracy=float(passed),
        tool_accuracy=1.0,
        policy_compliance=1.0,
        unnecessary_question_rate=0.0,
        scores=(
            ScenarioScore(
                scenario="booking",
                passed=passed,
                expected_intent="BOOK",
                actual_intent="BOOK" if passed else "OTHER",
                intent_correct=passed,
            ),
        ),
    )


class RepeatedRunTests(unittest.TestCase):
    def test_reports_metric_variability_and_scenario_pass_frequency(self):
        result = analyze_repeated_runs([make_report(True), make_report(False)])
        self.assertEqual(result["runs"], 2)
        self.assertEqual(result["pass_rate"]["mean"], 0.5)
        self.assertAlmostEqual(result["pass_rate"]["sample_stddev"], 2**-0.5)
        self.assertEqual(result["scenarios"][0]["passed_runs"], 1)
        self.assertEqual(result["scenarios"][0]["failed_run_numbers"], [2])
        self.assertGreater(result["pass_rate"]["confidence_interval"]["lower"], 0)
        self.assertLess(result["pass_rate"]["confidence_interval"]["lower"], 0.5)

    def test_requires_at_least_two_runs(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            analyze_repeated_runs([make_report(True)])

    def test_rejects_different_scenario_sets(self):
        different = make_report(True)
        different = EvaluationReport(
            **{
                **different.__dict__,
                "scores": (
                    ScenarioScore(
                        scenario="different",
                        passed=True,
                        expected_intent="BOOK",
                        actual_intent="BOOK",
                        intent_correct=True,
                    ),
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "different scenario set"):
            analyze_repeated_runs([make_report(True), different])

    def test_perfect_results_have_a_confidence_interval_below_one(self):
        lower, upper = wilson_interval(6, 6)
        self.assertAlmostEqual(lower, 0.6097, places=3)
        self.assertEqual(upper, 1.0)

    def test_flakiness_report_groups_reasons_and_run_instability_slices(self):
        scenario = Scenario(
            name="booking", input={"user": "book"}, intent="BOOK",
            must_ask=("identity",), tags=("safety", "booking"),
        )
        passed = evaluate(
            [scenario], {"booking": AgentResult(
                "booking", "BOOK", asked=("identity",),
                metadata={"model": "model-a", "prompt_version": "prompt-a"},
            )},
        )
        failed = evaluate(
            [scenario], {"booking": AgentResult(
                "booking", "OTHER",
                metadata={"model": "model-b", "prompt_version": "prompt-b"},
            )},
        )
        report = analyze_repeated_runs([passed, failed], scenarios=[scenario])
        self.assertEqual(len(report["flaky_scenarios"]), 1)
        row = report["flaky_scenarios"][0]
        self.assertEqual(row["tags"], ["safety", "booking"])
        self.assertEqual(row["failure_reasons"], {
            "missing_required_question": 1, "wrong_intent": 1,
        })
        self.assertEqual(report["by_tag"][0]["flaky_scenarios"], ["booking"])
        self.assertEqual({item["slice"] for item in report["by_model"]}, {"model-a", "model-b"})
        self.assertEqual({item["slice"] for item in report["by_prompt_version"]}, {"prompt-a", "prompt-b"})
        html = repeated_html_report(report)
        self.assertIn("Flakiness analysis", html)
        self.assertIn("wrong_intent", html)
        self.assertIn("By model", html)
