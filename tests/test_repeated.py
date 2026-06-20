import unittest

from agent_eval.models import EvaluationReport, ScenarioScore
from agent_eval.repeated import analyze_repeated_runs


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

