import unittest

from agent_eval.comparison import compare_reports
from agent_eval.models import EvaluationReport, ScenarioScore


def report(passed: bool, intent: float, unnecessary: float) -> EvaluationReport:
    score = ScenarioScore(
        scenario="one",
        passed=passed,
        expected_intent="A",
        actual_intent="A" if passed else "B",
        intent_correct=passed,
    )
    return EvaluationReport(
        total=1,
        passed=int(passed),
        failed=int(not passed),
        intent_accuracy=intent,
        tool_accuracy=None,
        policy_compliance=1.0,
        unnecessary_question_rate=unnecessary,
        scores=(score,),
    )


class EvaluationComparisonTests(unittest.TestCase):
    def test_reports_improved_scenario_and_metric_deltas(self):
        comparison = compare_reports(report(False, 0.0, 0.5), report(True, 1.0, 0.0))
        self.assertEqual(comparison["improved_scenarios"], ["one"])
        self.assertEqual(comparison["regressed_scenarios"], [])
        self.assertEqual(comparison["pass_delta"], 1)
        self.assertEqual(comparison["metric_deltas"]["intent_accuracy"], 1.0)
        self.assertEqual(comparison["metric_deltas"]["unnecessary_question_rate"], -0.5)

    def test_reports_regression(self):
        comparison = compare_reports(report(True, 1.0, 0.0), report(False, 0.0, 0.5))
        self.assertEqual(comparison["regressed_scenarios"], ["one"])

