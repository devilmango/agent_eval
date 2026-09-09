import unittest

from agent_eval.comparison import compare_reports, paired_bootstrap_intervals
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

    def test_paired_bootstrap_is_reproducible_and_uses_matched_scenarios(self):
        from dataclasses import replace

        base = [ScenarioScore(
            scenario=f"s{index}", passed=index % 2 == 0, expected_intent="A",
            actual_intent="A", intent_correct=index % 2 == 0,
        ) for index in range(6)]
        candidate = [replace(score, passed=index in {0, 1, 2, 5},
                             intent_correct=index in {0, 1, 2, 5})
                     for index, score in enumerate(base)]
        baseline_report = EvaluationReport(
            total=6, passed=sum(score.passed for score in base), failed=3,
            intent_accuracy=0.5, tool_accuracy=None, policy_compliance=1,
            unnecessary_question_rate=0, scores=tuple(base),
        )
        candidate_report = EvaluationReport(
            total=6, passed=4, failed=2, intent_accuracy=4 / 6,
            tool_accuracy=None, policy_compliance=1, unnecessary_question_rate=0,
            scores=tuple(candidate),
        )
        first = paired_bootstrap_intervals(baseline_report, candidate_report, samples=500, seed=12)
        second = paired_bootstrap_intervals(baseline_report, candidate_report, samples=500, seed=12)
        self.assertEqual(first, second)
        self.assertEqual(first["matched_scenarios"], 6)
        self.assertEqual(first["metrics"]["pass_rate"]["paired_scenarios"], 6)
        self.assertLessEqual(first["metrics"]["pass_rate"]["lower"], 1 / 6)
        self.assertGreaterEqual(first["metrics"]["pass_rate"]["upper"], 1 / 6)
