import unittest

from agent_eval.graders import grade_answer, load_grader_registry
from agent_eval.models import AgentResult, Scenario
from agent_eval.scoring import evaluate


class GraderTests(unittest.TestCase):
    def test_builtin_graders_score_answers_and_affect_scenario_pass(self):
        scenario = Scenario(
            name="hours",
            input={"user": "When do you close?"},
            intent="HOURS",
            answer_graders=(
                {"type": "contains_any", "values": ["5 PM", "17:00"]},
                {"type": "not_contains", "values": ["patient record"]},
            ),
        )
        result = AgentResult(scenario="hours", intent="HOURS", answer="We close at 5 PM.")
        report = evaluate([scenario], {"hours": result})
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.answer_quality_score, 0.75)
        self.assertEqual(len(report.scores[0].answer_grades), 2)

    def test_failed_grader_fails_scenario(self):
        scenario = Scenario(
            name="hours", input={"user": "When?"}, intent="HOURS",
            answer_graders=({"type": "contains_all", "values": ["5 PM"]},),
        )
        report = evaluate([scenario], {"hours": AgentResult("hours", "HOURS", answer="I am not sure.")})
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.answer_quality_score, 0.0)

    def test_unknown_grader_is_an_actionable_error(self):
        scenario = Scenario(
            name="x", input={"user": "x"}, intent="X",
            answer_graders=({"type": "not-installed", "value": 1},),
        )
        with self.assertRaisesRegex(ValueError, "unknown answer grader"):
            grade_answer(scenario, AgentResult("x", "X", answer="x"), load_grader_registry())
