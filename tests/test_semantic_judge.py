import unittest

from agent_eval.graders import JudgeRequest, grade_answer
from agent_eval.models import AgentResult, GradeResult, Scenario


class DeterministicJudge:
    version = "fixture-2"

    def judge(self, request: JudgeRequest) -> GradeResult:
        self.request = request
        return GradeResult("fixture", 0.82, True, "The answer is complete.", metadata={"provider": "fixture"})


class SemanticJudgeTests(unittest.TestCase):
    def test_records_rubric_model_judge_version_and_rationale(self):
        scenario = Scenario(
            name="hours", input={"user": "When do you close?"}, intent="HOURS",
            answer_graders=({
                "type": "llm_judge", "judge": "fixture", "model": "judge-test",
                "threshold": 0.8, "rubric": "Check whether the answer states closing time.",
            },),
        )
        judge = DeterministicJudge()
        (grade,) = grade_answer(
            scenario, AgentResult("hours", "HOURS", answer="We close at five."),
            {}, {"fixture": judge},
        )
        self.assertTrue(grade.passed)
        self.assertEqual(grade.feedback, "The answer is complete.")
        self.assertEqual(grade.metadata["grader_version"], "fixture-2")
        self.assertEqual(grade.metadata["model"], "judge-test")
        self.assertEqual(len(grade.metadata["rubric_sha256"]), 64)
        self.assertEqual(judge.request.user_input["user"], "When do you close?")
        self.assertEqual(judge.request.conversation[0]["user"], "When do you close?")
