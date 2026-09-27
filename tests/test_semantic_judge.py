import unittest
import tempfile
from pathlib import Path

from agent_eval.graders import JudgeRequest, grade_answer
from agent_eval.models import AgentResult, GradeResult, Scenario
from agent_eval.scoring import evaluate


class DeterministicJudge:
    version = "fixture-2"

    def judge(self, request: JudgeRequest) -> GradeResult:
        self.calls = getattr(self, "calls", 0) + 1
        self.request = request
        return GradeResult("fixture", 0.82, True, "The answer is complete.", metadata={"provider": "fixture"})


class SemanticJudgeTests(unittest.TestCase):
    def test_evaluation_can_explicitly_skip_semantic_judges(self):
        scenario = Scenario(
            name="hours", input={"user": "When do you close?"}, intent="HOURS",
            answer_graders=({"type": "llm_judge", "judge": "fixture", "rubric": "Check time."},),
        )
        report = evaluate([scenario], {"hours": AgentResult("hours", "HOURS", answer="five")},
                          skip_semantic_judges=True)
        self.assertTrue(report.scores[0].passed)
        self.assertIsNone(report.answer_quality_score)
        self.assertTrue(report.scores[0].answer_grades[0].metadata["skipped"])

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

    def test_repeats_and_caches_semantic_judgments(self):
        scenario = Scenario(
            name="hours", input={"user": "When do you close?"}, intent="HOURS",
            answer_graders=({"type": "llm_judge", "judge": "fixture", "rubric": "Check time."},),
        )
        judge = DeterministicJudge()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            (grade,) = grade_answer(scenario, AgentResult("hours", "HOURS", answer="five"), {},
                                    {"fixture": judge}, judge_cache=cache, judge_repetitions=3)
            self.assertEqual(judge.calls, 3)
            self.assertEqual(grade.metadata["repeated_scores"], [0.82, 0.82, 0.82])
            (cached,) = grade_answer(scenario, AgentResult("hours", "HOURS", answer="five"), {},
                                     {"fixture": judge}, judge_cache=cache, judge_repetitions=3)
            self.assertEqual(judge.calls, 3)
            self.assertTrue(cached.passed)
