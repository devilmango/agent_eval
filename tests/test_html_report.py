import unittest

from agent_eval.models import EvaluationReport, ScenarioScore
from agent_eval.report import html_report


class HtmlReportTests(unittest.TestCase):
    def test_escapes_trace_content(self):
        score = ScenarioScore(
            scenario="<img src=x onerror=alert(1)>",
            passed=False,
            expected_intent="SAFE",
            actual_intent="<script>alert(1)</script>",
            intent_correct=False,
            notes="<b>trace note</b>",
        )
        report = EvaluationReport(
            total=1,
            passed=0,
            failed=1,
            intent_accuracy=0.0,
            tool_accuracy=None,
            policy_compliance=1.0,
            unnecessary_question_rate=0.0,
            scores=(score,),
        )
        document = html_report(report)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", document)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", document)
        self.assertIn("&lt;b&gt;trace note&lt;/b&gt;", document)
        self.assertNotIn("<script>alert(1)</script>", document)

