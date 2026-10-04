import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from agent_eval.calibration import calibrate_grounding
from agent_eval.cli import main
from agent_eval.graders import load_grader_registry
from agent_eval.models import AgentResult, Scenario
from agent_eval.report import junit_report, sarif_report
from agent_eval.scoring import evaluate


ROOT = Path(__file__).resolve().parents[1]


class CIReportAndGroundingTests(unittest.TestCase):
    def _report(self):
        scenarios = [
            Scenario(name="supported", input={"user": "hours"}, intent="HOURS"),
            Scenario(name="failed", input={"user": "hours"}, intent="HOURS"),
        ]
        results = {
            "supported": AgentResult("supported", "HOURS"),
            "failed": AgentResult("failed", "WRONG"),
        }
        return evaluate(scenarios, results)

    def test_junit_report_has_one_case_per_scenario_and_failure_details(self):
        xml = ET.fromstring(junit_report(self._report()))
        self.assertEqual(xml.tag, "testsuite")
        self.assertEqual(xml.attrib["tests"], "2")
        self.assertEqual(xml.attrib["failures"], "1")
        cases = {case.attrib["name"]: case for case in xml.findall("testcase")}
        self.assertIsNone(cases["supported"].find("failure"))
        self.assertIn("intent", cases["failed"].find("failure").attrib["message"])

    def test_junit_marks_scenarios_with_skipped_semantic_graders(self):
        scenario = Scenario(
            name="review", input={"user": "review"}, intent="REVIEW",
            answer_graders=({"type": "llm_judge", "judge": "remote", "rubric": "Assess answer."},),
        )
        report = evaluate([scenario], {"review": AgentResult("review", "REVIEW")},
                          skip_semantic_judges=True)
        xml = ET.fromstring(junit_report(report))
        self.assertEqual(xml.attrib["skipped"], "1")
        self.assertIsNotNone(xml.find("testcase/skipped"))

    def test_sarif_report_emits_failed_scenarios_as_ci_results(self):
        sarif = json.loads(sarif_report(self._report()))
        self.assertEqual(sarif["version"], "2.1.0")
        run = sarif["runs"][0]
        self.assertEqual(run["tool"]["driver"]["name"], "Agent Eval")
        self.assertEqual(len(run["results"]), 1)
        result = run["results"][0]
        self.assertEqual(result["level"], "error")
        self.assertEqual(result["properties"]["scenario"], "failed")
        self.assertTrue(result["locations"])

    def test_run_cli_writes_sarif_and_keeps_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.sarif"
            with redirect_stdout(StringIO()):
                status = main([
                    "run", str(ROOT / "scenarios"), "--responses",
                    str(ROOT / "examples/responses-before.jsonl"), "--strict",
                    "--format", "sarif", "--output", str(output),
                ])
            self.assertEqual(status, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["runs"][0]["results"])

    def test_run_cli_writes_junit_xml(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.xml"
            with redirect_stdout(StringIO()):
                status = main([
                    "run", str(ROOT / "scenarios"), "--responses",
                    str(ROOT / "examples/responses.jsonl"), "--strict",
                    "--format", "junit", "--output", str(output),
                ])
            self.assertEqual(status, 0)
            self.assertEqual(ET.parse(output).getroot().attrib["tests"], "3")

    def test_evidence_grounded_grader_checks_token_support_and_citations(self):
        grader = load_grader_registry()["evidence_grounded"]
        config = {
            "evidence": [{"id": "clinic-hours", "text": "The clinic closes at 5 PM."}],
            "require_citations": True,
            "min_supported_token_ratio": 0.9,
        }
        good = grader.grade("The clinic closes at 5 PM. [clinic-hours]", config)
        unsupported = grader.grade("The clinic closes at 6 PM. [clinic-hours]", config)
        uncited = grader.grade("The clinic closes at 5 PM.", config)
        wrong_source = grader.grade("The clinic closes at 5 PM. [unknown]", config)
        self.assertTrue(good.passed)
        self.assertEqual(good.score, 1.0)
        self.assertFalse(unsupported.passed)
        self.assertIn("6", unsupported.metadata["unsupported_tokens"])
        self.assertFalse(uncited.passed)
        self.assertFalse(wrong_source.passed)
        self.assertEqual(wrong_source.metadata["unknown_citations"], ["unknown"])

    def test_evidence_grader_validation_is_actionable(self):
        grader = load_grader_registry()["evidence_grounded"]
        with self.assertRaisesRegex(ValueError, "evidence"):
            grader.grade("The answer.", {"evidence": []})

    def test_evidence_grounding_grader_integrates_with_scenario_scoring(self):
        scenario = Scenario(
            name="hours", input={"user": "When do you close?"}, intent="HOURS",
            answer_graders=({
                "type": "evidence_grounded", "require_citations": True,
                "min_supported_token_ratio": 0.9,
                "evidence": [{"id": "hours", "text": "The clinic closes at 5 PM."}],
            },),
        )
        supported = evaluate([scenario], {"hours": AgentResult(
            "hours", "HOURS", answer="The clinic closes at 5 PM. [hours]")})
        unsupported = evaluate([scenario], {"hours": AgentResult(
            "hours", "HOURS", answer="The clinic closes at 6 PM. [hours]")})
        self.assertEqual(supported.passed, 1)
        self.assertEqual(unsupported.failed, 1)

    def test_grounding_calibration_requires_human_review_metadata_and_scores_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            gold = Path(directory) / "gold.yaml"
            gold.write_text("""
gold_set: reviewed-grounding
reviewed_by: reliability-team
reviewed_at: 2026-10-15
grader:
  require_citations: true
  min_supported_token_ratio: 0.9
cases:
  - id: grounded
    answer: "The clinic closes at 5 PM. [hours]"
    evidence:
      - id: hours
        text: "The clinic closes at 5 PM."
    expected_passed: true
  - id: unsupported
    answer: "The clinic closes at 6 PM. [hours]"
    evidence:
      - id: hours
        text: "The clinic closes at 5 PM."
    expected_passed: false
""", encoding="utf-8")
            report = calibrate_grounding(gold, min_accuracy=1.0)
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["confusion_matrix"]["true_positive"], 1)
        self.assertEqual(report["confusion_matrix"]["true_negative"], 1)
        self.assertTrue(report["passed_gate"])

    def test_grounding_calibration_cli_exports_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration.json"
            with redirect_stdout(StringIO()):
                status = main([
                    "calibrate-grounding", str(ROOT / "examples/evidence-grounding-gold.yaml"),
                    "--min-accuracy", "0.8", "--format", "json", "--output", str(output),
                ])
            self.assertEqual(status, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["human_reviewed"])
            self.assertGreaterEqual(report["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
