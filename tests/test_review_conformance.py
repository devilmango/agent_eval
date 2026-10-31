import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import yaml

from agent_eval.conformance import check_grader_conformance, check_runner_conformance
from agent_eval.cli import main
from agent_eval.drafts import DraftError, create_minimized_failure_draft
from agent_eval.models import AgentTurnResult, GradeResult, TraceEvent


class GoodRunner:
    def run_turn(self, scenario, user_input, history, execution_control=None):
        if execution_control is not None:
            execution_control.raise_if_cancelled()
        return AgentTurnResult(
            intent="HELP", asked=("identity",),
            events=(TraceEvent("question", "identity"),), answer="I can help.",
        )


class BadRunner:
    def run_turn(self, scenario, user_input, history):
        return {"intent": "HELP"}


class GoodGrader:
    version = "1"

    def grade(self, answer, config):
        return GradeResult("fixture", 1.0, True, "ok", dict(config))


class NondeterministicGrader:
    def __init__(self):
        self.count = 0

    def grade(self, answer, config):
        self.count += 1
        return GradeResult("fixture", self.count / 2, True, "ok", dict(config))


class ReviewAndConformanceTests(unittest.TestCase):
    def report(self):
        return {
            "scenario": "appointment_reschedule",
            "target_failure_category": "intent",
            "reproduction_verified": True,
            "replay": {"enabled": True, "verified": True, "attempts": 4},
            "minimized_scenario": {
                "name": "appointment_reschedule",
                "input": {"user": "Move my appointment"},
                "intent": "RESCHEDULE",
            },
            "result": {
                "scenario": "appointment_reschedule", "intent": "CANCEL",
                "events": [{"type": "action", "name": "cancel_appointment"}],
                "turns": [{"intent": "CANCEL", "events": [{"type": "action", "name": "cancel_appointment"}]}],
            },
            "first_divergence": {"category": "intent", "expected": "RESCHEDULE", "actual": "CANCEL"},
        }

    def test_verified_minimized_failure_becomes_pending_review_draft_with_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "minimized.json"
            report_path.write_text(json.dumps(self.report()), encoding="utf-8")
            output = root / "drafts"
            created = create_minimized_failure_draft(report_path, output)
            draft = yaml.safe_load((output / created["file"]).read_text(encoding="utf-8"))
            self.assertTrue(draft["draft"])
            self.assertEqual(draft["review"]["status"], "pending")
            self.assertIsNone(draft["expected"]["intent"])
            trace_path = output / draft["minimized_failure"]["trace_file"]
            self.assertTrue(trace_path.is_file())
            self.assertEqual(json.loads(trace_path.read_text())['events'][0]['name'], "cancel_appointment")

    def test_minimized_failure_draft_cli_creates_review_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "minimized.json"
            report_path.write_text(json.dumps(self.report()), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                status = main(["draft-minimized-failure", str(report_path), "--output-dir", str(root / "drafts")])
            self.assertEqual(status, 0)
            self.assertIn("Expected behavior remains blank", output.getvalue())

    def test_minimized_failure_draft_rejects_unverified_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.report()
            report["replay"]["verified"] = False
            report_path = root / "minimized.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(DraftError):
                create_minimized_failure_draft(report_path, root / "drafts")

    def test_runner_conformance_checks_trace_cancellation_determinism_and_errors(self):
        report = check_runner_conformance(GoodRunner())
        self.assertTrue(report["passed"], report)
        checks = {check["name"]: check["status"] for check in report["checks"]}
        self.assertEqual(checks["trace_structure"], "passed")
        self.assertEqual(checks["cancellation"], "passed")
        self.assertEqual(checks["error_handling"], "passed")

    def test_bad_runner_fails_trace_conformance(self):
        report = check_runner_conformance(BadRunner())
        self.assertFalse(report["passed"])
        self.assertEqual(next(c for c in report["checks"] if c["name"] == "trace_structure")["status"], "failed")

    def test_grader_conformance_checks_result_errors_and_determinism(self):
        report = check_grader_conformance(GoodGrader(), config={"value": "x"})
        self.assertTrue(report["passed"], report)
        self.assertEqual({item["name"] for item in report["checks"]},
                         {"interface", "grade_structure", "determinism", "error_handling"})

    def test_nondeterministic_grader_fails_determinism(self):
        report = check_grader_conformance(NondeterministicGrader(), config={})
        self.assertFalse(report["passed"])
        self.assertEqual(next(c for c in report["checks"] if c["name"] == "determinism")["status"], "failed")

    def test_conformance_cli_runs_a_named_grader_and_emits_json(self):
        output = StringIO()
        with patch("agent_eval.cli.load_grader_registry", return_value={"fixture": GoodGrader()}):
            with redirect_stdout(output):
                status = main(["conformance", "grader", "fixture", "--format", "json"])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(output.getvalue())["passed"])


if __name__ == "__main__":
    unittest.main()
