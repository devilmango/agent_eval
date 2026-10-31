import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import yaml

from agent_eval.conformance import (check_grader_conformance, check_runner_conformance,
                                    run_conformance_matrix, _load_suite)
from agent_eval.conformance_history import (conformance_trend_report, load_conformance_history,
                                            record_conformance_report)
from agent_eval.cli import main
from agent_eval.drafts import (DraftError, create_minimized_failure_draft,
                               promote_scenario_draft, review_scenario_draft)
from agent_eval.models import AgentTurnResult, GradeResult, ToolCall, TraceEvent


class GoodRunner:
    def run_turn(self, scenario, user_input, history, execution_control=None):
        if execution_control is not None:
            execution_control.raise_if_cancelled()
        return AgentTurnResult(
            intent="HELP", asked=("identity",),
            events=(TraceEvent("question", "identity"),), answer="I can help.",
        )


class FixtureRunner:
    def run_turn(self, scenario, user_input, history, tool_sandbox=None, execution_control=None):
        if user_input.get("operation") == "wait_for_cancel":
            while True:
                execution_control.raise_if_cancelled()
        response = tool_sandbox.call("lookup", {"key": user_input["key"]})
        return AgentTurnResult(
            intent="LOOKUP", tools=("lookup",),
            tool_calls=(ToolCall("lookup", {"key": user_input["key"]}),),
            events=(TraceEvent("action", "lookup"),), answer=response["answer"],
        )


class FixtureGrader:
    def grade(self, answer, config):
        if config.get("invalid"):
            raise ValueError("invalid fixture config")
        score = 1.0 if config.get("expected") == answer else 0.0
        return GradeResult("fixture", score, score == 1.0, "checked")


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
    def test_checked_in_runner_fixture_template_has_valid_cases(self):
        suite_path = Path(__file__).parents[1] / "examples" / "conformance" / "runner-suite.yaml"
        cases = _load_suite(suite_path, "runner")
        self.assertEqual([case["name"] for case in cases], [
            "fixture-backed-lookup", "malformed-result-is-rejected", "cooperative-mid-run-cancellation",
        ])

    def report(self):
        return {
            "scenario": "appointment_reschedule",
            "target_failure_category": "intent",
            "reproduction_verified": True,
            "replay": {"enabled": True, "verified": True, "attempts": 4},
            "provenance": {"runner": "fixture", "model": "model-a", "prompt_version": "p2",
                           "seed": 7, "evaluation_config_fingerprint": "abc123"},
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
            self.assertEqual(draft["minimized_failure"]["provenance"]["runner"], "fixture")
            trace_path = output / draft["minimized_failure"]["evidence_integrity"]["trace_file"]
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

    def test_minimized_failure_evidence_hashes_block_review_and_promotion_after_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "minimized.json"
            report_path.write_text(json.dumps(self.report()), encoding="utf-8")
            output = root / "drafts"
            created = create_minimized_failure_draft(report_path, output)
            draft_path = output / created["file"]
            draft = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
            evidence = draft["minimized_failure"]["evidence_integrity"]
            (output / evidence["trace_file"]).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(DraftError, "integrity"):
                review_scenario_draft(draft_path, status="rejected", reviewer="Ada")
            with self.assertRaisesRegex(DraftError, "integrity"):
                promote_scenario_draft(draft_path, root / "scenarios")

    def test_minimized_failure_report_is_copied_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "minimized.json"
            report_path.write_text(json.dumps(self.report()), encoding="utf-8")
            output = root / "drafts"
            created = create_minimized_failure_draft(report_path, output)
            draft_path = output / created["file"]
            draft = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
            report_copy = output / draft["minimized_failure"]["evidence_integrity"]["report_file"]
            report_copy.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(DraftError, "integrity"):
                review_scenario_draft(draft_path, status="approved", reviewer="Ada")

    def test_runner_fixture_suite_checks_tool_results_expected_trace_errors_and_cancellation(self):
        suite = {
            "schema_version": 1,
            "plugin_type": "runner",
            "cases": [
                {
                    "name": "mocked lookup",
                    "scenario": {
                        "scenario": "lookup_case",
                        "input": {"user": "look up", "key": "hours"},
                        "expected": {"intent": "LOOKUP"},
                        "tool_fixtures": [{"name": "lookup", "arguments": {"key": "hours"},
                                           "response": {"answer": "Open until five"}}],
                    },
                    "expected_trace": {
                        "intent": "LOOKUP", "tools": ["lookup"],
                        "events": [{"type": "action", "name": "lookup"}],
                        "answer": "Open until five",
                    },
                },
                {
                    "name": "malformed output",
                    "scenario": {"scenario": "bad_case", "input": {"user": "bad"},
                                 "expected": {"intent": "LOOKUP"}},
                    "expect_error": {"type": "RunnerError", "contains": "AgentTurnResult"},
                },
                {
                    "name": "cancel during execution",
                    "scenario": {"scenario": "cancel_case", "input": {"operation": "wait_for_cancel"},
                                 "expected": {"intent": "LOOKUP"}},
                    "cancellation": {"after_start_ms": 10, "timeout_seconds": 1},
                },
            ],
        }

        class MalformedRunner:
            def run_turn(self, scenario, user_input, history):
                return {"unexpected": True}

        class MixedRunner(FixtureRunner):
            def run_turn(self, scenario, user_input, history, tool_sandbox=None, execution_control=None):
                if scenario.name == "bad_case":
                    return {"unexpected": True}
                return super().run_turn(scenario, user_input, history, tool_sandbox, execution_control)

        report = check_runner_conformance(MixedRunner(), suite=suite)
        self.assertTrue(report["passed"], report)
        self.assertEqual([row["status"] for row in report["fixture_cases"]], ["passed"] * 3)

    def test_grader_fixture_suite_accepts_valid_and_expected_error_cases(self):
        suite = {
            "schema_version": 1,
            "plugin_type": "grader",
            "cases": [
                {"name": "exact match", "answer": "yes", "config": {"expected": "yes"},
                 "expected_grade": {"score": 1.0, "passed": True}},
                {"name": "bad config", "answer": "yes", "config": {"invalid": True},
                 "expect_error": {"type": "ValueError", "contains": "invalid fixture config"}},
            ],
        }
        report = check_grader_conformance(FixtureGrader(), suite=suite)
        self.assertTrue(report["passed"], report)
        self.assertEqual([row["status"] for row in report["fixture_cases"]], ["passed", "passed"])

    def test_conformance_matrix_runs_targets_and_reports_runtime_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite_path = root / "suite.yaml"
            suite_path.write_text(yaml.safe_dump({
                "schema_version": 1, "plugin_type": "grader",
                "cases": [{"name": "builtin", "answer": "probe", "config": {"values": ["probe"]},
                           "expected_grade": {"passed": True}}],
            }), encoding="utf-8")
            targets_path = root / "targets.yaml"
            targets_path.write_text(yaml.safe_dump({"targets": [{"name": "current", "python": sys.executable}]}),
                                    encoding="utf-8")
            report = run_conformance_matrix(
                "grader", "contains_all", suite_path, targets_path,
            )
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["targets"][0]["python_version"].split(".")[:2],
                             [str(sys.version_info.major), str(sys.version_info.minor)])
            targets_path.write_text(yaml.safe_dump({"targets": [{"name": "wrong-version",
                                                                      "python": sys.executable,
                                                                      "agent_eval_version": "99.0"}]}),
                                    encoding="utf-8")
            mismatch = run_conformance_matrix("grader", "contains_all", suite_path, targets_path)
            self.assertFalse(mismatch["passed"])
            self.assertIn("version_errors", mismatch["targets"][0])

    def test_conformance_history_identifies_release_regression_and_intermitency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "history.jsonl"
            reports = [
                ("1.0.0", True), ("2.0.0", False), ("3.0.0", True),
            ]
            for version, passed in reports:
                report_path = root / f"{version}.json"
                report_path.write_text(json.dumps({
                    "plugin_type": "grader", "plugin": "sample-grader", "passed": passed,
                    "checks": [{"name": "determinism", "status": "passed" if passed else "failed"}],
                    "fixture_cases": [{"name": "empty-answer", "status": "passed" if passed else "failed"}],
                }), encoding="utf-8")
                record_conformance_report(report_path, store, plugin_version=version)
            rows = load_conformance_history(store)
            trend = conformance_trend_report(rows, plugin="sample-grader")
        self.assertEqual(trend["record_count"], 3)
        self.assertEqual(trend["regression_count"], 2)
        self.assertEqual(trend["intermittent_count"], 2)
        check = next(row for row in trend["checks"] if row["check"] == "determinism")
        self.assertEqual(check["first_regression_version"], "2.0.0")
        self.assertEqual(check["intermittent_since_version"], "2.0.0")
        self.assertEqual(check["current_status"], "passed")

    def test_conformance_history_cli_records_and_renders_trend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.json"
            source.write_text(json.dumps({
                "plugin_type": "runner", "plugin": "sample-runner", "passed": True,
                "checks": [{"name": "trace_structure", "status": "passed"}],
            }), encoding="utf-8")
            store = root / "nested" / "history.jsonl"
            self.assertEqual(main(["conformance-history", "record", str(source), "--store", str(store),
                                   "--plugin-version", "1.2.3"]), 0)
            output = StringIO()
            with redirect_stdout(output):
                status = main(["conformance-history", "trend", str(store), "--format", "json"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["checks"][0]["check"], "trace_structure")

    def test_drafts_verify_cli_checks_attached_report_and_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "minimized.json"
            report_path.write_text(json.dumps(self.report()), encoding="utf-8")
            output = root / "drafts"
            created = create_minimized_failure_draft(report_path, output)
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = main(["drafts", "verify", str(output / created["file"])])
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(stdout.getvalue())["verified"])

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
