import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import yaml

from agent_eval.cli import main
from agent_eval.drafts import (
    DraftError, create_scenario_drafts, list_scenario_drafts,
    promote_scenario_draft, review_scenario_draft,
)
from agent_eval.scenarios import load_scenario
from agent_eval.history import HistoryError, filter_history, html_trend_report, load_history, record_history, trend_report


class HistoryTests(unittest.TestCase):
    def test_history_appends_compact_reports_and_computes_trends(self):
        report = {
            "total": 4,
            "passed": 3,
            "failed": 1,
            "metrics": {"intent_accuracy": 0.75, "tool_accuracy": 1.0,
                        "policy_compliance": 1.0, "unnecessary_question_rate": 0.1,
                        "answer_quality_score": None},
            "execution": {"cost_usd": {"total": 0.12}, "models": ["model-a"],
                          "prompt_versions": ["p1"]},
            "scenarios": [{"scenario": "must-not-be-copied"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory) / "history.jsonl"
            first = record_history(store, report, label="release-a", source_type="results")
            report["passed"] = 4
            report["failed"] = 0
            report["metrics"]["intent_accuracy"] = 1.0
            second = record_history(store, report, label="release-b", source_type="bundle", run_id="bundle-id")
            records = load_history(store)
            trend = trend_report(records)
        self.assertEqual(len(records), 2)
        self.assertEqual(first["pass_rate"], 0.75)
        self.assertEqual(second["run_id"], "bundle-id")
        self.assertEqual(trend["deltas"]["intent_accuracy"], 0.25)
        self.assertEqual(trend["latest"]["label"], "release-b")
        self.assertNotIn("scenarios", records[0])

    def test_history_rejects_corrupt_records_and_invalid_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory) / "history.jsonl"
            store.write_text("not-json\n", encoding="utf-8")
            with self.assertRaises(HistoryError):
                load_history(store)
            with self.assertRaises(HistoryError):
                trend_report([], limit=0)
            store.write_text(json.dumps({
                "schema_version": 1, "recorded_at": "2026-01-01T00:00:00+00:00",
                "total": 1, "passed": 1, "failed": 0, "metrics": {"intent_accuracy": "bad"},
            }) + "\n", encoding="utf-8")
            with self.assertRaises(HistoryError):
                load_history(store)

    def test_history_filters_by_model_prompt_label_and_date_and_renders_html(self):
        base = {
            "schema_version": 1, "run_id": "one", "recorded_at": "2026-10-13T10:00:00+00:00",
            "label": "release <one>", "total": 2, "passed": 1, "failed": 1,
            "pass_rate": 0.5, "metrics": {"intent_accuracy": 0.5},
            "models": ["model-a"], "prompt_versions": ["prompt-1"],
        }
        other = {**base, "run_id": "two", "recorded_at": "2026-10-14T10:00:00+00:00",
                 "label": "release-two", "models": ["model-b"], "prompt_versions": ["prompt-2"],
                 "passed": 2, "failed": 0, "pass_rate": 1.0,
                 "metrics": {"intent_accuracy": 1.0}}
        selected = filter_history(
            [base, other], models=["model-a"], prompt_versions=["prompt-1"],
            labels=["release <one>"], since="2026-10-13", until="2026-10-13",
        )
        self.assertEqual([item["run_id"] for item in selected], ["one"])
        report = trend_report(selected)
        html = html_trend_report(report)
        self.assertIn("<!doctype html>", html)
        self.assertIn("release &lt;one&gt;", html)
        self.assertNotIn("release <one>", html)

    def test_history_cli_writes_filtered_html(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "history.jsonl"
            record = {
                "schema_version": 1, "run_id": "filtered", "recorded_at": "2026-10-13T10:00:00+00:00",
                "label": "release", "total": 1, "passed": 1, "failed": 0,
                "pass_rate": 1.0, "metrics": {"intent_accuracy": 1.0},
                "models": ["model-a"], "prompt_versions": ["prompt-1"],
            }
            store.write_text(json.dumps(record) + "\n", encoding="utf-8")
            output = root / "history.html"
            with redirect_stdout(StringIO()):
                status = main([
                    "history", "trend", str(store), "--model", "model-a",
                    "--prompt-version", "prompt-1", "--label", "release",
                    "--since", "2026-10-13", "--until", "2026-10-13",
                    "--format", "html", "--output", str(output),
                ])
            self.assertEqual(status, 0)
            self.assertIn("Evaluation history", output.read_text(encoding="utf-8"))


class ScenarioDraftTests(unittest.TestCase):
    def test_drafts_keep_observations_separate_from_expected_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.jsonl"
            mapping_path = root / "mapping.yaml"
            output_dir = root / "drafts"
            mapping_path.write_text(
                "fields:\n  scenario: labels.scenario\n  intent: labels.intent\n"
                "  asked: observed.questions\n  tools: observed.tools\n  actions: observed.actions\n",
                encoding="utf-8",
            )
            trace_path.write_text(json.dumps({
                "labels": {"scenario": "appointments/reschedule", "intent": "RESCHEDULE"},
                "request": {"user": "Move my visit to tomorrow"},
                "observed": {"questions": ["patient_identity"], "tools": ["lookup_appointment"],
                             "actions": []},
            }) + "\n", encoding="utf-8")
            report = create_scenario_drafts(trace_path, mapping_path, "request", output_dir)
            draft = yaml.safe_load((output_dir / report["files"][0]).read_text(encoding="utf-8"))
            self.assertTrue(draft["draft"])
            self.assertEqual(draft["observed"]["intent"], "RESCHEDULE")
            self.assertIsNone(draft["expected"]["intent"])
            self.assertIsNone(draft["expected"]["must_ask"])
            with self.assertRaises(DraftError):
                create_scenario_drafts(trace_path, mapping_path, "request", output_dir)

    def test_review_detects_duplicates_and_promotes_only_approved_valid_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.yaml"
            traces = root / "traces.jsonl"
            drafts_dir = root / "drafts"
            mapping.write_text("fields:\n  scenario: scenario\n  intent: intent\n", encoding="utf-8")
            trace = {"scenario": "booking", "intent": "BOOK",
                     "input": {"user": "Book a visit"}}
            traces.write_text(json.dumps(trace) + "\n" + json.dumps(trace) + "\n", encoding="utf-8")
            create_scenario_drafts(traces, mapping, "input", drafts_dir)
            listed = list_scenario_drafts(drafts_dir)
            self.assertEqual(listed[0]["status"], "pending")
            self.assertEqual(listed[1]["duplicate_of"], listed[0]["file"])

            first = drafts_dir / listed[0]["file"]
            with self.assertRaises(DraftError):
                review_scenario_draft(first, status="approved", reviewer="reviewer")
            document = yaml.safe_load(first.read_text(encoding="utf-8"))
            document["expected"] = {"intent": "BOOK", "must_ask": [], "tools": []}
            document["must_not"] = []
            first.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            review_scenario_draft(first, status="approved", reviewer="reviewer")

            duplicate = drafts_dir / listed[1]["file"]
            second_document = yaml.safe_load(duplicate.read_text(encoding="utf-8"))
            second_document["expected"] = {"intent": "BOOK", "must_ask": [], "tools": []}
            second_document["must_not"] = []
            duplicate.write_text(yaml.safe_dump(second_document, sort_keys=False), encoding="utf-8")
            review_scenario_draft(duplicate, status="approved", reviewer="reviewer")
            with self.assertRaises(DraftError):
                promote_scenario_draft(duplicate, root / "scenarios")

            promoted = promote_scenario_draft(first, root / "scenarios")
            parsed = load_scenario(Path(promoted["path"]))
            self.assertEqual(parsed.intent, "BOOK")
            self.assertEqual(list_scenario_drafts(drafts_dir)[0]["status"], "promoted")

    def test_draft_scenarios_cli_creates_review_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.yaml"
            traces = root / "traces.jsonl"
            output = root / "drafts"
            mapping.write_text("fields:\n  scenario: scenario\n  intent: intent\n", encoding="utf-8")
            traces.write_text(json.dumps({"scenario": "booking", "intent": "BOOK",
                                          "input": {"user": "Book an appointment"}}) + "\n",
                              encoding="utf-8")
            with redirect_stdout(StringIO()):
                status = main([
                    "draft-scenarios", str(traces), "--mapping", str(mapping),
                    "--input-path", "input", "--output-dir", str(output),
                ])
            self.assertEqual(status, 0)
            self.assertEqual(len(list(output.glob("*.yaml"))), 1)

    def test_trace_redaction_is_applied_before_creating_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.yaml"
            traces = root / "traces.jsonl"
            policy = root / "redaction.yaml"
            output = root / "drafts"
            mapping.write_text("fields:\n  scenario: scenario\n  intent: intent\n", encoding="utf-8")
            traces.write_text(json.dumps({
                "scenario": "booking", "intent": "BOOK",
                "request": {"user": "Call me at 555-0100", "email": "person@example.test"},
                "observations": [{"text": "customer said sensitive words"}],
            }) + "\n", encoding="utf-8")
            policy.write_text(
                "rules:\n"
                "  - path: request.email\n    action: remove\n"
                "  - path: request.user\n    action: mask\n"
                "  - path: observations[*].text\n    action: mask\n",
                encoding="utf-8",
            )
            create_scenario_drafts(traces, mapping, "request", output, redaction_config=policy)
            draft_text = next(output.glob("*.yaml")).read_text(encoding="utf-8")
            draft = yaml.safe_load(draft_text)
            self.assertEqual(draft["input"]["user"], "[REDACTED]")
            self.assertNotIn("email", draft_text)
            self.assertNotIn("555-0100", draft_text)
            self.assertNotIn("sensitive words", draft_text)


class HistoryCliTests(unittest.TestCase):
    def test_history_cli_records_results_then_shows_trend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenarios = root / "scenarios"
            scenarios.mkdir()
            (scenarios / "booking.yaml").write_text(
                "scenario: booking\ninput:\n  user: Book an appointment\nexpected:\n  intent: BOOK\n",
                encoding="utf-8",
            )
            responses = root / "results.jsonl"
            responses.write_text(json.dumps({"scenario": "booking", "intent": "BOOK"}) + "\n",
                                 encoding="utf-8")
            store = root / "history.jsonl"
            with redirect_stdout(StringIO()):
                status = main([
                    "history", "record", "--scenarios", str(scenarios), "--responses", str(responses),
                    "--store", str(store), "--label", "ci-run", "--strict",
                ])
            self.assertEqual(status, 0)
            output = StringIO()
            with redirect_stdout(output):
                status = main(["history", "trend", str(store)])
            self.assertEqual(status, 0)
            self.assertIn("Evaluation history: 1 runs", output.getvalue())
            self.assertEqual(len(load_history(store)), 1)


if __name__ == "__main__":
    unittest.main()
