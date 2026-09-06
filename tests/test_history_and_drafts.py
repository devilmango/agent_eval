import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import yaml

from agent_eval.cli import main
from agent_eval.drafts import DraftError, create_scenario_drafts
from agent_eval.history import HistoryError, load_history, record_history, trend_report


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
