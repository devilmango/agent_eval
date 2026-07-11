import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from agent_eval.cli import main


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_adapter_command_converts_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            status = main(
                [
                    "adapt",
                    str(ROOT / "examples/framework-traces.jsonl"),
                    "--mapping",
                    str(ROOT / "examples/adapter_mapping.yaml"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["scenario"], "appointment_reschedule")
            self.assertEqual(result["tools"], ["lookup_appointment"])

    def test_run_command_scores_example(self):
        with redirect_stdout(StringIO()):
            status = main(
                [
                    "run",
                    str(ROOT / "scenarios"),
                    "--responses",
                    str(ROOT / "examples/responses.jsonl"),
                    "--strict",
                ]
            )
        self.assertEqual(status, 0)

    def test_run_command_filters_scenarios_by_tag(self):
        output = StringIO()
        with redirect_stdout(output):
            status = main([
                "run", str(ROOT / "scenarios"), "--responses", str(ROOT / "examples/responses.jsonl"),
                "--tag", "faq",
            ])
        self.assertEqual(status, 0)
        self.assertIn("1 scenarios", output.getvalue())

    def test_repeat_command_enforces_confidence_aware_pass_rate_gate(self):
        with redirect_stdout(StringIO()):
            passing = main([
                "repeat", str(ROOT / "scenarios"), "--runs",
                str(ROOT / "examples/responses.jsonl"), str(ROOT / "examples/responses.jsonl"),
                "--min-runs", "2", "--min-pass-rate", "0.60",
            ])
            failing = main([
                "repeat", str(ROOT / "scenarios"), "--runs",
                str(ROOT / "examples/responses.jsonl"), str(ROOT / "examples/responses.jsonl"),
                "--min-pass-rate", "0.99",
            ])
        self.assertEqual(passing, 0)
        self.assertEqual(failing, 1)

    def test_adapt_otel_command_writes_canonical_result(self):
        document = {
            "resourceSpans": [{"scopeSpans": [{"spans": [{
                "traceId": "t", "spanId": "s", "name": "agent",
                "attributes": [
                    {"key": "gen_ai.operation.name", "value": {"stringValue": "invoke_agent"}},
                    {"key": "agent_eval.scenario", "value": {"stringValue": "booking"}},
                    {"key": "agent_eval.intent", "value": {"stringValue": "BOOK"}},
                ],
            }]}]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trace.json"
            target = Path(directory) / "results.jsonl"
            source.write_text(json.dumps(document), encoding="utf-8")
            status = main(["adapt-otel", str(source), "--output", str(target)])
            result = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual((result["scenario"], result["intent"]), ("booking", "BOOK"))

    def test_compare_command_finds_improvement(self):
        with redirect_stdout(StringIO()):
            status = main(
                [
                    "compare",
                    str(ROOT / "scenarios"),
                    "--baseline",
                    str(ROOT / "examples/responses-before.jsonl"),
                    "--candidate",
                    str(ROOT / "examples/responses.jsonl"),
                    "--strict",
                ]
            )
        self.assertEqual(status, 0)

    def test_compare_command_returns_failure_for_regression(self):
        with redirect_stdout(StringIO()):
            status = main(
                [
                    "compare",
                    str(ROOT / "scenarios"),
                    "--baseline",
                    str(ROOT / "examples/responses.jsonl"),
                    "--candidate",
                    str(ROOT / "examples/responses-before.jsonl"),
                    "--strict",
                ]
            )
        self.assertEqual(status, 1)

    def test_run_command_writes_html_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            status = main(
                [
                    "run",
                    str(ROOT / "scenarios"),
                    "--responses",
                    str(ROOT / "examples/responses.jsonl"),
                    "--format",
                    "html",
                    "--output",
                    str(output),
                    "--strict",
                ]
            )
            document = output.read_text(encoding="utf-8")
        self.assertEqual(status, 0)
        self.assertIn("<!doctype html>", document.lower())
        self.assertIn("appointment_reschedule", document)

    def test_repeat_command_writes_html_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "repeated.html"
            status = main(
                [
                    "repeat",
                    str(ROOT / "scenarios"),
                    "--runs",
                    str(ROOT / "examples/responses.jsonl"),
                    str(ROOT / "examples/responses-before.jsonl"),
                    "--format",
                    "html",
                    "--output",
                    str(output),
                ]
            )
            document = output.read_text(encoding="utf-8")
        self.assertEqual(status, 1)
        self.assertIn("Repeated-run report", document)
        self.assertIn("appointment_reschedule", document)
