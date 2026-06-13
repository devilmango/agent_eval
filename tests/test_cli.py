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
