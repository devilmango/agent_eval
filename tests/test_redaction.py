import json
import tempfile
import unittest
from pathlib import Path

from agent_eval.cli import main
from agent_eval.redaction import RedactionError, load_redaction_rules, redact_document


class TraceRedactionTests(unittest.TestCase):
    def test_removes_masks_and_handles_list_wildcards(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "redaction.yaml"
            policy.write_text(
                "rules:\n"
                "  - path: customer.email\n    action: remove\n"
                "  - path: calls[*].arguments.patient_id\n    action: mask\n"
                "  - path: calls[*].arguments.token\n    action: remove\n",
                encoding="utf-8",
            )
            rules = load_redaction_rules(policy)
            original = {
                "customer": {"email": "person@example.test"},
                "calls": [
                    {"arguments": {"patient_id": "P-123", "token": "secret"}},
                    {"arguments": {"patient_id": "P-456", "token": "secret-2"}},
                ],
            }
            redacted = redact_document(original, rules)
        self.assertNotIn("email", redacted["customer"])
        self.assertEqual(
            [call["arguments"]["patient_id"] for call in redacted["calls"]],
            ["[REDACTED]", "[REDACTED]"],
        )
        self.assertTrue(all("token" not in call["arguments"] for call in redacted["calls"]))
        self.assertEqual(original["customer"]["email"], "person@example.test")

    def test_rejects_invalid_redaction_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "redaction.yaml"
            policy.write_text("rules:\n  - path: request.user\n    action: scramble\n", encoding="utf-8")
            with self.assertRaises(RedactionError):
                load_redaction_rules(policy)

    def test_audit_reports_coverage_and_strictly_gates_required_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "redaction.yaml"
            traces = root / "traces.jsonl"
            report_path = root / "audit.json"
            policy.write_text(
                "rules:\n"
                "  - path: request.user\n    action: mask\n    required: true\n"
                "  - path: request.email\n    action: remove\n"
                "  - path: request.phone\n    action: remove\n",
                encoding="utf-8",
            )
            traces.write_text(
                json.dumps({"request": {"user": "first", "email": "a@example.test"}}) + "\n"
                + json.dumps({"request": {"user": "second"}}) + "\n",
                encoding="utf-8",
            )
            status = main([
                "redaction-audit", str(traces), "--config", str(policy), "--strict",
                "--format", "json", "--output", str(report_path),
            ])
            self.assertEqual(status, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["rules"][0]["matches"], 2)
            self.assertEqual(report["rules"][1]["unmatched_records"], 1)
            self.assertFalse(report["rules"][2]["matched"])
            traces.write_text(
                json.dumps({"request": {"email": "a@example.test"}}) + "\n", encoding="utf-8"
            )
            status = main([
                "redaction-audit", str(traces), "--config", str(policy), "--strict",
                "--format", "json", "--output", str(report_path),
            ])
            self.assertEqual(status, 1)
            self.assertFalse(json.loads(report_path.read_text(encoding="utf-8"))["passed"])

    def test_adapt_cli_redacts_before_writing_canonical_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.yaml"
            policy = root / "redaction.yaml"
            traces = root / "traces.jsonl"
            output = root / "results.jsonl"
            mapping.write_text(
                "fields:\n  scenario: scenario\n  intent: intent\n  answer: request.user\n",
                encoding="utf-8",
            )
            policy.write_text(
                "rules:\n  - path: request.user\n    action: mask\n"
                "    replacement: scrubbed\n",
                encoding="utf-8",
            )
            traces.write_text(json.dumps({
                "scenario": "booking", "intent": "BOOK",
                "request": {"user": "sensitive customer content"},
            }) + "\n", encoding="utf-8")
            status = main([
                "adapt", str(traces), "--mapping", str(mapping),
                "--redaction-config", str(policy), "--output", str(output),
            ])
            self.assertEqual(status, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["answer"], "scrubbed")
            self.assertNotIn("sensitive customer content", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
